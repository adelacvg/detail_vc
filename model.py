import logging
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.WARNING)
from einops import rearrange, repeat
import copy
import numpy as np
import time
import torch.autograd.profiler as profiler
import math
import torch
from torch import nn
from torch.nn import functional as F
from ttts.utils import commons
from ttts.vqvae import modules, attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm, spectral_norm
from ttts.utils.commons import init_weights, get_padding
from ttts.vqvae.modules import LinearNorm, Mish, Conv1dGLU
from ttts.vqvae.quantize import ResidualVectorQuantizer
from ttts.utils.vc_utils import MultiHeadAttention
from ttts.vqvae.alias_free_torch import *
from ttts.vqvae import activations, monotonic_align

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)
class MRTE(nn.Module):
    def __init__(
        self,
        text_channels=192,
        spec_channels=513,
        hidden_size=512,
        out_channels=192,
        kernel_size=5,
        n_heads=4,
        ge_layer=2,
    ):
        super(MRTE, self).__init__()
        self.cross_attention = MultiHeadAttention(hidden_size, hidden_size, n_heads)
        self.c_pre = nn.Conv1d(text_channels, hidden_size, 1)
        self.spec_pre = nn.Conv1d(spec_channels, hidden_size, 1)
        self.c_post = nn.Conv1d(hidden_size, out_channels, 1)

    def forward(self, text, text_mask, spec, spec_mask, ge):
        if ge == None:
            ge = 0
        attn_mask = spec_mask.unsqueeze(2) * text_mask.unsqueeze(-1)

        text_enc = self.c_pre(text * text_mask)
        spec_enc = self.spec_pre(spec * spec_mask)
        x = (
            self.cross_attention(
                text_enc * text_mask, spec_enc * spec_mask, attn_mask
            )
            + text_enc
            + ge
        )
        x = self.c_post(x * text_mask)
        return x

def build_word_mask(x2word, y2word):
    return (x2word[:, :, None] == y2word[:, None, :]).long()


def mel2ph_to_mel2word(mel2ph, ph2word):
    mel2word = (ph2word - 1).gather(1, (mel2ph - 1).clamp(min=0)) + 1
    mel2word = mel2word * (mel2ph > 0).long()
    return mel2word


def clip_mel2token_to_multiple(mel2token, frames_multiple):
    max_frames = mel2token.shape[1] // frames_multiple * frames_multiple
    mel2token = mel2token[:, :max_frames]
    return mel2token


def expand_states(h, mel2token):
    h = F.pad(h, [0, 0, 1, 0])
    mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
    h = torch.gather(h, 1, mel2token_)  # [B, T, H]
    return h
def group_hidden_by_segs(h, seg_ids, max_len):
    """
    :param h: [B, T, H]
    :param seg_ids: [B, T]
    :return: h_ph: [B, T_ph, H]
    """
    B, T, H = h.shape
    h_gby_segs = h.new_zeros([B, max_len + 1, H]).scatter_add_(1, seg_ids[:, :, None].repeat([1, 1, H]), h)
    all_ones = h.new_ones(h.shape[:2])
    cnt_gby_segs = h.new_zeros([B, max_len + 1]).scatter_add_(1, seg_ids, all_ones).contiguous()
    h_gby_segs = h_gby_segs[:, 1:]
    cnt_gby_segs = cnt_gby_segs[:, 1:]
    h_gby_segs = h_gby_segs / torch.clamp(cnt_gby_segs[:, :, None], min=1)
    return h_gby_segs, cnt_gby_segs

class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        gin_channels = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels
        self.gin_channels = gin_channels

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.text_embedding = nn.Embedding(4096, hidden_channels)
        nn.init.normal_(self.text_embedding.weight, 0.0, hidden_channels**-0.5)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, text, text_lengths):
        text_mask = torch.unsqueeze(
            commons.sequence_mask(text_lengths, text.size(1)), 1
        ).to(text.dtype)
        text = self.text_embedding(text)
        text = text.transpose(1, 2)
        text = self.encoder(text * text_mask, text_mask)
        # return text, text_mask
        stats = self.proj(text) * text_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return text, m, logs, text_mask

class SpecEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        sample,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        gin_channels = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.sample = sample
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels
        self.gin_channels = gin_channels

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.out_proj = nn.Conv1d(hidden_channels, out_channels, 1)
        if self.gin_channels is not None:
            self.ge_proj = nn.Linear(gin_channels,hidden_channels)
        if self.sample==True:
            self.proj = nn.Conv1d(out_channels, out_channels * 2, 1)

    def forward(self, y, y_lengths, g=None, refer=None, refer_lengths=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        if g is not None:
            y = y + self.ge_proj(g.squeeze(-1)).unsqueeze(-1)
        if refer is not None:
            y_mask2 = torch.unsqueeze(commons.sequence_mask(y_lengths+refer_lengths, y.size(2)+refer.shape[2]), 1).to(
                y.dtype
            )
            T = y.shape[-1]
            y = torch.cat([y,refer],dim=2)
            y = self.encoder(y * y_mask2, y_mask2)
            y = y[:,:,:T]
        else:
            y = self.encoder(y * y_mask, y_mask)
        y = self.out_proj(y)
        if self.sample==False:
            return y*y_mask

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        sample,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.sample = sample

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        if self.sample==True:
            self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        if self.sample == False:
            return x
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            l.remove_weight_norm()
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw

class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.enc = attentions.Encoder(
            filter_channels, filter_channels, 2,
            4, kernel_size, p_dropout
        )
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None, dur_detail=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        if dur_detail is not None:
            x = x+dur_detail
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.enc(x * x_mask, x_mask)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        prosody_channels=20,
        n_speakers=0,
        gin_channels=0,
        semantic_frame_rate=None,
        **kwargs
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.t_enc = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            6,
            kernel_size,
            p_dropout,
            latent_channels=192,
            gin_channels = None,
        )
        self.detail_enc = SpecEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            False,
            n_heads,
            4,
            kernel_size,
            p_dropout,
            latent_channels=192,
            gin_channels = gin_channels,
        )
        self.refer_feature_enc = SpecEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            False,
            n_heads,
            4,
            kernel_size,
            p_dropout,
            latent_channels=192,
        )

        self.enc_p = []
        self.enc_p.extend(
            [
                SpecEncoder(
                    inter_channels, hidden_channels, filter_channels, False, n_heads,
                4, kernel_size, p_dropout,gin_channels=gin_channels),
                SpecEncoder(
                    inter_channels, hidden_channels, filter_channels, True, n_heads,
                6, kernel_size, p_dropout,gin_channels=gin_channels),
            ]
        )
        self.spec_proj = nn.Conv1d(spec_channels, hidden_channels, 1)
        self.enc_p = nn.ModuleList(self.enc_p)
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, True,
            5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        self.ref_enc = modules.MelStyleEncoder(
            spec_channels, style_vector_dim=gin_channels
        )
        self.dur_detail_enc = SpecEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            False,
            n_heads,
            4,
            kernel_size,
            p_dropout,
            latent_channels=192,
            gin_channels = gin_channels,
        )
        self.dur_detail_emb = nn.Embedding(4096, hidden_channels)
        nn.init.normal_(self.dur_detail_emb.weight, 0.0, hidden_channels**-0.5)
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )
    def mas(self,z_p,m_p,logs_p,x_mask,y_mask):
        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )
        return attn
    def forward(self, y, y_lengths, text, text_lengths):
        y_mask = torch.unsqueeze(
            commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        ge = self.ref_enc(y * y_mask, y_mask)

        text, m_t, logs_t, text_mask = self.t_enc(text, text_lengths)

        z, m_q, logs_q = self.enc_q(y, y_lengths,ge)
        z_p = self.flow(z, y_mask, g=ge)

        attn = self.mas(z_p, m_t, logs_t, text_mask, y_mask)
        x_mask = text_mask
        w = attn.sum(2)

        durs = w.squeeze(1).long()
        dur_detail = self.dur_detail_emb(durs).transpose(1,2)
        dur_detail_ = self.dur_detail_enc(text.detach(), text_lengths, g=ge)
        l_dur_detail = torch.sum(((dur_detail - dur_detail_) ** 2)*x_mask) / torch.sum(x_mask)

        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(text, x_mask, g=ge, dur_detail=dur_detail)
        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )  # for averaging
        l_length = l_length_dp

        # expand prior
        m_t = torch.matmul(attn.squeeze(1), m_t.transpose(1, 2)).transpose(1, 2)
        logs_t = torch.matmul(attn.squeeze(1), logs_t.transpose(1, 2)).transpose(1, 2)
        text =  torch.matmul(attn.squeeze(1), text.transpose(1, 2)).transpose(1, 2)

        spec = self.spec_proj(y)*y_mask
        detail = self.enc_p[0](spec, y_lengths, g=ge)
        detail_ = self.detail_enc(text, y_lengths, g=ge)
        l_detail = torch.sum(((detail - detail_) ** 2)*y_mask) / torch.sum(y_mask)
        text = text + detail
        refer_feature = self.refer_feature_enc(spec,y_lengths)
        text, m_p, logs_p = self.enc_p[1](text,y_lengths,g=ge, refer=refer_feature,refer_lengths=y_lengths)

        quantized = text

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=ge)
        return (
            o,
            ids_slice,
            l_length,
            l_detail,
            l_dur_detail,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q, m_t, logs_t),
            quantized,
        )


    def infer(self, text, text_lengths, refer, refer_lengths, noise_scale=0.667,sdp_ratio=0,noise_scale_w=0.8,length_scale=1.0):
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
        ge = self.ref_enc(refer * refer_mask, refer_mask)
        text, m_t, logs_t, text_mask = self.t_enc(text, text_lengths)
        x_mask = text_mask
        dur_detail = self.dur_detail_enc(text, text_lengths, g=ge)

        logw = self.dp(text, x_mask, g=ge, dur_detail=dur_detail)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask).float()
        text = torch.matmul(attn.squeeze(1), text.transpose(1, 2)).transpose(1,2)

        detail = self.detail_enc(text, y_lengths, g=ge)
        text = text + detail
        spec = self.spec_proj(refer)*refer_mask
        refer_feature = self.refer_feature_enc(spec,refer_lengths)
        x, m_p, logs_p = self.enc_p[1](text, y_lengths, g=ge, refer=refer_feature,refer_lengths=refer_lengths)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=ge, reverse=True)
        o = self.dec(z, g=ge)
        return  o

    def vc(self, source, source_lengths, refer, refer_lengths, length_scale=1.0):
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
        source_mask = torch.unsqueeze(
            commons.sequence_mask(source_lengths, source.size(2)), 1).to(source.dtype)
        ge_ref = self.ref_enc(refer * refer_mask, refer_mask)
        ge_src = self.ref_enc(source * source_mask, source_mask)

        z, m_q, logs_q = self.enc_q(source, source_lengths,ge_src)
        z_p = self.flow(z, source_mask, g=ge_src)

        z = self.flow(z_p, source_mask, g=ge_ref, reverse=True)
        o = self.dec(z, g=ge_ref)
        return  o

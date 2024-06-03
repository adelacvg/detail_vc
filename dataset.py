import json
import random
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import torchaudio.functional as AuF
import torch.utils.data
import torchaudio
import torchvision
from tqdm import tqdm
import librosa
import wave
import logging
import subprocess
from glob import glob
polyglot_logger.setLevel("ERROR")
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR)

def get_duration(in_audio):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', in_audio], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

class DetailDataset(torch.utils.data.Dataset):
    def __init__(self, hps):
        paths = read_jsonl(hps.dataset.path)
        pre = os.path.expanduser(hps.dataset.pre)
        # self.paths = glob(os.path.join(hps.dataset.path, "**/*.wav"), recursive=True)
        self.paths = [os.path.join(pre,d['path']) for d in paths]
        self.texts = [d['text'] for d in paths]
        self.langs = [d['lang'] for d in paths]
        self.latins = [d['latin'] for d in paths]
        wav_lengths = [d['wav_length'] for d in paths]
        srs = [d['sr'] for d in paths]
        self.hop_length = hps.data.hop_length
        self.win_length = hps.data.win_length
        self.sampling_rate = hps.data.sampling_rate
        # lengths = [len(x) for x in self.texts]
        filtered_paths = []
        lengths=[]
        filtered_texts = []
        filtered_langs = []
        filtered_latins = []
        for path,text,lang,latin,wav_length,sr in zip(
            self.paths,self.texts,self.langs,self.latins,wav_lengths,srs):
            duration = wav_length/sr
            if duration < 10 and duration > 1 and lang in hps.dataset.lang:
                filtered_paths.append(path)
                lengths.append(duration*16000//self.hop_length)
                filtered_texts.append(text)
                filtered_langs.append(lang)
                filtered_latins.append(latin)
        self.tok_zh = VoiceBpeTokenizer('ttts/tokenizers/zh_tokenizer.json')
        self.tok_en = VoiceBpeTokenizer('ttts/tokenizers/en_tokenizer.json')
        self.tok_jp = VoiceBpeTokenizer('ttts/tokenizers/jp_tokenizer.json')
        self.tok_kr = VoiceBpeTokenizer('ttts/tokenizers/kr_tokenizer.json')
        self.paths = filtered_paths
        self.texts = filtered_texts
        self.langs = filtered_langs
        self.latins = filtered_latins
        self.lengths = lengths
        print("dataset size: ",len(lengths))

    def __getitem__(self, index):
        text = self.latins[index]
        lang = self.langs[index]
        if lang == "ZH":
            text = self.tok_zh.encode(text.lower())
        elif lang == "JP":
            text = self.tok_jp.encode(text.lower())
        elif lang == "EN":
            text = self.tok_en.encode(text.lower())
        elif lang == "KR":
            text = self.tok_kr.encode(text.lower())
        else:
            return None
        text = torch.LongTensor(text)
        if lang == "ZH":
            text = text + 256*0
            lang = 0
        elif lang == "JP":
            text = text + 256*1
            lang = 1
        elif lang == "EN":
            text = text + 256*2
            lang = 2
        elif lang == "KR":
            text = text + 256*3
            lang = 3
        else:
            return None

        wav_path = self.paths[index]
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav[0].unsqueeze(0)
        wav = AuF.resample(wav, sr, self.sampling_rate)
        wav = wav[:,:int(self.hop_length * (wav.shape[-1]//self.hop_length))]
        wav = torch.clamp(wav, min=-1.0, max=1.0)
        if wav.shape[-1]<30000:
            return None
        assert wav.shape[-1]>20480
        return  wav, text, lang

    def __len__(self):
        return len(self.paths)

class DetailCollater():
    def __init__(self):
        pass
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        assert len(batch) > 1
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(-1) for x in batch]),
            dim=0, descending=True)
        max_wav_len = max([x[0].size(1) for x in batch])
        max_text_len = max([x[1].size(0) for x in batch])

        wav_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))
        langs = torch.LongTensor(len(batch))

        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded = torch.LongTensor(len(batch), max_text_len)

        wav_padded.zero_()
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            wav = row[0]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            text = row[1]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            lang = row[2]
            langs[i] = lang
        wav_padded = wav_padded.squeeze(1)
        return {
            'wav': wav_padded,
            'wav_lengths':wav_lengths,
            'text':text_padded,
            'text_lengths':text_lengths,
            'langs':langs
        }


import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import scipy.io.wavfile as wav
from python_speech_features import mfcc

class IPADataset(Dataset):
    def __init__(self, csv_path, wav_dir, label_map, max_len=100):
        self.data = pd.read_csv(csv_path)
        self.wav_dir = wav_dir
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        rate, sig = wav.read(os.path.join(self.wav_dir, row.filename))
        mfcc_feat = mfcc(sig, rate, numcep=13)

        if mfcc_feat.shape[0] < self.max_len:
            pad = torch.zeros(self.max_len - mfcc_feat.shape[0], 13)
            mfcc_feat = torch.cat((torch.tensor(mfcc_feat), pad), 0)
        else:
            mfcc_feat = torch.tensor(mfcc_feat[:self.max_len])

        mfcc_feat = mfcc_feat.unsqueeze(0)  # Add channel dim
        label = torch.tensor(self.label_map[row.ipa_label])
        return mfcc_feat.float(), label

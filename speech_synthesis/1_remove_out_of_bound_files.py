''' Remove out-of-bound files, leveraging Waveglow and Tacotron2 scripts '''
# Waveglow mel2samp.py --> Tacotron2 layers.py --> TacotronSTFT --> mel_spectrogram
# pipeline kept throwing assertion error so removed non-complying files
# while reducing noise in the rest

import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from scipy.io.wavfile import read
import os
import noisereduce as nr
from scipy.io import wavfile


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


for file in os.listdir(filepath):
    if file not in os.listdir(destpath):
        audio, sr = load_wav_to_torch(os.path.join(filepath, file))
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        # print(torch.min(audio_norm.data))
        # print(torch.max(audio_norm.data))
        torch_min = torch.min(audio_norm.data)
        torch_max = torch.max(audio_norm.data)
        if torch_min >= -1 and torch_max <= 1:
            # perform noise reduction
            rate, data = wavfile.read(os.path.join(filepath, file))
            reduced_noise = nr.reduce_noise(y=data, sr=rate)
            wavfile.write(os.path.join(destpath, file), rate, reduced_noise)
        else:
            print('music file detected')

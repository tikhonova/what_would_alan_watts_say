'''
Changing bit depth from 32 to 16 in order for the model to run with default settings
'''
import os

import numpy as np
import scipy.io.wavfile as wav

audio_path = 'E:/AlanWatts/dataset/split_audio/'

for file in os.listdir(audio_path):
    filename = audio_path + f'{file}'
    fs, data = wav.read(filename)
    data = data.astype(np.int16)
    wav.write(f'{filename}', fs, data)
    # sampling_rate, data = read(filename)
    # audio = torch.FloatTensor(data)
    # audio_min = torch.round(torch.min(audio.data))
    # audio_max = torch.round(torch.max(audio.data))
    # audio_bit_depth = audio.dtype
    # audio_norm = audio / 32768.0
    # print(sampling_rate, torch.min(audio_norm.data), torch.max(audio_norm.data))

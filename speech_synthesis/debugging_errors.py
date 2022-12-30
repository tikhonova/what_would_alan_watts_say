''' AssertionError: Torch not compiled with CUDA enabled

Could be several reasons, one of the first things to check is whether your GPU is compatible with CUDA you're installing.

In my case, the error persisted, and what helped is
- uninstalling torch: pip uninstall torch
- purging cache: pip cache purge
- running: pip install torch -f https://download.pytorch.org/whl/torch_stable.html

as suggested in https://github.com/pytorch/pytorch/issues/30664
'''

''' AssertionError File "E:\tacotron2\layers.py", line 76, in mel_spectrogram assert(torch.min(y.data) >= -1)

This one is interesting.
I went into layers.py and started uncovering layers of code, which led me to discover that
Tacotron is using a seemingly arbitrary threshold for normalizing the waveform data, max_wav_value of 32768.0 in https://github.com/NVIDIA/tacotron2/blob/master/hparams.py,
which is in fact the maximum value that can be represented with a 16-bit signed integer.

In digital audio, the waveform of a sound is typically represented as a series of samples, each of which is a numerical value that represents the amplitude of the waveform at a particular moment in time. 
As explained by the internet, the range of possible values for each sample is determined by the bit depth of the audio, which specifies the number of bits used to represent each sample.
In the case of 16-bit audio, each sample is represented using a 16-bit signed integer, which can take on values from -32768 to 32767. 

As uncovered below, my audio has a 32 bit depth and thus a different threshold / max value it can take.
Min/Max matrix: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html  

Tacotron 2 is typically trained on 16-bit audio data, which is a common bit depth for audio data and provides a good balance between precision and file size. 
However, it is possible to train the model on audio data with a different bit depth, such as 32-bit audio, if the data is suitable for training and the model is able to process it.

What I'll do is just normalize the data to the range of a 16-bit signed integer, dividing by 2147483648 instead (i.e. updating hparams yet again).
'''

import numpy as np
import pandas as pd
import torch.utils.data
from scipy.io.wavfile import read


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


audiopaths_and_text = load_filepaths_and_text('E:/AlanWatts/dataset/filelists/audio_text_train_filelist.txt')

over1 = 0
lessthanminus1 = 0

# df = pd.DataFrame(columns=['audiopath', 'sampling_rate', 'audio', 'audio_norm'])

df = pd.DataFrame(
    columns=['audiopath', 'audio_bit_depth', 'min_audio', 'max_audio', 'min_audio_norm', 'max_audio_norm'])

for audiopath, text in audiopaths_and_text:
    sampling_rate, data = read(audiopath)
    # print(sampling_rate, data)
    audio = torch.FloatTensor(data.astype(np.float32))
    # print(audio, sampling_rate)
    # print(audio)
    # print(type(audio))
    audio_min = torch.round(torch.min(audio.data))
    audio_max = torch.round(torch.max(audio.data))
    audio_bit_depth = audio.dtype
    audio_norm = audio / 32768.0
    # print(type(audio_norm))
    # print(audio_norm)
    # audio_list = audio.tolist()
    # audio_norm_list = audio_norm.tolist()
    audio_min_norm = torch.round(torch.min(audio_norm.data))
    audio_max_norm = torch.round(torch.max(audio_norm.data))
    if audio_min_norm < -1 or audio_max_norm > 1:
        df = df.append({'audiopath': audiopath, 'audio_bit_depth': audio_bit_depth, 'min_audio': audio_min.item(),
                        'max_audio': audio_max.item(), 'min_audio_norm': audio_min_norm.item(),
                        'max_audio_norm': audio_max_norm.item()},
                       ignore_index=True)
    if audio_min_norm < -1 and audio_max_norm < 1:
        lessthanminus1 += 1
    if audio_min_norm > -1 and audio_max_norm > 1:
        over1 += 1

    # df = df.append({'audiopath': audiopath, 'sampling_rate': sampling_rate, 'audio': audio_list, 'audio_norm': audio_norm_list},
    #                ignore_index=True)

    # audio_norm = audio_norm.unsqueeze(0)
    # audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)

df.to_csv(path_or_buf='E:/AlanWatts/dataset/data_out_of_bounds.csv', sep=',')

''' RuntimeError: shape '[1, 1, 960000]' is invalid for input of size 1920000 File "E:\tacotron2\stft.py", line 84, in transform input_data = input_data.view(num_batches, 1, num_samples)

This may be caused by WAV files having more than one channel.
Solution: https://stackoverflow.com/questions/5120555/how-can-i-convert-a-wav-from-stereo-to-mono-in-python
'''

from pydub import AudioSegment
import os

filepath = 'E:/AlanWatts/dataset/split_audio/'
destpath = 'E:/AlanWatts/dataset/split_audio_upd/'
for file in os.listdir(filepath):
    sound = AudioSegment.from_wav(filepath + f'{file}')
    sound = sound.set_channels(1)
    sound.export(destpath + f'{file}', format="wav")

''' torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried setting the max_split_size_mb argument to a lower value. 

Relevant info found:
When you allocate a block of memory on the GPU, it is divided into smaller chunks, called "splits", which are managed separately by the GPU memory allocator.
If the splits are too small, it can be difficult to find a contiguous block of memory that is large enough to satisfy the allocation request, which can lead to memory allocation failures. 
On the other hand, if the splits are too large, it can lead to inefficient use of memory,
as there may be gaps between splits that are too large to be used by smaller allocation requests.
'''

# Below didn't work for me but reducing the batch size in hparams did
python - c
"import torch; torch.cuda.empty_cache()"
setx
PYTORCH_CUDA_ALLOC_CONF
"max_split_size_mb:128"
# source: https://stackoverflow.com/questions/73747731/runtimeerror-cuda-out-of-memory-how-setting-max-split-size-mb

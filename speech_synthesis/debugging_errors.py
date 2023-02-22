''' ffmpeg not found when using pydub utils
___
If using Win, need to download from the official website and add to path, then reload git bash.
https://github.com/jiaaro/pydub/issues/348
'''

''' AssertionError: Distributed mode requires CUDA
___
a MUST-read to confirm that both GPU and drivers support the CUDA version you've installed (or about to install):
https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with

Tacotron2 script was not seeing my cuda. I have a GPU RTX 2070 and CUDA toolkit 11.7.

Checks:
>> nvcc -V (to get detail re nvcc: NVIDIA (R) Cuda compiler driver)
>> nvidia-smi (to obtain GPU info)
>> conda list cudatoolkit (to see package info confirming installation)
>> python -c "import torch; print(torch.cuda.is_available())" (it printed False for me, which means PyTorch was not compiled with CUDA support,
                                                               so I had to reinstall it with the cuda flag following the steps from the next assertion error described below.)

In the end, it 'Successfully installed  +cu117'.
Note that on Windows you must also specify 'gloo' backend in hparams.py as Windows doesn't support nccl distributed computing as of the time of writing.
'''



''' AssertionError: Torch not compiled with CUDA enabled
___
Could be several reasons, one of the first things to check is whether your GPU is compatible with CUDA you're installing.

In my case, the error persisted, and what helped is
- uninstalling torch: pip uninstall torch
- purging cache: pip cache purge
- running: pip install torch -f https://download.pytorch.org/whl/torch_stable.html

as suggested in https://github.com/pytorch/pytorch/issues/30664
'''

''' torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 66.00 MiB (GPU 0; 8.00 GiB total capacity; 7.23 GiB already allocated; 0 bytes free; 
7.25 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  
See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
___
When you allocate a block of memory on the GPU, it is divided into smaller chunks, called "splits", which are managed separately by the GPU memory allocator.
If the splits are too small, it can be difficult to find a contiguous block of memory that is large enough to satisfy the allocation request, which can lead to memory allocation failures. 
On the other hand, if the splits are too large, it can lead to inefficient use of memory, as there may be gaps between splits that are too large to be used by smaller allocation requests.

There are two things that can be adjusted:
- the batch_size parameter in hparams.py determines the number of samples processed in each forward and backward pass through the model during training. 
- the max_split_size_mb parameter prevents the allocator from splitting blocks larger than the specified size (in MB), i.e. determines the max size of each chunk of data to be processed by the model.

Below is an additional argument that I've added to train.py, which I then passed in the terminal when calling train.py.
What worked for me is 8000 MiB max_split_size_mb and a batch size of 2 (sigh). UPDATE: batch size of 16 after lowering learning / decoding steps.
python -m multiproc E:/tacotron2/train.py --output_directory E:/tacotron2/checkpoints --log_directory E:/tacotron2/logdir --max-split-size-mb 8000 --warm_start
'''
parser.add_argument('--max-split-size-mb', default=256,
                    type=int, help='Maximum size of tensors that can be split')

''' AssertionError File "E:\tacotron2\layers.py", line 76, in mel_spectrogram assert(torch.min(y.data) >= -1)
___
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

What I can do is normalize the data to the range of a 16-bit signed integer, dividing by 2147483648 instead (i.e. updating hparams yet again). NB the division itself happens on line 43 of data_utils.py.
'''

# check for records that are out of bounds
# data saved in df which is then written to csv

import numpy as np
import pandas as pd
import torch.utils.data
from scipy.io.wavfile import read


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


over1 = 0
lessthanminus1 = 0
audiopaths_and_text = load_filepaths_and_text('E:/AlanWatts/dataset/filelists/audio_text_train_filelist.txt')
df = pd.DataFrame(
    columns=['audiopath', 'sampling_rate', 'audio_bit_depth', 'min_audio', 'max_audio', 'min_audio_norm',
             'max_audio_norm'])
# df = pd.DataFrame(columns=['audiopath', 'sampling_rate', 'audio', 'audio_norm'])

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
        df = df.append({'audiopath': audiopath, 'sampling_rate': sampling_rate, 'audio_bit_depth': audio_bit_depth,
                        'min_audio': audio_min.item(),
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

''' Poor alignment
___
This is a complex issue as it may have to do with a variety of factors.
I was not getting any alignment (which you can see in the first tab of your Tensorboard) no matter how I finetuned hparams.
Initial approach:
- I ran my files through a noise reduction algorithm. 
- Reduced the length of each audio clip to <15 seconds and cut silence blocks, transcribing thereafter. 
- Then I preprocessed text.
- Lastly, I changed hyper parameters, optimizing for lower attention (reducing the number of filters in the CNN, attention units and heads; see a copy of hparams in my repo, if curious).
- Shorter sentences lead to less padding, smaller model contributes to less computation and higher batch size, and therefore faster convergence.
- Some relevant info here and similar GitHub issues: https://github.com/Rayhane-mamah/Tacotron-2/issues/32
What eventually helped:
- Reducing the duration even further, down to 4-5 seconds.
- Warm starting from the Tacotron2 pretrained, publicly shared model.
- I also cut the volume of my dataset by half, just to see if it might make a positive impact on how fast the model converges.
- With 37.5 hrs of audio, audio clip 4-5 second duration, batch size of 32, and learning rate of 1e-4, I got the sweet diagonal after only 5000 steps (compare that to 170,000 steps of horizontal lines in some of my previous iterations).
- Note that this happened despite the error rate of my subtitles being higher than standard
'''

''' RuntimeError: shape '[1, 1, 960000]' is invalid for input of size 1920000 File "E:\tacotron2\stft.py", line 84, in transform input_data = input_data.view(num_batches, 1, num_samples)
___
This may be caused by WAV files having more than one channel.
Solution: https://stackoverflow.com/questions/5120555/how-can-i-convert-a-wav-from-stereo-to-mono-in-python
'''
from pydub import AudioSegment
import os

filepath = 'E:/AlanWattsMaterialSorted/split_audio/'
destpath = 'E:/AlanWattsMaterialSorted/split_audio2/'
for file in os.listdir(filepath):
    sound = AudioSegment.from_wav(filepath + f'{file}')
    sound = sound.set_channels(1)
    sound.export(destpath + f'{file}', format="wav")

''' Model may occasionally throw an assertion error:   
File "E:\tacotron2\waveglow\tacotron2\layers.py", line 75, in mel_spectrogram
assert(torch.min(y.data) >= -1)
AssertionError'''

# there is an assertion check in Tacotron's layers.py that Waveglow references as well
assert (torch.min(y.data) >= -1)
assert (torch.max(y.data) <= 1)
# they both check to see if the min/max value of tensor y is within the bounds of the audio signal that Tacotron 2 was trained on
# thus ensuring that the input audio tensor does not contain values outside the expected range

# see 1_remove_out_of_bound_files.py that removes files that fall out of bounds, while reducing noise for those that are within range

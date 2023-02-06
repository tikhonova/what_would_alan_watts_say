'''
Reduce channels prior to noise reduction if needed
I came running into numpy.core._exceptions.MemoryError: Unable to allocate 429. GiB for an array with shape (959680, 60002) and data type float64
This step helped solve for it
-- https://stackoverflow.com/questions/5120555/how-can-i-convert-a-wav-from-stereo-to-mono-in-python
from pydub import AudioSegment
import os

filepath = 'E:/AlanWattsMaterialSorted/split_audio/'
destpath = 'E:/AlanWattsMaterialSorted/split_audio2/'
for file in os.listdir(filepath):
    sound = AudioSegment.from_wav(filepath + f'{file}')
    sound = sound.set_channels(1)
    sound.export(destpath + f'{file}', format="wav")
'''

'''
Reduce noise
'''
import multiprocessing
import os
from multiprocessing import freeze_support

import noisereduce as nr
from scipy.io import wavfile

file_path = "E:/AlanWattsMaterialSorted/split_audio2/"
dest_path = "E:/AlanWattsMaterialSorted/split_audio3/"


def reduce_noise(file_path: str, file: str, dest_path: str) -> None:
    # load data
    rate, data = wavfile.read(os.path.join(file_path, file))

    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    # perform noise reduction
    try:
        wavfile.write(os.path.join(dest_path, file), rate, reduced_noise)
    except Exception():
        pass


inputs = [(file_path, file, dest_path) for file in os.listdir(file_path) if file not in os.listdir(dest_path)]
# print(inputs)

# for i, x, z in inputs:
#     reduce_noise(i, x, z)

if __name__ == '__main__':
    freeze_support()
    p = multiprocessing.Pool(8)
    p.starmap(reduce_noise, inputs)

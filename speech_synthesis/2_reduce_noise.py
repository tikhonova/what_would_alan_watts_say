'''
Reduce noise
'''
import multiprocessing
import os
from multiprocessing import freeze_support

import noisereduce as nr
from scipy.io import wavfile

file_path = "E:/AlanWatts/dataset/split_audio2/"
dest_path = "E:/AlanWatts/dataset/split_audio3/"


def reduce_noise(file_path: str, file: str, dest_path: str) -> None:
    # load data
    rate, data = wavfile.read(os.path.join(file_path, file))
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(os.path.join(dest_path, file), rate, reduced_noise)


inputs = [(file_path, file, dest_path) for file in os.listdir(file_path)]
# print(inputs)

# for i, x, z in inputs:
#     reduce_noise(i, x, z)

if __name__ == '__main__':
    freeze_support()
    p = multiprocessing.Pool(16)
    p.starmap(reduce_noise, inputs)

'''
Split audio into segments for easier inference
'''

import math
import multiprocessing
import os
from multiprocessing import freeze_support

from pydub import AudioSegment


class SplitWavAudio:

    def __init__(self, file_path, file, dest_path):
        self.file_path = file_path
        self.file = file
        self.dest_path = dest_path
        self.filepath = os.path.join(file_path, file)
        self.audio = AudioSegment.from_wav(self.filepath)

    #  self.resample = self.audio.resample(sample_rate_Hz=22050, channels=1)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 7 * 1000  # convert to milliseconds
        t2 = to_min * 7 * 1000
        split_audio = self.audio[t1:t2]
        resampled = split_audio.set_frame_rate(22050)  # change the sampling rate to 22050 Hz
        resampled.export(self.dest_path + '\\' + split_filename, format="wav")

    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 7)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.file
            self.single_split(i, i + min_per_split, split_fn)
            print(str(i) + ' done')
            if i == total_mins - min_per_split:
                print('split successfully')


def split_audio_files(file_path: str, file: str, dest_path: str) -> None:
    split_wav = SplitWavAudio(file_path, file, dest_path)
    split_wav.multiple_split(1)


file_path = "E:/AlanWattsMaterialSorted/split_audio3/"
dest_path = "E:/AlanWattsMaterialSorted/split_audio4/"

# split_audio_files('E:/AlanWattsMaterialSorted/split_audio3/', '0_03_-_Myth_and_Religion_-_Not_What_Should_Be.mp3_2022-11-19_19_52_45.wav', 'E:/AlanWattsMaterialSorted/split_audio4/')

# create a list of input arguments for the function
inputs = [(file_path, file, dest_path) for file in os.listdir(file_path)]

print(inputs)
# create a Process pool with 16 worker processes
if __name__ == '__main__':
    freeze_support()
    p = multiprocessing.Pool(16)
    p.starmap(split_audio_files, inputs)

# for i, x, z in inputs:
#     split_audio_files(i, x, z)
'''With this implementation, the worker processes will execute the convert_audio_to_wav function
with each set of arguments in the inputs list, and each worker process will convert one audio file at a time.'''

# confirm sampling rate
# file_path = "E:/AlanWatts/dataset/split_audio/"
# import os
# import wave
#
# for file_name in os.listdir(file_path):
#     print(file_name)
#     with wave.open(os.path.join(file_path, file_name), "rb") as wave_file:
#         frame_rate = wave_file.getframerate()
#         print(frame_rate)
#         break

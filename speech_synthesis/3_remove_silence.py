'''
Remove empty files, part of snippet borrowed from https://jaimeleal.github.io/how-to-speech-synthesis
'''

import os

import scipy.io.wavfile as wavfile

path = 'E:/AlanWatts/dataset/split_audio3/'
min_duration = 3
zero_dur_files = []


# removing empty audio clips
def duration(file_path):
    (source_rate, source_sig) = wavfile.read(file_path)
    duration_seconds = len(source_sig) / float(source_rate)
    return duration_seconds


# Remove files with length of less than X seconds (set in min_duration)
for index, file in enumerate(os.listdir(path)):
    if duration((os.path.join(path, file))) < min_duration:
        zero_dur_files.append(file)
        os.remove((os.path.join(path, file)))

print(len(zero_dur_files))
print(zero_dur_files)

'''
Remove silent parts
'''
import os
import multiprocessing
from multiprocessing import freeze_support
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

file_path = "E:/AlanWatts/dataset/split_audio3/"
dest_path = "E:/AlanWatts/dataset/split_audio3/"


def remove_sil(file_path: str, file: str, dest_path: str, format="wav"):
    sound = AudioSegment.from_file(os.path.join(file_path, file), format=format)
    non_sil_times = detect_nonsilent(sound, min_silence_len=50, silence_thresh=sound.dBFS * 1.5)
    if len(non_sil_times) > 0:
        non_sil_times_concat = [non_sil_times[0]]
        if len(non_sil_times) > 1:
            for t in non_sil_times[1:]:
                if t[0] - non_sil_times_concat[-1][-1] < 200:
                    non_sil_times_concat[-1][-1] = t[1]
                else:
                    non_sil_times_concat.append(t)
        non_sil_times = [t for t in non_sil_times_concat if t[1] - t[0] > 350]
        sound[non_sil_times[0][0]: non_sil_times[-1][1]].export(os.path.join(dest_path, file), format='wav')


inputs = [(file_path, file, dest_path) for file in os.listdir(file_path)]
# print(inputs)
#
# for i, x, z in inputs:
#     remove_sil(i, x, z)

if __name__ == '__main__':
    freeze_support()
    p = multiprocessing.Pool(16)
    p.starmap(remove_sil, inputs)

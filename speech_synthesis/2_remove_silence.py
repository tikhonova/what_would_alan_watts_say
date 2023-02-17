'''
Remove silent parts
'''

import multiprocessing
import os
from multiprocessing import freeze_support

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

file_path = "E:/AlanWattsMaterialSorted/split_audio4/"
dest_path = "E:/AlanWattsMaterialSorted/split_audio5/"


def remove_sil(file_path: str, file: str, dest_path: str, format="wav"):
    sound = AudioSegment.from_file(os.path.join(file_path, file), format=format)
    non_sil_times = detect_nonsilent(sound, min_silence_len=50, silence_thresh=sound.dBFS * 1.5)
    if len(non_sil_times) == 0:
        return None
    elif len(non_sil_times) > 0:
        non_sil_times_concat = [non_sil_times[0]]
        if len(non_sil_times) > 1:
            for t in non_sil_times[1:]:
                if t[0] - non_sil_times_concat[-1][-1] < 200:
                    non_sil_times_concat[-1][-1] = t[1]
                else:
                    non_sil_times_concat.append(t)
        non_sil_times = [t for t in non_sil_times_concat if t[1] - t[0] > 350]
        sound[non_sil_times[0][0]: non_sil_times[-1][1]].export(os.path.join(dest_path, file), format='wav')


# remove_sil(file_path, "8_42_Lao_Tzus_Tao_Te_Ching.mp3_2022-11-19_19_52_45.wav", dest_path,format="wav" )

inputs = [(file_path, file, dest_path) for file in os.listdir(file_path) if file not in os.listdir(dest_path)
          if file is not None]
# print(inputs)
#
# for i, x, z in inputs:
#     remove_sil(i, x, z)

if __name__ == '__main__':
    freeze_support()
    p = multiprocessing.Pool(16)
    p.starmap(remove_sil, inputs)

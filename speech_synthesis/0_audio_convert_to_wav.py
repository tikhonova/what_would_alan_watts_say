'''
Convert mp4, mp3, avi to wav
'''

import multiprocessing
import os
from multiprocessing import freeze_support

from pydub import AudioSegment

filepath = "E:/AlanWattsMaterialSorted/mp3/"
dest_path = "E:/AlanWattsMaterialSorted/audio/"


def convert_audio_to_wav(filename: str, filepath: str, dest_path: str) -> None:
    filepath = os.path.join(filepath, filename)
    dest_filepath = os.path.join(dest_path, f"{filename[:-4]}.wav")
    given_audio = AudioSegment.from_file(filepath, format="mp3")  # replace with mp4 or avi
    given_audio.export(dest_filepath, format="wav")


# create a list of input arguments for the function
inputs = [(filename, filepath, dest_path) for filename in os.listdir(filepath) if filename not in os.listdir(dest_path)]

# create a Process pool with 16 worker processes

if __name__ == '__main__':
    freeze_support()
    p = multiprocessing.Pool(16)
    p.starmap(convert_audio_to_wav, inputs)

'''With this implementation, the worker processes will execute the convert_audio_to_wav function 
with each set of arguments in the inputs list, and each worker process will convert one audio file at a time.'''

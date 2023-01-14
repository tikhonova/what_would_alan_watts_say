'''
Create a df list with filenames and durations
'''

import pandas as pd

audio_path = 'E:/AlanWatts/dataset/split_audio/'


def create_duration_df(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    audio_data = []

    for file in onlyfiles:
        audio = AudioSegment.from_file(join(path, file))
        duration = len(audio)
        # convert the duration from miliseconds to seconds
        duration_seconds = duration / 1000
        audio_data.append({"filename": file, "duration": round(duration_seconds)})

    df = pd.DataFrame(audio_data)
    df = df.groupby(['duration']).size().reset_index(name='Counts')
    # return df
    df.to_csv("E:/AlanWatts/dataset/list.csv", index=False, header=False, sep=',', mode='a')


create_duration_df(audio_path)
# df.head(5)

'''
Move files with 13-15 sec duration to a separate directory
'''
import os
from pydub import AudioSegment
from os import listdir
from os.path import isfile, join

audio_path = 'E:/AlanWatts/dataset/split_audio/'


def sort_files_by_duration(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    files_folder = join(path, "files")
    if not os.path.exists(files_folder):
        os.makedirs(files_folder)

    for file in onlyfiles:
        audio = AudioSegment.from_file(join(path, file))
        duration = round(len(audio) / 1000)
        if duration in [13, 14, 15]:
            os.rename(join(path, file), join(files_folder, file))


# call the function
sort_files_by_duration(audio_path)

import os

import jiwer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

''' transcript text cleanup '''

filepath = 'E:/AlanWatts/dataset/transcripts/'

for file in os.listdir(filepath):
    file_modified = False
    if file_modified:
        break
    filename = filepath + f'{file}'
    print(filename)
    file_contents = open(filename, "r", encoding="utf-8").readlines()
    print(file_contents)
    transformed = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ReduceToSingleSentence()
    ])(file_contents)
    print(transformed)
    f = open(filename, "w", encoding="utf-8")
    f.write(''.join(transformed))
    f.close()
    file_modified = True

''' Make metadata.csv and filelists via https://jaimeleal.github.io/how-to-speech-synthesis '''

filepath = 'E:/AlanWatts/dataset/transcripts/'

files = os.listdir(filepath)
rows = []
# it = 0
for file in files:
    filename = filepath + f'{file}'
    # it += 1
    # if it <= 2:
    file_contents = open(filename, "r", encoding="utf-8").readlines()
    rows.append([file[:-4], ''.join(file_contents)])

print(rows)

df = pd.DataFrame(rows, columns=["name", "transcript"])

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df["wav_path"] = df["name"].apply("E:/AlanWatts/dataset/split_audio/{}.wav".format)
df["mel_path"] = df["name"].apply("E:/AlanWatts/dataset/mels/{}.pt".format)

# Add new columns
df["metadata"] = df["name"] + "|" + df[
    "transcript"]  # see Tacotron2 documentation reference `<audio file path>|<transcript>`  https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2#getting-the-data
df["wav_text"] = df["wav_path"] + "|" + df["transcript"]
df["mel_text"] = df["mel_path"] + "|" + df["transcript"]

# Split files intro training, testing, and validation
train, test = train_test_split(df, test_size=0.2, random_state=1)
test, val = train_test_split(test, test_size=0.05, random_state=1)

metadata = df["metadata"]
audio_text_test_filelist = test["wav_text"]
audio_text_train_filelist = train["wav_text"]
audio_text_val_filelist = val["wav_text"]
mel_text_test_filelist = test["mel_text"]
mel_text_train_filelist = train["mel_text"]
mel_text_val_filelist = val["mel_text"]

metadata.to_csv("dataset/metadata.csv", index=False)
np.savetxt("dataset/filelists/audio_text_test_filelist.txt", audio_text_test_filelist.values, fmt="%s")
np.savetxt("dataset/filelists/audio_text_train_filelist.txt", audio_text_train_filelist.values, fmt="%s")
np.savetxt("dataset/filelists/audio_text_val_filelist.txt", audio_text_val_filelist.values, fmt="%s")
np.savetxt("dataset/filelists/mel_text_test_filelist.txt", mel_text_test_filelist.values, fmt="%s")
np.savetxt("dataset/filelists/mel_text_train_filelist.txt", mel_text_train_filelist.values, fmt="%s")
np.savetxt("dataset/filelists/mel_text_val_filelist.txt", mel_text_val_filelist.values, fmt="%s")

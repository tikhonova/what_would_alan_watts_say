'''remove silent parts'''
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def remove_sil(path_in, path_out, format="wav"):
    sound = AudioSegment.from_file(path_in, format=format)
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
        sound[non_sil_times[0][0]: non_sil_times[-1][1]].export(path_out, format='wav')


''' Removing empty files, part of snippet borrowed from https://jaimeleal.github.io/how-to-speech-synthesis '''
import os
import scipy.io.wavfile as wavfile

min_duration = 1
path = 'E:/AlanWatts/dataset/split_audio/'
zero_dur_files = []


# removing empty audio clips
def duration(file_path):
    (source_rate, source_sig) = wavfile.read(file_path)
    duration_seconds = len(source_sig) / float(source_rate)
    return duration_seconds


# Remove files with length of less than 1 second
for index, file in enumerate(os.listdir(path)):
    if duration((os.path.join(path, file))) < min_duration:
        zero_dur_files.append(file)
        os.remove((os.path.join(path, file)))

print(len(zero_dur_files))
print(zero_dur_files)

# removing empty text files
import os

path = 'E:/AlanWatts/dataset/transcripts/'
min_length = 1
blank_files = []

for file in os.listdir(path):
    filename = path + f'{file}'
    contents = open(filename).readlines()
    if len(contents) < min_length:
        blank_files.append(file)
        os.remove(filename)

print(len(blank_files))
print(blank_files)

# leaving only those audios and transcripts that have a corresponding match
audio_path = 'E:/AlanWatts/dataset/split_audio/'
transcript_path = 'E:/AlanWatts/dataset/transcripts/'

audio_files = [file[:-4] for file in os.listdir(audio_path)]
transcripts = [file[:-4] for file in os.listdir(transcript_path)]

counter = 0
for file in transcripts:
    if file not in audio_files:
        print(file)
        counter += 1
        os.remove((os.path.join(transcript_path, f"{file}.txt")))

counter = 0
for file in audio_files:
    if file not in transcripts:
        print(file)
        counter += 1
        os.remove((os.path.join(audio_path, f"{file}.wav")))

''' transcript text cleanup '''
import os
import jiwer

filepath = 'E:/AlanWatts/dataset/transcripts/'

for file in os.listdir(filepath):
    file_modified = False
    if file_modified:
        break
    filename = filepath + f'{file}'
    # print(filename)
    file_contents = open(filename, "r", encoding="utf-8").readlines()
    # print(file_contents)
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
    #   print(transformed)
    f = open(filename, "w", encoding="utf-8")
    f.write(''.join(transformed))
    f.close()
    file_modified = True

''' Make metadata.csv and filelists via https://jaimeleal.github.io/how-to-speech-synthesis '''
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

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

# print(rows)

df = pd.DataFrame(rows, columns=["name", "transcript"])

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df["wav_path"] = df["name"].apply("E:/AlanWatts/dataset/split_audio/{}.wav".format)

# Add new columns
df["metadata"] = df["name"] + "|" + df[
    "transcript"]  # see Tacotron2 documentation reference `<audio file path>|<transcript>`  https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2#getting-the-data
df["wav_text"] = df["wav_path"] + "|" + df["transcript"]

# Split files intro training, testing, and validation
train, test = train_test_split(df, test_size=0.1, random_state=1)
test, val = train_test_split(test, test_size=0.05, random_state=1)

metadata = df["metadata"]
audio_text_test_filelist = test["wav_text"]
audio_text_train_filelist = train["wav_text"]
audio_text_val_filelist = val["wav_text"]

metadata.to_csv("E:/AlanWatts/dataset/metadata.csv", index=False)
np.savetxt("E:/AlanWatts/dataset/filelists/audio_text_test_filelist.txt", audio_text_test_filelist.values, fmt="%s")
np.savetxt("E:/AlanWatts/dataset/filelists/audio_text_train_filelist.txt", audio_text_train_filelist.values, fmt="%s")
np.savetxt("E:/AlanWatts/dataset/filelists/audio_text_val_filelist.txt", audio_text_val_filelist.values, fmt="%s")

''' Meta filelist for Waveglow '''
import os
import pandas as pd

filepath = 'E:/AlanWatts/dataset/split_audio/'
files = os.listdir(filepath)
rows = []
# it = 0
for file in files:
    filename = filepath + f'{file}'
    rows.append(filename)

df = pd.DataFrame(rows)
df.to_csv("E:/AlanWatts/dataset/waveglow.txt", index=False, header=False, sep='\t', mode='a')

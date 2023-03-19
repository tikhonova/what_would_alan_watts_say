"""
https://cloud.google.com/speech-to-text/docs/sync-recognize
https://github.com/GoogleCloudPlatform/python-docs-samples/blob/HEAD/speech/snippets/transcribe.py

 16000 Google provided sample transcribed with no issue
 testing if converted to 20050
 import librosa
 import soundfile
 y, s = librosa.load('E:/AlanWatts/dataset/test/sample.wav', sr=22050)
 soundfile.write('E:/AlanWatts/dataset/test/sample_upd.wav', y, s)

 ffprobe sample_upd.wav -- Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 22050 Hz, 1 channels, s16, 352 kb/s
 ffprobe my_file.wav    -- Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 22050 Hz, 1 channels, s16, 352 kb/s
"""
import multiprocessing
import os
from multiprocessing import freeze_support

import pandas as pd
from google.cloud import speech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'E:/AlanWatts/dataset/keystt.json'  # setting os environment credentials
filepath = 'E:/AlanWatts/dataset/split_audio2/'
dest_path = 'E:/AlanWatts/dataset/transcripts3/'

# instantiate a client
client = speech.SpeechClient()
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=22050,
    audio_channel_count=1,
    model='phone_call',  # recognizes low quality audio better than default
    use_enhanced=1,  # if available
    language_code="en-US")

# create a list of arguments to pass to the CLI script
inputs = [(filename, filepath, dest_path) for filename in os.listdir(filepath) if
          f'{filename[:-4]}.txt' not in os.listdir(dest_path)]


# print(inputs)


def transcribe_audio(filename: str, filepath: str, dest_path: str) -> None:
    filepath = os.path.join(filepath, filename)
    dest_filepath = os.path.join(dest_path, f"{filename[:-4]}.txt")
    with open(filepath, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)

    # detects speech in the audio file
    response = client.recognize(config=config, audio=audio)
    # print(response.total_billed_time, 'total billed time')
    # print(response.results, 'result')

    transcript = []
    confidence = []

    for result in response.results:
        alternative = result.alternatives[0]
        # print('Transcript: {}'.format(alternative.transcript))
        # print('Confidence: {}'.format(alternative.confidence))
        transcript.append(
            alternative.transcript)  # appending because there are occasionally several responses per audio, depending on the number/length of pauses
        confidence.append(alternative.confidence)
    # print(transcript)
    # print(confidence)
    if len(confidence) > 0:
        if min(confidence) >= 0.44:  # 44% confidence made sense for my sample
            textfile = open(dest_filepath, 'w')
            transcript_concat = ' '.join(transcript)
            textfile.write(transcript_concat)
            textfile.close()
            df = pd.DataFrame(columns=[[f'{filename}', 'confidence', 'transcript']],
                              data=[[f'{filename}', confidence, transcript_concat]])
            df.to_csv(dest_path + 'list_of_transcripts.csv', mode='a', header=False)


# for filename, filepath, dest_path in inputs:
#    transcribe_audio(filename, filepath, dest_path)

# create a Process pool with 16 worker processes
if __name__ == '__main__':
    freeze_support()
    p = multiprocessing.Pool(16)
    p.starmap(transcribe_audio, inputs)

''' error rate analysis '''
import jiwer

with open("E:/AlanWatts/dataset/ground_truth.txt") as f:
    ground_truth = f.readlines()

with open("E:/AlanWatts/dataset/transcripts2/6_5_Zen_Mind_for_Beginners_I.mp3_2022-11-19_19_52_45.txt") as f:
    hypothesis = f.readlines()

transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ReduceToSingleSentence(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])  # (hypothesis)

wer = jiwer.wer(ground_truth,
                hypothesis,
                truth_transform=transformation,
                hypothesis_transform=transformation)

print(wer)
# 0-20% word error rate based on just a couple clips.
# for comparison, Autosub often returned completely incorrect or blank transcriptions
# and the default Amazon Transcribe transcript showed a Word Error Rate (WER) of 31.87%.

'''
Convert mp4, mp3, avi to wav

How to start redis server on Win 10:
1.Run in WSL: sudo service redis-server start
2.Run Celery in terminal: celery -A convert_to_wav_and_transform.celery worker --loglevel=debug -P eventlet --concurrency=100
3.Run flower UI via localhost: celery -A convert_to_wav_and_transform.celery flower --port=5555

'''

import os

from celery import Celery
from pydub import AudioSegment

celery = Celery('convert_audio',
                broker='redis://localhost:6379/0')  # app object initialized with Celery constructor and config params i.e. broker URL

filepath = "E:/AlanWattsMaterialSorted/mp3/"
dest_path = "E:/AlanWattsMaterialSorted/audio/"


# define Celery task for converting audio files to WAV format
@celery.task
def convert_audio_to_wav(filename, filepath, dest_path):
    given_audio = AudioSegment.from_file(f"{filepath}" + f"{filename}", format="mp3")  # replace with mp4 or avi
    given_audio.export(f"{dest_path}" + f"{filename[:-4]}.wav", format="wav")


for filename in os.listdir(filepath):
    if filename not in os.listdir(dest_path):
        convert_audio_to_wav.delay(filename, filepath,
                                   dest_path)  # .delay adds task to Celery queue allowing multiple audio files to be converted concurrently

'''
Convert mp4, mp3, avi to wav
'''

# import os
#
# from pydub import AudioSegment
# from tqdm import tqdm
#
# # os.path.join('E:\\AlanWatts\\venv\\Lib\\site-packages\\ffmpeg-essentials_build\\bin')  # adding ffmpeg to Path
#
# filepath = "E:/AlanWattsMaterialSorted/mp3/"
# dest_path = "E:/AlanWattsMaterialSorted/audio/"
#
# for filename in os.listdir(filepath):
#     if filename not in os.listdir(dest_path):
#         '''iterates through mp4 files'''
#         for i in tqdm(range(100)):
#             '''launches progress bar'''
#             #   if filename[:-4] == "Why not now - 1.mp4_2022-11-19_19_52_45":  # testing
#             #      print(f"{filepath}" + f"{filename}")
#             given_audio = AudioSegment.from_file(f"{filepath}" + f"{filename}", format="mp3")
#             given_audio.export(f"{dest_path}" + f"{filename[:-4]}.wav", format="wav")

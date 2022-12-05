'''
Convert mp4 to wav
'''

import os

from pydub import AudioSegment
from tqdm import tqdm

# os.path.join('E:\\AlanWatts\\venv\\Lib\\site-packages\\ffmpeg-essentials_build\\bin') # adding ffmpeg to Path

filepath = "E:/AlanWattsMaterialSorted/mp4/"
dest_path = "E:/AlanWattsMaterialSorted/audio/"

for filename in os.listdir(filepath):
    '''iterates through mp4 files'''
    for i in tqdm(range(100)):
        '''launches progress bar'''
        #   if filename[:-4] == "Why not now - 1.mp4_2022-11-19_19_52_45": # testing
        #      print(f"{filepath}" + f"{filename}")
        given_audio = AudioSegment.from_file(f"{filepath}" + f"{filename}", format="mp4")
        given_audio.export(f"{dest_path}" + f"{filename[:-4]}.wav", format="wav")

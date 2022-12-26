# Transcribed wav audio clips leveraging https://github.com/abhirooptalasila/AutoSub via CLI

## Notes as of 12/25/2022

### python

Note that it currently runs on Python <=3.9.
I downloaded installer from the web and ran export PATH="/c/ProgramData/Python36:$PATH"via Git Bash.

### pip

Make sure you have the correct pip installed for your python,e.g.: version python3.6 -m ensurepip --upgrade

### dependencies

Installed from requirements.txt
Had to install some libraries in addition to what was listed there.
Make sure that whatever you install leverages your older pip for python 3.6 via python -m pip install -U scikit-learn
Otherwise, you'll run into ERROR: Could not find a version that satisfies the requirement

### model

I downloaded 0.9.3 deepspeech-0.9.3-models and scorer from https://github.com/mozilla/DeepSpeech/releases
Make sure the model files are in AutoSub/autosub

### params

usage: main.py [-h] [--format {srt,vtt,txt} [{srt,vtt,txt} ...]]
[--split-duration SPLIT_DURATION] [--dry-run]
[--engine [{ds,stt}]] [--file FILE] [--model MODEL]
[--scorer SCORER]

Example: python main.py --engine ds --format txt --file
/e/AlanWattsMaterialSorted/split_audio/test/0_02_The_Game_of_Black-and-White.mp3_2022-11-19_19_52_45.wav



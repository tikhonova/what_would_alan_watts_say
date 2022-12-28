import multiprocessing
import os
import subprocess
from multiprocessing import freeze_support

from pydub import AudioSegment as asm
from tqdm import tqdm

''' reduce the sample rate of audio file '''

filepath = "E:/AlanWattsMaterialSorted/split_audio/"
temp_path = 'E:/AutoSub/autosub/output/'

# iterate through files in directory
for i in tqdm(range(100)):
    for file in os.listdir(filepath):
        # check if file is .wav
        if file.endswith(".wav"):
            file_path = os.path.join(filepath, file)
            # read the .wav file
            sound = asm.from_file(file_path, format='wav', frame_rate=22050)
            # set the frame rate to 16000
            sound = sound.set_frame_rate(16000)
            # save the modified audio file
            sound.export(file_path,
                         format='wav')  # e.g. reducing size of 25 files from 175 MB to 71 MB might help resolve the occasional MemoryError

''' remove empty spaces in audio clip names '''

old_files = [file for file in os.listdir(filepath)]
new_files = [file.replace(' ', '_') for file in os.listdir(filepath)]

for file in old_files:
    os.rename(os.path.join(filepath, file), os.path.join(filepath, file.replace(' ', '_')))

''' multiprocessing '''

# create a list of arguments to pass to the CLI script
inputs = []
files = [os.path.join(filepath, file) for file in os.listdir(filepath) if
         f'{file[:-4]}.txt' not in os.listdir(temp_path)]


def define_inputs(arg1, arg2, arg3, arg4):
    for file in files:
        inputs.append((arg1, arg2, arg3, f'{arg4}' + ' ' + f'{file}'))


define_inputs(' --engine ds', ' --format txt', ' --split_duration 60', ' --file')


def run_cli_script(*input_args):
    # run the CLI script with the given arguments: engine, format, split duration, file to process
    subprocess.run(['python', 'main.py', input_args])


def run():
    p = multiprocessing.Pool(16)  # of cores
    num_batches = len(inputs) // 15  # number of batches to process based on the num of files (e.g. 15) to run at a time
    for i in range(num_batches):
        # get the inputs for the current batch
        batch_inputs = [(inputs[i * 15 + j]) for j in
                        range(15)]  # compiling a batch of indices based on the num of files in a batch
        p.starmap(run_cli_script, batch_inputs)


# use the ProcessPoolExecutor to run the CLI script in parallel
if __name__ == '__main__':
    freeze_support()
    run()

'''multiprocessing snippet'''
import multiprocessing
import os
import subprocess
from multiprocessing import freeze_support

# remove empty spaces in audio clip names
filepath = "E:/AlanWattsMaterialSorted/split_audio/"
temp_path = 'E:/AutoSub/autosub/audio'

old_files = [file for file in os.listdir(filepath) if file not in os.listdir(temp_path)]
new_files = [file.replace(' ', '_') for file in os.listdir(filepath)]

for file in old_files:
    os.rename(os.path.join(filepath, file), os.path.join(filepath, file.replace(' ', '_')))

# create a list of arguments to pass to the CLI script
inputs = []
files = [os.path.join(filepath, file) for file in os.listdir(filepath)]


def define_inputs(arg1, arg2, arg3, arg4):
    for file in files:
        inputs.append((arg1, arg2, arg3, f'{arg4}' + ' ' + f'{file}'))


define_inputs(' --engine ds', ' --format txt', ' --split_duration 60', ' --file')


def run_cli_script(*input_args):
    # run the CLI script with the given arguments: engine, format, split duration, file to process
    subprocess.run(['python', 'main.py', input_args])


def run():
    p = multiprocessing.Pool(16)
    my_inputs = [(inputs[i]) for i in range(0, len(inputs))]
    p.starmap(run_cli_script, my_inputs)


# use the ProcessPoolExecutor to run the CLI script in parallel
if __name__ == '__main__':
    freeze_support()
    run()

"""
1) Combine files with same extension under single directory
"""

import datetime
import os
import shutil

path = 'E:/AlanWattsMaterial/Alan Watts- Out of Your Mind (Essential Lectures)'

# This will create a properly organized list with all the filename that is there in the directory
list_ = os.listdir(path)

# This will go through each and every file
for file_ in list_:
    name, ext = os.path.splitext(file_)

    # This is going to store the extension type
    ext = ext[1:]

    # This forces the next iteration if it is the directory
    if ext == '':
        continue

    # This will move the file to the directory where the name 'ext' already exists
    if os.path.exists(path + '/' + ext):
        shutil.move(path + '/' + file_, path + '/' + ext + '/' + file_)

    # This will create a new directory if the directory does not already exist
    else:
        os.makedirs(path + '/' + ext)
        shutil.move(path + '/' + file_, path + '/' + ext + '/' + file_)

# ^^^ courtesy of https://www.geeksforgeeks.org/python-sort-and-store-files-with-same-extension/

"""
2) Get a a list of all file extensions
"""

root_with_files = 'E:/AlanWattsMaterial/'
file_formats = set(f.split('.')[-1] for dir, dirs, files in os.walk(root_with_files) \
                   for f in files if '.' in f)
print(file_formats)
# {'mp4', 'mobi', 'mp3', 'MP3', 'Mp3', 'avi', 'pdf', 'txt', 'jpg'}

"""
3) Move them out of sub directories to a single dir for mp3, another for mp4, etc.
"""
root_with_files = 'E:/AlanWattsMaterial/'
dest_root = 'E:/AlanWattsMaterialSorted/'

# set up a directory for each file format
for set_element in file_formats:
    new_root = dest_root + set_element
    isExist = os.path.exists(new_root)
    if not isExist:
        os.mkdir(new_root)

now = str(datetime.datetime.now())[:19]
now = now.replace(":", "_")
now = now.replace(' ', "_")  # generating a timestamp to append to each file to avoid dupes
print(now)

for file_format in file_formats:
    for root, dirs, files in os.walk(root_with_files):
        for file in files:
            if file.lower().endswith(f'{file_format}'):
                shutil.copy(os.path.join(root, file),
                            os.path.join(dest_root + file_format, file + f'_{now}' + f'.{file_format}'))

"""
4) Evaluate quantity and size of the material
"""

mydict = {}
for file_format in file_formats:
    counter = 0
    for root, dirs, files in os.walk(dest_root):
        for file in files:
            if file.lower().endswith(f'{file_format}'):
                counter += 1
                mydict[file_format] = counter
print(mydict)
# {'mp4': 18, 'jpg': 1, 'txt': 1, 'pdf': 7, 'avi': 4, 'mp3': 599, 'mobi': 11}

mydict = {}
for file_format in file_formats:
    size = 0
    for root, dirs, files in os.walk(dest_root):
        for file in files:
            if file.lower().endswith(f'{file_format}'):
                fp = os.path.join(root, file)
                size += os.path.getsize(fp)
                mydict[file_format] = round(size / 1000000, 2)  # converting bytes to MB
print(mydict)
# {'mp4': 2558.19, 'jpg': 0.05, 'txt': 0.0, 'pdf': 30.52, 'avi': 555.06, 'mp3': 8115.6, 'mobi': 6.22}

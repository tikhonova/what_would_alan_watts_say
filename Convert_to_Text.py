'''
Convert PDF to text
Convert mobi to text
'''
import os

import mobi

filepath = "E:/AlanWattsMaterialSorted/mobi/"
dest_path = "E:/AlanWattsMaterialSorted/txt/"

for root, dirs, files in os.walk(filepath):
    for file in files:
        filename = filepath + f'{file}'
        # for duck in ducks:
        #     duck = duck
        print(filename)
        tempdir, filepath = mobi.extract(filename)
        file_content = open(filepath, "r")
        content = file_content.read()
        print(dest_path + f'{file}' + '.txt')
        f = open(dest_path + f'{file}' + '.txt', "w")
        f.write(content)
        f.close()
        break

# https://stackoverflow.com/questions/25665/python-module-for-converting-pdf-to-text

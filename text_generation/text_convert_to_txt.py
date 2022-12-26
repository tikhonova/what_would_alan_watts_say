'''
Convert mobi to text
'''

import os

import mobi
from tqdm import tqdm

filepath = "E:/AlanWattsMaterialSorted/mobi/"
dest_path = "E:/AlanWattsMaterialSorted/txt/"

for i in tqdm(range(100)):
    for root, dirs, files in os.walk(filepath):
        for file in files:
            print(file)
            filename = filepath + f'{file}'
            print(filename)
            # for duck in ducks:
            #     duck = duck
            tempdir, test = mobi.extract(filename)
            print(tempdir, test)
            file_content = open(test, "r", encoding="utf-8")
            print(file_content.encoding)
            content = file_content.read()
            print(dest_path + f'{file}' + '.txt')
            f = open(dest_path + f'{file}' + '.txt', "w", encoding="utf-8")
            f.write(content)
            f.close()

'''
Convert PDF to text
https://stackoverflow.com/questions/25665/python-module-for-converting-pdf-to-text
'''

from pdfminer.high_level import extract_text

filepath = "E:/AlanWattsMaterialSorted/pdf/"
dest_path = "E:/AlanWattsMaterialSorted/txt/"

for i in tqdm(range(100)):
    for root, dirs, files in os.walk(filepath):
        for file in files:
            text = extract_text(f"{filepath}{file}")
            f = open(dest_path + f'{file}' + '.txt', "w", encoding="utf-8")
            f.write(text)
            f.close()

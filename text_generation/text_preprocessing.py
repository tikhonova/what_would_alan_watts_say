''' combine text into a dataframe
https://stackoverflow.com/questions/51960263/pandas-python-merge-multiple-file-text
'''

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import seaborn as sns
import nltk

nltk.download('punkt')

filepath = 'E:/AlanWatts/dataset/text/'
rows = []

for file in os.listdir(filepath):
    filename = filepath + f'{file}'
    with open(filename, 'r') as f:
        text = f.read()
        rows.append([file, text])

df = pd.DataFrame(rows, columns=['filename', 'text'])

df.to_csv('E:/AlanWatts/dataset/text/output.txt', index=False, sep='\t')

'''
Split dataset into train-validation-test: 70–20–10
https://towardsdatascience.com/fine-tuning-gpt2-for-text-generation-using-pytorch-2ee61a4f1ba7
'''
df = pd.read_csv('E:/AlanWatts/dataset/text/output.txt', sep='\t')
df.head(2)

train_test_ratio = 0.9
train_valid_ratio = 7 / 9
df_full_train, df_test = train_test_split(df, train_size=train_test_ratio, random_state=1)
df_train, df_valid = train_test_split(df_full_train, train_size=train_valid_ratio, random_state=1)


def build_dataset(df, dest_path):
    f = open(dest_path, 'w')
    data = ''
    summaries = df['text'].tolist()
    for summary in summaries:
        summary = str(summary).strip()
        summary = re.sub(r"\s", " ", summary)
        bos_token = '<BOS>'
        eos_token = '<EOS>'
        data += bos_token + ' ' + summary + ' ' + eos_token + '\n'

    f.write(data)


build_dataset(df_train, filepath + 'train.txt')
build_dataset(df_valid, filepath + 'valid.txt')
build_dataset(df_test, filepath + 'test.txt')

'''train'''
import time
import datetime

import pandas as pd
import seaborn as sns
import numpy as np

import torch

torch.manual_seed(42)

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

df = pd.read_csv('E:/AlanWatts/dataset/text/train.txt', sep='\t')
df.dropna(inplace=True)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                          pad_token='<|pad|>')  # gpt2-medium

#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-05-26 22:20:05
LastEditors: yangyuxiang
LastEditTime: 2021-05-26 23:37:14
FilePath: /Chinese-Dialogue-System/ranking/preprocessor.py
'''
from config import Config
from tqdm import tqdm
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append('..')


def read_file(file_path):
    examples = []
    with open(file_path, 'r') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            line = line.split('\t')
            question1, question2, label = line[0], line[1], line[2]
            examples.append([question1, question2, label])

    return examples


def read_csv(file_path):
    examples = []
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            index, question1, question2, label = line
            examples.append([question1, question2, label])

    return examples


def to_tsv(data, file_path):
    df = pd.DataFrame(data, columns=['question1', 'question2', 'label'])
    df.to_csv(os.path.join(Config.root_path, file_path), sep='\t', index=False)


if __name__ == "__main__":
    data1 = read_csv(os.path.join(Config.root_path,
                                  'data/file/atec_nlp_sim_train.csv'))
    data2 = read_csv(os.path.join(Config.root_path,
                                  'data/file/atec_nlp_sim_train_add.csv'))
    data3 = read_file(os.path.join(
        Config.root_path, 'data/file/task3_train.txt'))

    data1.extend(data2)
    data1.extend(data3)

    train_data, test_data = train_test_split(
        data1, test_size=0.2, random_state=33, shuffle=True)
    train_data, dev_data = train_test_split(
        train_data, test_size=0.2, random_state=33, shuffle=True)

    to_tsv(train_data, 'data/ranking/train.tsv')
    to_tsv(dev_data, 'data/ranking/dev.tsv')
    to_tsv(test_data, 'data/ranking/test.tsv')

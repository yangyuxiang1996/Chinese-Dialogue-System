#!/usr/bin/env python
# coding=utf-8
'''
Author: yangyuxiang
Date: 2021-06-06 17:05:26
LastEditors: yangyuxiang
LastEditTime: 2021-06-06 18:07:43
FilePath: /Chinese-Dialogue-System/generative/data.py
Description: 
'''
import os
import json
import sys
sys.path.append('..')
from config import Config


def process(data_path, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))

    with open(save_path, 'w') as f:
        count = 0
        with open(data_path, 'r') as r:
            js_file = json.load(r)
            for line in js_file:
                if len(line) == 2:
                    count += 1
                    line = '\t'.join(line)
                    f.write(line+'\n')
        print("count: " + str(count))


if __name__ == '__main__':
    root_path = Config.root_path
    train_raw = os.path.join(
        root_path, 'data/LCCC-base-split', 'LCCC-base_train.json')
    valid_raw = os.path.join(
        root_path, 'data/LCCC-base-split', 'LCCC-base_valid.json')
    test_raw = os.path.join(
        root_path, 'data/LCCC-base-split', 'LCCC-base_test.json')
    process(train_raw, os.path.join(root_path, 'data/generative/train.tsv'))
    process(valid_raw, os.path.join(root_path, 'data/generative/dev.tsv'))
    process(test_raw, os.path.join(root_path, 'data/generative/test.tsv'))

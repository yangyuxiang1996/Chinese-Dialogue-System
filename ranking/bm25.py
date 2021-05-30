#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-05-27 22:14:08
LastEditors: yangyuxiang
LastEditTime: 2021-05-27 23:49:33
FilePath: /Chinese-Dialogue-System/ranking/bm25.py
'''
import logging
import math
import os
import sys
from collections import Counter
import csv
import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
tqdm.pandas()
sys.path.append('..')
from config import Config
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


class BM25(object):
    def __init__(self, do_train=True, train_path=None, save_path=None):
        self.data = pd.read_csv(train_path, sep='\t', header=0,
                                quoting=csv.QUOTE_NONE,
                                names=['question1', 'question2', 'target'])
        logging.info('data: {}'.format(self.data.head(5)))
        logging.info('data shape: {}'.format(self.data.shape))
        self.stop_words = self.load_stop_words(Config.stop_words)
        if do_train:
            self.idf, self.avgdl = self.get_idf()
            self.save(save_path)
        else:
            self.idf, self.avgdl = self.load(save_path)

    def load_stop_words(self, path):
        stop_words = []
        with open(path, 'r') as f:
            for line in f.readlines():
                stop_words.append(line.strip())
        return stop_words

    def cal_idf(self, word, documents):
        N = len(documents)
        count = 0
        for doc in documents:
            if word in doc:
                count += 1
        return math.log(N / (1 + count))

    def get_idf(self):
        self.data['question2'] = self.data['question2'].progress_apply(
            lambda x: " ".join(jieba.cut(x)))
        self.vocab = Counter()
        documents = self.data['question2'].values
        for document in documents:
            for word in document.split():
                self.vocab[word] += 1
        logging.info("the vocab size: {}".format(len(self.vocab)))
        idf = {k: self.cal_idf(k, documents)
               for k, v in tqdm(self.vocab.items())}
        self.data['question2_len'] = self.data['question2'].apply(
            lambda x: len(x.split()))
        avgdl = np.mean(self.data['question2_len'].values)
        return idf, avgdl

    def load(self, save_path):
        self.idf = joblib.load(os.path.join(save_path, 'bm25_idf.bin'))
        self.avgdl = joblib.load(os.path.join(save_path, 'bm25_avgdl.bin'))

    def save(self, save_path):
        joblib.dump(self.idf, os.path.join(save_path, 'bm25_idf.bin'))
        joblib.dump(self.avgdl, os.path.join(save_path, 'bm25_avgdl.bin'))

    def cal_bm25(self, q, d, k1=1.2, k2=200, b=0.75):
        stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
        words = pseg.cut(q)
        f_i = {}
        qf_i = {}
        for word, pos in words:
            if word not in self.stop_words and pos not in stop_flag:
                f_i[word] = q.count(word)
                qf_i[word] = d.count(word)
        K = k1 * (1 - b + b * (len(d) / self.avgdl))  # 计算K值
        ri = {}
        for key in f_i:
            ri[key] = f_i[key] * (k1+1) * qf_i[key] * (k2+1) / \
                ((f_i[key] + K) * (qf_i[key] + k2))  # 计算R

        score = 0
        for key in ri:
            score += self.idf.get(key, 20.0) * ri[key]
        return score


if __name__ == "__main__":
    do_train = True
    train_path = os.path.join(Config.root_path, 'data/ranking/train.tsv')
    save_path = os.path.join(Config.root_path, 'model/ranking')
    bm25 = BM25(do_train=True, train_path=train_path, save_path=save_path)

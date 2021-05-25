#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-05-21 12:32:33
LastEditors: yangyuxiang
LastEditTime: 2021-05-24 13:46:17
FilePath: /Chinese-Dialogue-System/retrieval/word2vec.py
'''
import os
import sys
sys.path.append('..')
from utils.preprocessor import clean, read_file
from config import Config
import logging
import multiprocessing
from time import time
import pandas as pd
import jieba
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


def read_data(file_path):
    '''
    @description: 读取数据，清洗
    @param {type}
    file_path: 文件所在路径
    @return: Training samples.
    '''
    data = read_file(file_path, is_train=True)
    data = pd.DataFrame(data, columns=['session_id', 'role', 'content'])
    data['clean'] = data['content'].apply(lambda x: clean(x))
    # print("data shape: {}".format(data.shape))
    # print("data: {}".format(data.head(5)))
    return data


def train_w2v(train, to_file):
    '''
    @description: 训练word2vec model，并保存
    @param {type}
    train: 数据集 DataFrame
    to_file: 模型保存路径
    @return: None
    '''
    # stop_words_list = []
    # with open(Config.stop_words, 'r') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         stop_words_list.append(line)

    # train['cut'] = train['clean'].apply(
    #     lambda x: [w for w in jieba.cut(x) if w not in stop_words_list])
    train['cut'] = train['clean'].apply(lambda x: x.split())
    unigram_sents = train['cut'].values
    phrase_model = Phrases(
        unigram_sents, min_count=5, progress_per=10000, delimiter=b' ')
    bigram = Phraser(phrase_model)
    logging.info("sentence: %s" % unigram_sents[0])
    logging.info("the bigram sentence: %s" % bigram[unigram_sents[0]])
    corpus = bigram[unigram_sents]

    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=2,
                         window=2,
                         size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=15,
                         workers=cores - 1,
                         iter=7)
    w2v_model.build_vocab(corpus)
    w2v_model.train(corpus, epochs=15, total_examples=w2v_model.corpus_count)
    if not os.path.exists(os.path.dirname(to_file)):
        os.makedirs(os.path.dirname(to_file))
    w2v_model.wv.save(to_file)


if __name__ == "__main__":
    train = read_data(Config.train_raw)
    train_w2v(train, Config.w2v_path)

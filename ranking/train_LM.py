#!/usr/bin/env python
# coding=utf-8
'''
Author: yangyuxiang
Date: 2021-05-30 19:21:32
LastEditors: yangyuxiang
LastEditTime: 2021-05-30 22:24:07
FilePath: /Chinese-Dialogue-System/ranking/train_lm.py
Description:
'''
import logging
from re import S
import sys
import os
from collections import defaultdict

import jieba
from gensim import corpora
from gensim.models import TfidfModel, Word2Vec
from gensim.models.fasttext import FastText
import multiprocessing

sys.path.append('..')
from config import Config

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


class Trainer(object):
    def __init__(self):
        self.train_data = self.read_data(os.path.join(
            Config.root_path, "data/ranking/train.tsv"))
        self.dev_data = self.read_data(os.path.join(
            Config.root_path, "data/ranking/dev.tsv"))
        self.test_data = self.read_data(os.path.join(
            Config.root_path, "data/ranking/test.tsv"))
        self.stop_words = open(Config.stop_words).readlines()
        self.data = self.train_data + self.dev_data + self.test_data
        self.preprocessor()
        self.train()
        self.save()

    def read_data(self, path):
        samples = []
        with open(path, 'r') as f:
            next(f)
            for line in f.readlines():
                try:
                    question1, question2, label = line.split('\t')
                except Exception:
                    logging.exception("exception: " + line)
                samples.append(question1)
                samples.append(question2)
        logging.info("read data from {} with {} examples.".format(path, len(samples)))
        return samples

    def preprocessor(self):
        logging.info(" loading data.... ")
        self.data = [[word for word in jieba.cut(sentence) if word not in self.stop_words]
                     for sentence in self.data]
        self.freq = defaultdict(int)
        for sentence in self.data:
            for word in sentence:
                self.freq[word] += 1
        self.data = [[word for word in sentence if self.freq[word] > 1]
                     for sentence in self.data]
        logging.info(' building dictionary....')
        self.dictionary = corpora.Dictionary(self.data)
        self.dictionary.save(os.path.join(
            Config.root_path, 'model/ranking/ranking.dict'))
        self.corpus = [self.dictionary.doc2bow(text) for text in self.data]
        corpora.MmCorpus.serialize(os.path.join(Config.root_path, 'model/ranking/ranking.mm'),
                                   self.corpus)
        logging.info("corpus: {}".format(self.corpus[0]))

    def train(self):
        logging.info("training model tfidf...")
        self.tfidf = TfidfModel(self.corpus, normalize=True)
        logging.info('tfidf: {}'.format(self.tfidf[self.corpus[0]]))
        logging.info("training model word2vec...")
        cores = multiprocessing.cpu_count()
        self.w2v = Word2Vec(min_count=2,
                            window=2,
                            vector_size=300,
                            sample=6e-5,
                            alpha=0.03,
                            min_alpha=0.0007,
                            negative=15,
                            workers=cores - 1)
        self.w2v.build_vocab(self.data)
        self.w2v.train(self.data,
                       epochs=15,
                       total_examples=self.w2v.corpus_count)
        logging.info("training model fasttext...")
        self.fast = FastText(vector_size=300,
                             window=3,
                             min_count=1,
                             min_n=3,
                             max_n=6,
                             word_ngrams=1)
        self.fast.build_vocab(self.data)
        self.fast.train(corpus_iterable=self.data, 
                        epochs=15,
                        total_examples=self.fast.corpus_count,
                        total_words=self.fast.corpus_total_words)

    def save(self):
        logging.info(' save tfidf model ...')
        self.tfidf.save(os.path.join(Config.root_path, 'model/ranking/tfidf'))
        logging.info(' save word2vec model ...')
        self.w2v.save(os.path.join(Config.root_path, 'model/ranking/w2v'))
        logging.info(' save fasttext model ...')
        self.fast.save(os.path.join(Config.root_path, 'model/ranking/fast'))


if __name__ == '__main__':
    Trainer()

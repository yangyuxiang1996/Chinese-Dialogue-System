#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-05-20 23:03:12
LastEditors: yangyuxiang
LastEditTime: 2021-06-05 22:54:14
FilePath: /Chinese-Dialogue-System/intention/business.py
'''
import jieba
import logging
import os
import fasttext
import jieba.posseg as pseg
import pandas as pd
from tqdm import tqdm
import time
import sys
sys.path.append('..')
from utils.preprocessing import clean, filter_content
from config import Config
tqdm.pandas()
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


class Intention(object):
    def __init__(self,
                 data_path=Config.train_path,  # Original data path.
                 test_path=Config.test_path,
                 sku_path=Config.ware_path,  # Sku file path.
                 model_path=None,  # Saved model path.
                 kw_path=None,  # Key word file path.
                 # Path to save training data for intention.
                 model_train_file=Config.business_train,
                 # Path to save test data for intention.
                 model_test_file=Config.business_test,
                 stop_words=Config.stop_words,
                 from_train=False,
                 only_train=False):
        self.model_path = model_path
        self.data = pd.read_csv(data_path)
        self.test_data = pd.read_csv(test_path)
        self.stop_words_list = []
        with open(stop_words, 'r') as f:
            for line in f.readlines():
                self.stop_words_list.append(line.strip())

        if model_path and os.path.exists(model_path):
            self.fast = fasttext.load_model(model_path)
        else:
            if only_train:
                self.fast = self.train(model_train_file, model_test_file)
            else:
                self.keywords = self.build_keyword(
                    sku_path, save_path=kw_path, from_train=from_train)
                self.data_process(self.data, model_train_file)
                self.data_process(self.test_data, model_test_file)
                self.fast = self.train(model_train_file, model_test_file)

    def build_keyword(self, sku_path, save_path, from_train=False):
        '''
        @description: 构建业务咨询相关关键词，并保存，关键词来源：jieba词性标注，业务标注
        @param {type}
        sku_path： JD sku 文件路径
        to_file： 关键词保存路径
        @return: 关键词list
        '''

        logging.info('Building keywords.')

        key_words = set()
        with open(sku_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('sku'):
                    continue
                line = line.strip().split()
                if '/' in line[1]:
                    for word in line[1].split('/'):
                        key_words.add(word)
                elif '、' in line[1]:
                    for word in line[1].split('、'):
                        key_words.add(word)
                else:
                    key_words.add(line[1])

        if from_train:
            for sentence in self.data['custom'].values:
                sen_list = pseg.cut(sentence)
                for token, pos in sen_list:
                    if len(token) > 1 and pos in ['n', 'vn', 'nz']:
                        key_words.add(token)

        if save_path:
            with open(save_path, 'w') as f:
                for word in key_words:
                    f.write(word + "\n")

        return key_words

    def data_process(self, data, model_data_file):
        '''
        @description: 判断咨询中是否包含业务关键词， 如果包含label为1， 否则为0
                      并处理成fasttext 需要的数据格式:
                      "__label__" + label + "\t" + content + "\n"
        @param {type}
        model_data_file： 模型训练数据保存路径
        model_test_file： 模型验证数据保存路径
        @return:
        '''
        logging.info('processing data: %s.' % model_data_file)
        data['custom'] = data['custom'].fillna('')
        examples = []
        for sentence in data['custom'].values:
            if sentence != '':
                if any(kw in sentence for kw in self.keywords):
                    label = 1
                else:
                    label = 0
            text = '\t'.join(['__label__%s' % label, clean(sentence)])
            examples.append(text)

        with open(model_data_file, 'w') as f:
            for text in examples:
                f.write(text+'\n')

        logging.info('Processing data, finished!')

    def train(self, model_train_file, model_test_file):
        '''
        @description: 读取模型训练数据训练， 并保存
        @param {type}
        model_data_file： 模型训练数据位置
        model_test_file： 模型验证文件位置
        @return: fasttext model
        '''
        logging.info('Training classifier.')
        start_time = time.time()

        classifier = fasttext.train_supervised(model_train_file,
                                               label="__label__",
                                               dim=100,
                                               epoch=20,
                                               lr=0.5,
                                               wordNgrams=2,
                                               loss='softmax',
                                               thread=5,
                                               verbose=True)

        self.test(classifier, model_test_file)
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.mkdir(os.path.dirname(self.model_path))
        classifier.save_model(self.model_path)
        logging.info('Model saved.')
        logging.info('used time: {:.4f}s'.format(time.time() - start_time))
        
        return classifier

    def test(self, classifier, model_test_file):
        '''
        @description: 验证模型
        @param {type}
        classifier： model
        model_test_file： 测试数据路径
        @return:
        '''
        logging.info('Testing trained model.')
        result = classifier.test(model_test_file)  # N, P, R

        # F1 score
        f1 = result[1] * result[2] * 2 / (result[2] + result[1])
        logging.info("f1: %.4f" % f1)

    def predict(self, text):
        '''
        @description: 预测
        @param {type}
        text： 文本
        @return: label, score
        '''
        logging.info('Predicting.')
        clean_text = clean(filter_content(text))
        logging.info('text: %s' % text)
        logging.info('clean text: %s' % clean_text)
        start_time = time.time()
        label, score = self.fast.predict(clean_text)
        logging.info('used time: {:.4f}s'.format(time.time() - start_time))
        return label, score


if __name__ == "__main__":
    it = Intention(data_path=Config.train_path,
                   test_path=Config.test_path,
                   sku_path=Config.ware_path,
                   model_path=Config.ft_path,
                   kw_path=Config.keyword_path,
                   from_train=True,
                   only_train=False)
    print(it.predict('你好想问有卖鞋垫吗[SEP][链接x]'))
    print(it.predict('今天天气真好'))

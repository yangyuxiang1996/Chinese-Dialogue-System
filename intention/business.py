#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-05-20 23:03:12
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-31 16:56:39
FilePath: /Chinese-Dialogue-System/intention/business.py
'''
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
                 stop_words=Config.stop_words):
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
            # self.keywords = self.build_keyword(sku_path, save_path=kw_path)
            # self.data_process(self.data, model_train_file)
            self.keywords = [word.strip() for word in open(kw_path, 'r').readlines()]
            # self.data_process(self.test_data, model_test_file)
            self.fast = self.train(model_train_file, model_test_file)

    def build_keyword(self, sku_path, save_path):
        '''
        @description: 构建业务咨询相关关键词，并保存，关键词来源：jieba词性标注，业务标注
        @param {type}
        sku_path： JD sku 文件路径
        to_file： 关键词保存路径
        @return: 关键词list
        '''

        logging.info('Building keywords.')
        tokens = []
        # Filtering words according to POS tags.

        tokens = self.data['custom'].dropna().progress_apply(
            lambda x: [
                token for token, pos in pseg.cut(x) if pos in ['n', 'vn', 'nz']
                ])

        key_words = set(
            [tk for idx, sample in tokens.iteritems()
                for tk in sample if len(tk) > 1])
        logging.info('Key words built.')
        sku = []
        with open(sku_path, 'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split('\t')
                sku.extend(line[-1].split('/'))
        key_words |= set(sku)
        logging.info('Sku words merged.')
        if save_path is not None:
            with open(save_path, 'w') as f:
                for i in key_words:
                    f.write(i + '\n')
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
        data = data.dropna()
        data['is_business'] = data['custom'].progress_apply(
            lambda x: 1 if any(kw in str(x) for kw in self.keywords) else 0)
        with open(model_data_file, 'w') as f:
            for index, row in tqdm(data.iterrows(), total=data.shape[0]):
                print(row['custom'])
                if len(row['custom']) > 1:
                    outline = clean(row['custom']) + "\t__label__" + str(int(row['is_business'])) + "\n"
                    f.write(outline)
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
                                               epoch=10,
                                               lr=0.1,
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
                   kw_path=Config.keyword_path)
    print(it.predict('你好想问有卖鞋垫吗[SEP][链接x]'))
    print(it.predict('你有对象吗'))

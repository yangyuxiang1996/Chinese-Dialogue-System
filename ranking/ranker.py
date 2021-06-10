#!/usr/bin/env python
# coding=utf-8
'''
Author: yangyuxiang
Date: 2021-06-05 17:47:49
LastEditors: yangyuxiang
LastEditTime: 2021-06-10 08:54:01
FilePath: /Chinese-Dialogue-System/ranking/ranker.py
Description:
'''

import sys
import os
import csv
import logging

import lightgbm as lgb
import pandas as pd
import joblib
from tqdm import tqdm

sys.path.append('..')
from config import Config
from ranking.model import MatchNN
from utils.similarity import TextSimilarity
from retrieval.hnsw_faiss import wam

from sklearn.model_selection import train_test_split
import numpy as np

tqdm.pandas()

# Parameters for lightGBM
params = {
    'boosting_type': 'gbdt',
    'max_depth': 10,
    'objective': 'binary',
    'nthread': 3,
    'num_leaves': 64,
    'learning_rate': 0.05,
    'max_bin': 512,
    'subsample_for_bin': 200,
    'subsample': 0.5,
    'subsample_freq': 5,
    'colsample_bytree': 0.8,
    'reg_alpha': 5,
    'reg_lambda': 10,
    'min_split_gain': 0.5,
    'min_child_weight': 1,
    'min_child_samples': 5,
    'scale_pos_weight': 1,
    'max_position': 20,
    'group': 'name:groupId',
    'metric': 'auc'
}


class RANK(object):
    def __init__(self,
                 do_train=True,
                 model_path=os.path.join(Config.root_path, 'model/ranking/lightgbm')):
        self.ts = TextSimilarity()
        self.matchingNN = MatchNN()
        self.ranker = lgb.LGBMRanker(params)
        self.train_data = pd.read_csv(os.path.join(Config.root_path, 'data/ranking/train.tsv'),
                                      sep='\t',
                                      header=0,
                                      quoting=csv.QUOTE_NONE)
        self.test_data = pd.read_csv(os.path.join(Config.root_path, 'data/ranking/test.tsv'),
                                     sep='\t',
                                     header=0,
                                     quoting=csv.QUOTE_NONE)
        self.dev_data = pd.read_csv(os.path.join(Config.root_path, 'data/ranking/dev.tsv'),
                                    sep='\t',
                                    header=0,
                                    quoting=csv.QUOTE_NONE)
        if do_train:
            logging.info('Training mode')
#             self.train_data = self.generate_feature(self.train_data)
            logging.info("train_data columns: {}".format(self.train_data.columns))
            logging.info("train_data shape: {}".format(self.train_data.shape))
            logging.info("train_data: {}".format(self.train_data[:5]))
            self.dev_data = self.generate_feature(self.dev_data)
            logging.info("dev_data shape: {}".format(self.dev_data.shape))
            exit
            self.trainer()
            self.save(model_path)

        else:
            logging.info('Predicting mode')
            self.test_data = self.generate_feature(self.test_data)
            logging.info("test_data shape: {}".format(self.test_data.shape))
            self.ranker = joblib.load(model_path)
            self.predict(self.test_data)

    def generate_feature(self, data):
        '''
        @description: 生成模型训练所需要的特征
        @param {type}
        data Dataframe
        @return: Dataframe
        '''
        # 生成人工特征
        logging.info('Generating feature...')
#         data['features'] = data.apply(lambda row: self.ts.generate_all(
#             row['question1'], row['question2']), axis=1)
#         data_features = data['features'].apply(pd.Series)
#         data = pd.concat([data, data_features],
#                          axis=1).drop('features', axis=1)
        # 生成深度匹配特征
        data['bert_feature'] = data.apply(lambda row: self.matchingNN.predict(
            row['question1'], row['question2'])[1], axis=1)

        data['label'] = data['label'].astype('int8')

        return data

    def trainer(self):
        logging.info('Training lightgbm model.')
        except_columns = ['question1', 'question2', 'label']
        feature_columns = [
            col for col in self.train_data.columns
            if col not in except_columns]
        X_train, y_train = self.train_data[feature_columns], self.train_data['label']
        X_test, y_test = self.dev_data[feature_columns], self.dev_data['label']
        query_train = [X_train.shape[0]]
        query_eval = [X_test.shape[0]]
        self.ranker.fit(X_train, y_train, group=query_train,
                        eval_set=[(X_test, y_test)], eval_group=[query_eval],
                        eval_at=[5, 10, 20], early_stopping_rounds=50,
                        verbose=1)

    def save(self, model_path):
        logging.info('Saving lightgbm model.')
        if not os.path.exists(os.path.dirname(model_path)):
            os.mkdir(os.path.dirname(model_path))
        joblib.dump(self.ranker, model_path)

    def predict(self, data: pd.DataFrame):
        """Doing prediction.
        Args:
            data (pd.DataFrame): the output of self.generate_feature
        Returns:
            result[list]: The scores of all query-candidate pairs.
        """
        except_columns = ['question1', 'question2', 'label']
        feature_columns = [
            col for col in self.train_data.columns
            if col not in except_columns]
        result = self.ranker.predict(data[feature_columns])
        return result


if __name__ == "__main__":
    rank = RANK(do_train=True)

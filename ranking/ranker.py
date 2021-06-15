#!/usr/bin/env python
# coding=utf-8
'''
Author: yangyuxiang
Date: 2021-06-05 17:47:49
LastEditors: yangyuxiang
LastEditTime: 2021-06-15 21:24:51
FilePath: /Chinese-Dialogue-System/ranking/ranker.py
Description:
'''

import numpy as np
import sys
import os
import csv
import logging

import lightgbm as lgb
import pandas as pd
import joblib
from tqdm import tqdm

sys.path.append('..')
from sklearn.model_selection import train_test_split
from retrieval.hnsw_faiss import wam
from utils.similarity import TextSimilarity
from ranking.model import MatchNN
from config import Config
tqdm.pandas()

# Parameters for lightGBM
params = {
    'task': 'train',  # 执行的任务类型
    'boosting_type': 'gbdt',  # 基学习器
    'objective': 'lambdarank',  # 排序任务(目标函数)
    'metric': 'ndcg',  # 度量的指标(评估函数)
    'max_position': 10,  # @NDCG 位置优化
    'metric_freq': 1,  # 每隔多少次输出一次度量结果
    'train_metric': True,  # 训练时就输出度量结果
    'ndcg_at': [10],
    'max_bin': 255,  # 一个整数，表示最大的桶的数量。默认值为 255。lightgbm 会根据它来自动压缩内存。如max_bin=255 时，则lightgbm 将使用uint8 来表示特征的每一个值。
    'num_iterations': 200,  # 迭代次数，即生成的树的棵数
    'learning_rate': 0.01,  # 学习率
    'num_leaves': 31,  # 叶子数
    # 'max_depth':6,
    'tree_learner': 'serial',  # 用于并行学习，‘serial’： 单台机器的tree learner
    'min_data_in_leaf': 30,  # 一个叶子节点上包含的最少样本数量
    'verbose': 2  # 显示训练时的信息
}


class RANK(object):
    def __init__(self,
                 mode='train',
                 model_path=os.path.join(Config.root_path, 'model/ranking/lightgbm')):
        self.ts = TextSimilarity()
        self.matchingNN = MatchNN()
        self.ranker = lgb.LGBMRanker(**params)
        self.train_data = pd.read_csv(os.path.join(Config.root_path, 'data/ranking/train.tsv'),
                                      sep='\t',
                                      header=0,
                                      quoting=csv.QUOTE_NONE)
        self.dev_data = pd.read_csv(os.path.join(Config.root_path, 'data/ranking/dev.tsv'),
                                    sep='\t',
                                    header=0,
                                    quoting=csv.QUOTE_NONE)

        if mode == 'train':
            logging.info('Training mode')
            self.train_data = self.generate_feature(self.train_data, 'train')
            logging.info("train_data columns: {}".format(
                self.train_data.columns))
            logging.info("train_data shape: {}".format(self.train_data.shape))
            logging.info("train_data: {}".format(self.train_data[:5]))
            self.dev_data = self.generate_feature(self.dev_data, 'dev')
            logging.info("dev_data shape: {}".format(self.dev_data.shape))
            self.ranker = self.trainer()
            self.save(self.ranker, model_path)

        else:
            self.ranker = joblib.load(model_path)

    def generate_feature(self, data, mode='train'):
        '''
        @description: 生成模型训练所需要的特征
        @param {type}
        data Dataframe
        @return: Dataframe
        '''
        if os.path.exists(os.path.join(Config.root_path, 'data/ranking/%s_features.pkl' % mode)):
            data = pd.read_pickle(os.path.join(
                Config.root_path, 'data/ranking/%s_features.pkl' % mode))
        else:
            # 生成人工特征
            logging.info('Generating statistic feature...')
            data['features'] = data.progress_apply(lambda row: self.ts.generate_all(
                row['question1'], row['question2']), axis=1)
            data_features = data['features'].apply(pd.Series)
            data = pd.concat([data, data_features],
                             axis=1).drop('features', axis=1)
            # 生成深度匹配特征
            logging.info('Generating deep feature...')
            data['bert_feature'] = data.progress_apply(lambda row: self.matchingNN.predict(
                row['question1'], row['question2'])[1], axis=1)

            if 'label' in data.columns:
                data['label'] = data['label'].astype('int8')
            data.to_pickle(os.path.join(Config.root_path,
                           'data/ranking/%s_features.pkl' % mode))

        return data

    def trainer(self):
        logging.info('Training lightgbm model.')
        except_columns = ['question1', 'question2', 'label']
        feature_columns = [
            col for col in self.train_data.columns
            if col not in except_columns]
        X_train, y_train = self.train_data[feature_columns].values, self.train_data['label'].values
        X_test, y_test = self.dev_data[feature_columns].values, self.dev_data['label'].values
        query_train = [X_train.shape[0]]
        query_eval = [X_test.shape[0]]
        self.ranker.fit(X_train, y_train,
                        group=query_train,
                        eval_set=[(X_test, y_test)],
                        eval_group=[query_eval],
                        eval_at=[5, 10, 20],
                        early_stopping_rounds=50,
                        verbose=1)
#         train_data = lgb.Dataset(X_train, label=y_train, group=query_train)
#         eval_data = lgb.Dataset(X_test, label=y_test, group=query_eval)
#         self.ranker = lgb.train(params, train_data, valid_sets=[eval_data])
        return self.ranker

    def save(self, model, model_path):
        logging.info('Saving lightgbm model.')
        if not os.path.exists(os.path.dirname(model_path)):
            os.mkdir(os.path.dirname(model_path))
        joblib.dump(model, model_path)

    def predict(self, data: pd.DataFrame):
        """Doing prediction.
        Args:
            data (pd.DataFrame): the output of self.generate_feature
        Returns:
            result[list]: The scores of all query-candidate pairs.
        """
        except_columns = ['question1', 'question2', 'label']
        feature_columns = [
            col for col in data.columns
            if col not in except_columns]
        result = self.ranker.predict(data[feature_columns], raw_score=True)
        return result


if __name__ == "__main__":
    rank = RANK(mode="train")
    logging.info('Predicting mode')
    test_data = pd.read_csv(os.path.join(Config.root_path, 'data/ranking/test.tsv'),
                            sep='\t',
                            header=0,
                            quoting=csv.QUOTE_NONE)
    test_data = rank.generate_feature(test_data, 'test')
    logging.info("test_data shape: {}".format(test_data.shape))
    result = rank.predict(test_data)
    logging.info("test_data predict: {}".format(test_data[:5]))

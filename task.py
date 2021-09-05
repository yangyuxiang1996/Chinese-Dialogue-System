#!/usr/bin/env python
# coding=utf-8
'''
Author: yangyuxiang
Date: 2021-06-10 20:48:51
LastEditors: yangyuxiang
LastEditTime: 2021-06-15 21:23:23
FilePath: /Chinese-Dialogue-System/ranking/task.py
Description:
'''
import os
import sys
from ranking import ranker
from utils.preprocessing import clean
sys.path.append('..')
from config import Config
from intention.business import Intention
from retrieval.hnsw_faiss import wam, HNSW
from ranking.ranker import RANK
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


def retrieve(k):
    dev_data = pd.read_csv(Config.dev_path)
    test_data = pd.read_csv(Config.test_path)
    data = dev_data.append(test_data)
    logging.info("data: {}".format(data[:5]))

    it = Intention(data_path=Config.train_path,
                   test_path=Config.test_path,
                   sku_path=Config.ware_path,
                   model_path=Config.ft_path,
                   kw_path=Config.keyword_path)

    hnsw = HNSW(Config.w2v_path,
                Config.M,
                Config.efConstruction,
                Config.efSearch,
                Config.hnsw_path,
                Config.train_path)

    res = pd.DataFrame()
    querys = data['custom'].dropna().values
#     logging.info('querys: '.format(querys))
    cnt = 0
    for query in querys:
        query = query.strip()
        # print(query)
        intention, score = it.predict(query)
        if len(query) > 1 and intention[0] == '__label__1':
            cnt += 1
            res = res.append(
                pd.DataFrame(
                    {'query': [query]*k,
                     'retrieved': hnsw.search(query, k)['custom']}))

    logging.info("__label__1: {}".format(cnt))
    res = pd.DataFrame(res, columns=['query', 'retrieved'])

    res.to_csv(os.path.join(Config.root_path, 'result/retrieved.csv'),
               header=True, index=False)


def rank():
    rank = RANK(mode='predict')
    data = pd.read_csv(os.path.join(
        Config.root_path, 'result/retrieved.csv'), header=0)
    data.columns = ['question1', 'question2']
    rank_data = rank.generate_feature(data, mode='retrieve')
    logging.info("rank_data: {}".format(rank_data[:5]))
    rank_scores = rank.predict(rank_data)

    ranked = pd.DataFrame()
    ranked['question1'] = rank_data['question1']
    ranked['question2'] = rank_data['question2']
    ranked['score'] = pd.Series(rank_scores)
    ranked.to_csv('result/ranked.csv', index=False)


if __name__ == "__main__":
    retrieve(5)
    rank()
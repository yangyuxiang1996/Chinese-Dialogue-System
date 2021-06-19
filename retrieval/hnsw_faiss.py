#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-05-22 09:24:45
LastEditors: yangyuxiang
LastEditTime: 2021-06-19 17:06:35
FilePath: /Chinese-Dialogue-System/retrieval/hnsw_faiss.py
'''
import time
import os
import sys
import logging
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import faiss
sys.path.append('..')
from utils.preprocessing import clean
from config import Config
from tqdm import tqdm
tqdm.pandas()

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


def wam(sentence, w2v_model):
    '''
    @description: 通过word average model 生成句向量
    @param {type}
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    @return: The sentence vector.
    '''
    sentence = sentence.split()
    sen_len = len(sentence)
    sen_vec = np.zeros(w2v_model.vector_size).astype("float32")
    for word in sentence:
        try:
            wv = w2v_model.wv[word]
            sen_vec += wv
        except Exception as e:
            sen_vec += np.random.randn(300).astype("float32")

    return sen_vec / sen_len


class HNSW(object):
    def __init__(self,
                 w2v_path,
                 M=Config.M,
                 efConstruction=Config.efConstruction,
                 efSearch=Config.efSearch,
                 model_path=None,
                 data_path=None,
                 ):
        self.w2v_model = KeyedVectors.load(w2v_path)
        self.data = self.load_data(data_path)
        if model_path and os.path.exists(model_path):
            # 加载
            self.index = self.load_hnsw(model_path)
        elif data_path:
            # 训练
            self.index = self.build_hnsw(model_path,
                                         M=M,
                                         efConstruction=efConstruction,
                                         efSearch=efSearch)
        else:
            logging.error('No existing model and no building data provided.')

    def load_data(self, data_path):
        '''
        @description: 读取数据，并生成句向量
        @param {type}
        data_path：问答pair数据所在路径
        @return: 包含句向量的dataframe
        '''
        if os.path.exists(data_path.replace('.csv', '_for_hnsw.pkl')):
            logging.info("Reading data from %s" % data_path.replace('.csv', '_for_hnsw.pkl'))
            data = pd.read_pickle(data_path.replace('.csv', '_for_hnsw.pkl'))
            logging.info("data: %s" % data.head(5))
        else:
            logging.info("Reading data from %s" % data_path)
            data = pd.read_csv(data_path, header=0)
            data['custom_vec'] = data['custom'].progress_apply(
                lambda s: wam(clean(s), self.w2v_model))
            # data['assistance_vec'] = data['assistance'].apply(
            #     lambda s: wam(s, self.w2v_model))
            data = data.dropna()
            logging.info("data: %s" % data.head(5))
            data.to_pickle(data_path.replace('.csv', '_for_hnsw.pkl'))
        return data

    def evaluate(self, index, vecs):
        '''
        @description: 评估模型。
        @param {type} vecs: The vectors to evaluate.
        @return {type} None
        '''
        logging.info('Evaluating.')

        nq, d = vecs.shape
        t0 = time.time()
        D, I = index.search(vecs, 1)
        t1 = time.time()

        missing_rate = (I == -1).sum() / float(nq)
        recall_at_1 = (I == np.arange(nq)).sum() / float(nq)
        logging.info('\t %7.3f ms per query, R@1 %.4f, missing_rate %.4f' %
                     ((t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))

    def build_hnsw(self, to_file, M=64, efConstruction=2000, efSearch=32):
        '''
        @description: 训练hnsw模型
        @param {type}
        to_file： 模型保存目录
        ef：
        m：
        @return:
        '''
        logging.info('Building hnsw index.')
        # nb = len(self.data['assistance_vec'])
        vecs = np.stack(
            self.data['custom_vec'].values).reshape(-1, 300).astype("float32")
        # vecs = np.zeros(shape=(nb, 300), dtype=np.float32)
        # for i, vec in enumerate(self.data['assistance_vec'].values):
        #     vecs[i, :] = vec
        dim = self.w2v_model.vector_size
        index_hnsw = faiss.IndexHNSWFlat(dim, M)
        index_hnsw.hnsw.efConstruction = efConstruction
        index_hnsw.hnsw.efSearch = efSearch

        res = faiss.StandardGpuResources()  # use a single GPU
        gpu_index_hnsw = faiss.index_cpu_to_gpu(res, 0, index_hnsw)  # make it a GPU index
        gpu_index_hnsw.verbose = True

        logging.info('xb: {}'.format(vecs.shape))
        logging.info('dtype: {}'.format(vecs.dtype))
        gpu_index_hnsw.add(vecs)

        logging.info("total: %s" % str(gpu_index_hnsw.ntotal))

        assert to_file is not None
        logging.info('Saving hnsw index to %s' % to_file)
        if not os.path.exists(os.path.dirname(to_file)):
            os.mkdir(os.path.dirname(to_file))
        faiss.write_index(gpu_index_hnsw, to_file)

        self.evaluate(gpu_index_hnsw, vecs[:10000])

        return gpu_index_hnsw

    def load_hnsw(self, model_path):
        '''
        @description: 加载训练好的hnsw模型
        @param {type}
        model_path： 模型保存的目录
        @return: hnsw 模型
        '''
        logging.info(f'Loading hnsw index from {model_path}.')
        hnsw = faiss.read_index(model_path)

        return hnsw

    def search(self, text, k=5):
        '''
        @description: 通过hnsw 检索
        @param {type}
        text: 检索句子
        k: 检索返回的数量
        @return: DataFrame containing the customer input, assistance response
                and the distance to the query.
        '''
        test_vec = wam(clean(text), self.w2v_model)
        test_vec = test_vec.reshape(1, -1)

        D, I = self.index.search(test_vec, k)
        logging.info("D: {}".format(D))
        logging.info("I: {}".format(I))

        return pd.concat(
            (self.data.iloc[I[0]]['custom'].reset_index(),
             self.data.iloc[I[0]]['assistance'].reset_index(drop=True),
             pd.DataFrame(D.reshape(-1, 1), columns=['q_distance'])),
            axis=1)


if __name__ == "__main__":
    hnsw = HNSW(Config.w2v_path,
                Config.M,
                Config.efConstruction,
                Config.efSearch,
                Config.hnsw_path,
                Config.train_path)
    test = '我要转人工'
    print(hnsw.search(test, k=10))
    eval_vecs = np.stack(hnsw.data['custom_vec'].values).reshape(-1, 300).astype('float32')
    hnsw.evaluate(hnsw.index, eval_vecs[:10000])

#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-05-22 09:24:45
LastEditors: Yuxiang Yang
LastEditTime: 2021-09-01 17:41:11
FilePath: /Chinese-Dialogue-System/retrieval/hnsw_faiss.py
'''
import time
import os
import sys
import logging
import pandas as pd
import numpy as np
import random
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


class Index(object):
    def __init__(self,
                 w2v_path,
                 model_type=Config.model_type,
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
            self.index = self.load_model(model_path)
        elif data_path:
            # 训练
            self.index = self.build_model(model_type,
                                          model_path,
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
            data = data.dropna(axis=0, subset=["custom"])
            data['custom'] = data['custom'].progress_apply(lambda s: clean(s))
            data = data[data['custom'].str.len() > 1]
            data['custom_vec'] = data['custom'].progress_apply(lambda s: wam(s, self.w2v_model))
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
        
        logging.info("I: {}".format(I))
        logging.info(self.data.iloc[I[0][0]])
        logging.info("nq: {}".format(np.arange(nq)))
        missing_rate = (I == -1).sum() / float(nq)
        recall_at_1 = (I == np.arange(nq)).sum() / float(nq)
        logging.info('\t %7.3f ms per query, R@1 %.4f, missing_rate %.4f' %
                     ((t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))

    def build_model(self,
                    model_type='IndexHNSWFlat',
                    to_file=None,
                    M=64,
                    efConstruction=2000,
                    efSearch=32,
                    d=300,
                    nlist=100,
                    k=4):
        '''
        @description: 训练hnsw模型
        @param {type}
        to_file： 模型保存目录
        ef：
        m：
        @return:
        '''
        logging.info('Building index.')
        # nb = len(self.data['assistance_vec'])
        vecs = np.stack(self.data['custom_vec'].values).reshape(-1, d).astype("float32")
        start_time = time.time()
        # vecs = np.zeros(shape=(nb, 300), dtype=np.float32)
        # for i, vec in enumerate(self.data['assistance_vec'].values):
        #     vecs[i, :] = vec
        dim = self.w2v_model.vector_size
        if model_type == 'IndexHNSWFlat':
            index = faiss.IndexHNSWFlat(dim, M)
            index.hnsw.efConstruction = efConstruction
            index.hnsw.efSearch = efSearch
        elif model_type == 'IndexFlatL2':
            index = faiss.IndexFlatL2(d)
        elif model_type == 'IndexIVFFlat':
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            assert not index.is_trained
            index.train(vecs)  # IndexIVFFlat是需要训练的，这边是学习聚类
            assert index.is_trained
        elif model_type == 'IndexIVFPQ':
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFPQ(quantizer, d, nlist, 5, 8)
            index.train(vecs)
        else:
            raise ValueError("%s is not supported!" % model_type)
        
        index.verbose = True
        # res = faiss.StandardGpuResources()  # use a single GPU
        # gpu_index_hnsw = faiss.index_cpu_to_gpu(res, 0, index_hnsw)  # make it a GPU index
        # gpu_index_hnsw.verbose = True

        logging.info('xb: {}'.format(vecs.shape))
        logging.info('dtype: {}'.format(vecs.dtype))
        index.add(vecs)

        logging.info("total: %s" % str(index.ntotal))
        logging.info("using time: %s", time.time() - start_time)

        assert to_file is not None
        logging.info('Saving hnsw index to %s' % to_file)
        if not os.path.exists(os.path.dirname(to_file)):
            os.mkdir(os.path.dirname(to_file))
        faiss.write_index(index, to_file)

        self.evaluate(index, vecs[:10000])

        return index

    def load_model(self, model_path):
        '''
        @description: 加载训练好的hnsw模型
        @param {type}
        model_path： 模型保存的目录
        @return: hnsw 模型
        '''
        logging.info(f'Loading index from {model_path}.')
        index = faiss.read_index(model_path)

        return index

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
#              self.data.iloc[I[0]]['assistance'].reset_index(drop=True),
             pd.DataFrame(D.reshape(-1, 1), columns=['q_distance'])),
            axis=1)


if __name__ == "__main__":
    index = Index(Config.w2v_path,
                  Config.model_type,
                  Config.M,
                  Config.efConstruction,
                  Config.efSearch,
                  Config.hnsw_path,
                  Config.train_path)
    test = '这款电脑能用不'
#     test=index.data['custom'].iloc[0]
    logging.info(test)
    logging.info(index.search(test, k=5))
    
    eval_vecs = np.stack(index.data['custom_vec'].values).reshape(-1, 300).astype('float32')
    index.evaluate(index.index, eval_vecs[:10000])
    

#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-05-20 15:55:20
LastEditors: yangyuxiang
LastEditTime: 2021-05-21 11:16:20
FilePath: /Chinese-Dialogue-System/config.py
'''
import os
import torch


class Config(object):
    root_path = os.path.abspath(os.path.dirname(__file__))
    train_raw = os.path.join(root_path, 'data/file/chat.txt')
    dev_raw = os.path.join(root_path, 'data/file/开发集.txt')
    test_raw = os.path.join(root_path, 'data/file/测试集.txt')
    ware_path = os.path.join(root_path, 'data/file/ware.txt')

    sep = '[SEP]'
    stop_words = os.path.join(root_path, 'data/file/stop_words.txt')

    # main
    train_path = os.path.join(root_path, 'data/train_no_blank.csv')
    dev_path = os.path.join(root_path, 'data/dev.csv')
    test_path = os.path.join(root_path, 'data/test.csv')
    # intention
    business_train = os.path.join(root_path, 'data/intention/business.train')
    business_test = os.path.join(root_path, 'data/intention/business.test')
    keyword_path = os.path.join(root_path, 'data/intention/key_word.txt')

    ''' Intention '''
    # fasttext
    ft_path = os.path.join(root_path, "model/intention/fastext.bin")

    ''' Retrival '''
    # Embedding
    w2v_path = os.path.join(root_path, "model/retrieval/word2vec")

    # HNSW parameters
    ef_construction = 3000  # ef_construction defines a construction time/accuracy trade-off
    M = 64  # M defines tha maximum number of outgoing connections in the graph
    hnsw_path = os.path.join(root_path, 'model/retrieval/hnsw_index')

    # 通用配置
    is_cuda = True
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    


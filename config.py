#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-05-20 15:55:20
LastEditors: yangyuxiang
LastEditTime: 2021-06-08 22:31:55
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
    efConstruction = 40  # ef_construction defines a construction time/accuracy trade-off. 40 for default
    efSearch = 32
    M = 64  # M defines tha maximum number of outgoing connections in the graph
    hnsw_path = os.path.join(root_path, 'model/retrieval/hnsw_index')

    # 通用配置
    is_cuda = False
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    max_seq_len = 128
    batch_size = 8
    lr = 2e-05
    epochs = 1

    pretrained_path = os.path.join(root_path, 'lib/bert/')
    vocab_path = os.path.join(root_path, 'lib/bert/vocab.txt')
    config_path = os.path.join(root_path, 'lib/bert/bert_config.json')
    bert_model = os.path.join(root_path, 'model/ranking/best.pth.tar')

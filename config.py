#!/usr/bin/env python
# coding=utf-8
'''
Description: 
Author: yangyuxiang
Date: 2021-05-20 15:55:20
LastEditors: yangyuxiang
LastEditTime: 2021-06-10 07:35:45
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
    train_path = os.path.join(root_path, 'data/train_no_blank_all.csv')
    dev_path = os.path.join(root_path, 'data/dev.csv')
    test_path = os.path.join(root_path, 'data/test.csv')
    # intention
    business_train = os.path.join(root_path, 'data/intention/business_all.train')
    business_test = os.path.join(root_path, 'data/intention/business_all.test')
    keyword_path = os.path.join(root_path, 'data/intention/key_word_all.txt')

    ''' Intention '''
    # fasttext
    ft_path = os.path.join(root_path, "model/intention/fasttext_all.bin")

    ''' Retrival '''
    # Embedding
    w2v_path = os.path.join(root_path, "model/retrieval/word2vec_all")

    # HNSW parameters
    efConstruction = 100  # ef_construction defines a construction time/accuracy trade-off. 40 for default
    efSearch = 64
    M = 64  # M defines tha maximum number of outgoing connections in the graph
    hnsw_path = os.path.join(root_path, 'model/retrieval/hnsw_index_all_efConstruction{}_efSearch{}' \
                             .format(efConstruction, efSearch))

    # 通用配置
    is_cuda = False
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    max_seq_len = 128
    batch_size = 32
    lr = 2e-05
    epochs = 10

    pretrained_path = os.path.join(root_path, 'lib/chinese_roberta_wwm_large_ext_pytorch/')
    vocab_path = os.path.join(root_path, 'lib/chinese_roberta_wwm_large_ext_pytorch/vocab.txt')
    config_path = os.path.join(root_path, 'lib/chinese_roberta_wwm_large_ext_pytorch/config.json')
    bert_model = os.path.join(root_path, 'model/ranking/roberta.best.pth.tar')

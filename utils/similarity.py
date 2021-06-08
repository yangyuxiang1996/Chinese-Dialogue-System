#!/usr/bin/env python
# coding=utf-8
'''
Author: yangyuxiang
Date: 2021-05-30 22:33:02
LastEditors: yangyuxiang
LastEditTime: 2021-06-05 23:46:43
FilePath: /Chinese-Dialogue-System/utils/similarity.py
Description:
'''
import logging
import sys
sys.path.append('..')
import os
import jieba.posseg as pseg
import numpy as np
from gensim import corpora, models
from config import Config
from retrieval.hnsw_faiss import wam
from ranking.bm25 import BM25

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class TextSimilarity(object):
    def __init__(self):
        logging.info('load dictionary')
        self.dictionary = corpora.Dictionary.load(
            os.path.join(Config.root_path, 'model/ranking/ranking.dict'))
        logging.info('load corpus')
        self.corpus = corpora.MmCorpus(os.path.join(
            Config.root_path, 'model/ranking/ranking.mm'))
        logging.info('load tfidf')
        self.tfidf = models.TfidfModel.load(
            os.path.join(Config.root_path, 'model/ranking/tfidf'))
        logging.info('load bm25')
        self.bm25 = BM25(do_train=False)
        logging.info('load word2vec')
        self.w2v_model = models.KeyedVectors.load(
            os.path.join(Config.root_path, 'model/ranking/w2v'))
        logging.info('load fasttext')
        self.fasttext = models.FastText.load(
            os.path.join(Config.root_path, 'model/ranking/fast'))

    # get LCS(longest common subsquence),DP
    def lcs(self, str_a, str_b):
        """Longest common substring

        Returns:
            ratio: The length of LCS divided by the length of
                the shorter one among two input strings.
        """
        if not str_a or not str_b:
            return 0

        len1 = len(str_a)
        len2 = len(str_b)

        dp = [[0 for _ in range(len2+1)] for _ in range(len1+1)]

        for i in range(1, len1+1):
            for j in range(1, len2+1):
                if str_a[i-1] == str_b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        ratio = dp[len1][len2] / float(min(len1, len2))
        return ratio

    def editDistance(self, str1, str2):
        """Edit distance

        Returns:
            ratio: Minimum edit distance divided by the length sum
                of two input strings.
        """
        if not str1:
            return len(str2)
        if not str2:
            return len(str1)

        m = len(str1)
        n = len(str2)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]  # (m+1) * (n+1)

        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j

        for i in range(1, m+1):
            for j in range(1, n+1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1]) + 1

        return dp[m][n] / float(m+n)

    @classmethod
    def tokenize(self, str_a):
        '''
        接受一个字符串作为参数，返回分词后的结果字符串(空格隔开)和集合类型
        '''
        wordsa = pseg.cut(str_a)
        cuta = ""
        seta = set()
        for key in wordsa:
            cuta += key.word + " "
            seta.add(key.word)

        return [cuta, seta]

    def JaccardSim(self, str_a, str_b):
        '''
        Jaccard相似性系数
        计算sa和sb的相似度 len（sa & sb）/ len（sa | sb）
        '''
        a = self.tokenize(str_a)[1]
        b = self.tokenize(str_b)[1]
        jaccard_sim = 1.0 * len(a & b) / len(a | b)
        return jaccard_sim

    @staticmethod
    def cos_sim(a, b):
        a = np.array(a)
        b = np.array(b)
        cos_sim = np.sum(a*b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))
        return cos_sim

    @staticmethod
    def eucl_sim(a, b):
        a = np.array(a)
        b = np.array(b)
        eucl_sim = 1 / (1 + np.sqrt((np.sum(a - b)**2)))
        return eucl_sim

    @staticmethod
    def pearson_sim(a, b):
        a = np.array(a)
        b = np.array(b)

        a = a - np.average(a)
        b = b - np.average(b)
        pearson_sim = np.sum(a * b) / (np.sqrt(np.sum(a**2))
                                       * np.sqrt(np.sum(b**2)))
        return pearson_sim

    def tokenSimilarity(self, str_a, str_b, method='w2v', sim='cos'):
        '''
        基于分词求相似度，默认使用cos_sim 余弦相似度,默认使用前20个最频繁词项进行计算
        method: w2v, tfidf, fasttext
        sim: cos, pearson, eucl
        '''
        assert method in ['w2v', 'tfidf', 'fasttext']
        assert sim in ['cos', 'pearson', 'eucl']
        a = self.tokenize(str_a)
        b = self.tokenize(str_b)
        if method == 'w2v':
            vec_a = wam(a, self.w2v_model)
            vec_b = wam(b, self.w2v_model)
        elif method == 'fasttest':
            vec_a = wam(str_a, self.fasttext)
            vec_b = wam(str_b, self.fasttext)
        else:
            vec_a = np.array(
                self.tfidf[self.dictionary.doc2bow(str_a.split())]).mean()
            vec_b = np.array(
                self.tfidf[self.dictionary.doc2bow(str_b.split())]).mean()
        if sim == 'cos':
            result = self.cos_sim(vec_a, vec_b)
        elif sim == 'person':
            result = self.pearson_sim(vec_a, vec_b)
        else:
            result = self.eucl_sim(vec_a, vec_b)

        return result

    def generate_all(self, str1, str2):
        return {
            'lcs':
            self.lcs(str1, str2),
            'edit_dist':
            self.editDistance(str1, str2),
            'jaccard':
            self.JaccardSim(str1, str2),
            'bm25':
            self.bm25.cal_bm25(str1, str2),
            'w2v_cos':
            self.tokenSimilarity(str1, str2, method='w2v', sim='cos'),
            'w2v_eucl':
            self.tokenSimilarity(str1, str2, method='w2v', sim='eucl'),
            'w2v_pearson':
            self.tokenSimilarity(str1, str2, method='w2v', sim='pearson'),
            'w2v_wmd':
            self.tokenSimilarity(str1, str2, method='w2v', sim='wmd'),
            'fast_cos':
            self.tokenSimilarity(str1, str2, method='fasttest', sim='cos'),
            'fast_eucl':
            self.tokenSimilarity(str1, str2, method='fasttest', sim='eucl'),
            'fast_pearson':
            self.tokenSimilarity(str1,
                                 str2,
                                 method='fasttest',
                                 sim='pearson'),
            'fast_wmd':
            self.tokenSimilarity(str1, str2, method='fasttest', sim='wmd'),
            'tfidf_cos':
            self.tokenSimilarity(str1, str2, method='tfidf', sim='cos'),
            'tfidf_eucl':
            self.tokenSimilarity(str1, str2, method='tfidf', sim='eucl'),
            'tfidf_pearson':
            self.tokenSimilarity(str1, str2, method='tfidf', sim='pearson')
        }

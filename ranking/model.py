#!/usr/bin/env python
# coding=utf-8
'''
Author: yangyuxiang
Date: 2021-05-31 15:22:01
LastEditors: yangyuxiang
LastEditTime: 2021-06-11 09:12:26
FilePath: /Chinese-Dialogue-System/ranking/model.py
Description:
'''
import logging
import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (AdamW, BertModel, BertTokenizer,
                          get_linear_schedule_with_warmup,
                          BertConfig,
                          BertForSequenceClassification)
sys.path.append('..')
from utils.dataset import DataPrecessForSentence
from config import Config
tqdm.pandas()
logging.basicConfig(format='%(levelname)s - %(levelname)s : %(message)s',
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


class BertModelTrain(nn.Module):
    """
    The base model for training a matching network.
    """

    def __init__(self, pretrained_path=Config.pretrained_path):
        super(BertModelTrain, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_path, num_labels=2)
        self.device = torch.device(
            'cuda') if Config.is_cuda else torch.device('cpu')
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
#         logging.info("outputs: {}".format(outputs))

        loss, logits = outputs
        probabilities = nn.functional.softmax(logits, dim=-1)

        return loss, logits, probabilities


class BertModelPredict(nn.Module):
    """
    The base model for doing prediction using trained matching network.
    """

    def __init__(self, config_path=Config.config_path):
        super(BertModelPredict, self).__init__()
        config = BertConfig.from_pretrained(config_path)
        self.bert = BertForSequenceClassification(config)
        self.device = torch.device('cuda' if Config.is_cuda else 'cpu')

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,)
        logits = output[0]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities


class MatchNN(nn.Module):
    def __init__(self,
                 model_path=Config.bert_model,
                 vocab_path=Config.vocab_path,
                 data_path=os.path.join(Config.root_path, 'data/ranking/train.tsv'),
                 is_cuda=False,
                 max_sequence_length=128):
        super(MatchNN, self).__init__()
        self.vocab_path = vocab_path
        self.model_path = model_path
        self.data_path = data_path
        self.max_sequence_length = max_sequence_length
        self.is_cuda = is_cuda
        self.device = torch.device('cuda' if is_cuda else 'cpu')
        self.load_model()

    def load_model(self):
        self.model = BertModelPredict().to(self.device)
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            self.vocab_path, do_lower_case=True)
        self.dataPro = DataPrecessForSentence(
            self.bert_tokenizer, self.data_path, self.max_sequence_length)

    def predict(self, q1, q2):
        q1 = self.bert_tokenizer.tokenize(q1)
#         print(q1)
        q2 = self.bert_tokenizer.tokenize(q2)
#         print(q2)

        result = [self.dataPro.trunate_and_pad(q1,q2)]
#         print("result:", result)
        seqs = [i[0] for i in result]
        seq_ids = torch.Tensor([i[1] for i in result]).type(torch.long)
        seq_masks = torch.Tensor([i[2] for i in result]).type(torch.long)
        seq_segments = torch.Tensor([i[3] for i in result]).type(torch.long)
        if self.is_cuda:
            seq_ids = seq_ids.to(self.device)
            seq_masks = seq_masks.to(self.device)
            seq_segments = seq_segments.to(self.device)
#             logging.info("seqs: {}".format(seqs))
#             logging.info("seq_ids: {}".format(seq_ids))
#             logging.info("seq_masks: {}".format(seq_masks))
#             logging.info("seq_segments: {}".format(seq_segments))
        with torch.no_grad():
            logits, probabilities = self.model(input_ids=seq_ids,
                                               attention_mask=seq_masks,
                                               token_type_ids=seq_segments)
            probabilities = probabilities.cpu().detach().numpy()  # shape: (batch_size, num_labels)
            label = probabilities.argmax()
#             print('----------------------')
#             print(probabilities, probabilities.shape)
#             print(label)
            score = probabilities.tolist()[0][label]
        return label, score

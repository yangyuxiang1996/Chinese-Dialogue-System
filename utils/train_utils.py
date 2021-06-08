#!/usr/bin/env python
# coding=utf-8
'''
Author: yangyuxiang
Date: 2021-06-02 21:34:43
LastEditors: yangyuxiang
LastEditTime: 2021-06-03 08:18:03
FilePath: /Chinese-Dialogue-System/utils/train_utils.py
Description:
'''
import os
import torch
import torch.nn as nn
import logging
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('..')
from config import Config


def generate_sent_masks(enc_hiddens, source_lengths):
    """ Generate sentence masks for encoder hidden states.
    @param enc_hiddens (Tensor): encodings of shape (b, src_len, h), where b = batch size,
                                 src_len = max source length, h = hidden size. 
    @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.len = batch size
    @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                where src_len = max source length, b = batch size.
    """
    enc_masks = torch.zeros(enc_hiddens.shape[0], enc_hiddens.shape[1],
                            dtype=torch.float)
    for seq_id, seq_len in enumerate(source_lengths):
        enc_masks[seq_id, :seq_len] = 1

    return enc_masks


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.shape()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contigous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_mask * reshaped_tensor, dim=-1)
    result = result * reshaped_mask

    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)  # (bxnxm).bmm(bxmxp) -> (bxnxp)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)

    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contigous().float()
    return weighted_sum * mask


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    return (out_classes == targets).sum().item()


def train(model, dataloader, optimizer, epoch_number, max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.
    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    model.train()
    epoch_start = time.time()
    batch_time_avg = 0.0
    batch_loss = 0.0
    correct_preds = 0
    device = torch.device('cuda') if Config.is_cuda else torch.device('cpu')
    logging.info('epoch: {}, number of batches: {}'.format(
        epoch_number, len(dataloader)))
    for batch_index, batch_data in enumerate(tqdm(dataloader)):
        batch_start = time.time()
        batch_seqs, batch_seq_ids, batch_seq_masks, \
            batch_seq_segments, batch_labels = batch_data
        if batch_index == 0:
            logging.info("input: {}".format(batch_seqs))
            logging.info('input_ids: {}'.format(batch_seq_ids))
            logging.info('attention_masks: {}'.format(batch_seq_masks))
            logging.info('token_type_ids: {}'.format(batch_seq_segments))
            logging.info('labels: {}'.format(batch_labels))
        if batch_index > 2:
            continue
        if Config.is_cuda:
            batch_seq_ids = batch_seq_ids.to(device)
            batch_seq_masks = batch_seq_masks.to(device)
            batch_seq_segments = batch_seq_segments.to(device)
            batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        loss, logits, probabilities = model(input_ids=batch_seq_ids,
                                            attention_mask=batch_seq_masks,
                                            token_type_ids=batch_seq_segments,
                                            labels=batch_labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        batch_loss += loss.item()
        correct_preds += correct_predictions(probabilities, batch_labels)
        if batch_index % 1 == 0:
            description = "batch: {}, Avg. batch proc. time: {:.4f}s, loss: {:.4f}".format(
                batch_index, batch_time_avg / (batch_index+1), batch_loss / (batch_index+1))
            logging.info(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = batch_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy


def test(model, dataloader):
    model.eval()
    device = torch.device('cuda') if Config.is_cuda else torch.device('cpu')
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    with torch.no_grad():
        for index, batch_data in enumerate(tqdm(dataloader)):
            batch_start = time.time()
            batch_seqs, batch_seq_ids, batch_seq_masks, \
                batch_seq_segments, batch_labels = batch_data
            if Config.is_cuda:
                batch_seq_ids = batch_seq_ids.to(device)
                batch_seq_masks = batch_seq_masks.to(device)
                batch_seq_segments = batch_seq_segments.to(device)
                batch_labels = batch_labels.to(device)

            _, _, probabilities = model(input_ids=batch_seq_ids,
                                        attention_mask=batch_seq_masks,
                                        token_type_ids=batch_seq_segments,
                                        labels=batch_labels)
            accuracy += correct_predictions(probabilities, batch_labels)
            batch_time += time.time() - batch_start
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, \
        roc_auc_score(all_labels, all_prob)


def validate(model, dataloader):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for index, batch_data in enumerate(dataloader):
            # Move input and output data to the GPU if one is used.
            if index > 2:
                continue
            batch_seqs, batch_seq_ids, batch_seq_masks, \
                batch_seq_segments, batch_labels = batch_data
            seq_ids = batch_seq_ids.to(device)
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)
            loss, logits, probabilities = \
                model(seq_ids, masks, segments, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probabilities, labels)
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return epoch_time, epoch_loss, epoch_accuracy, roc_auc_score(
        all_labels, all_prob)

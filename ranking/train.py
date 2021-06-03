#!/usr/bin/env python
# coding=utf-8
'''
Author: yangyuxiang
Date: 2021-06-02 16:31:46
LastEditors: yangyuxiang
LastEditTime: 2021-06-03 08:10:21
FilePath: /Chinese-Dialogue-System/ranking/train.py
Description:
'''
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers.optimization import AdamW
import sys
sys.path.append('..')
from config import Config
from model import BertModelTrain
from utils.dataset import DataPrecessForSentence
from utils.train_utils import train, validate

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    datefmt="%H:%M:%S",
                    level=logging.INFO)

seed = 9
torch.manual_seed(seed)
if Config.is_cuda:
    torch.cuda.manual_seed_all(seed)


def main(train_file=os.path.join(Config.root_path, 'data/ranking/train.tsv'),
         dev_file=os.path.join(Config.root_path, 'data/ranking/dev.tsv'),
         model_path=Config.bert_model,
         epochs=10,
         batch_size=32,
         lr=2e-05,
         patience=3,
         max_grad_norm=10.0,
         checkpoint=None):
    logging.info(20 * "=" + " Preparing for training " + 20 * "=")
    bert_tokenizer = BertTokenizer.from_pretrained(Config.vocab_path,
                                                   do_lower_case=True)
    device = torch.device("cuda") if Config.is_cuda else torch.device("cpu")
    if not os.path.exists(os.path.dirname(model_path)):
        os.mkdir(os.path.dirname(model_path))
    logging.info("\t* Loading training data...")
    train_dataset = DataPrecessForSentence(bert_tokenizer=bert_tokenizer,
                                           file=train_file,
                                           max_char_len=Config.max_seq_len)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,)
    logging.info("\t* Loading validation data...")
    dev_dataset = DataPrecessForSentence(bert_tokenizer=bert_tokenizer,
                                         file=dev_file,
                                         max_char_len=Config.max_seq_len)
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                batch_size=batch_size,
                                shuffle=True,)
    logging.info("\t* Building model...")
    model = BertModelTrain().to(device)

    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.85,
                                                           patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        logging.info(
            "\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
        # Compute loss and accuracy before starting (or resuming) training.
        _, valid_loss, valid_accuracy, auc = validate(model, dev_dataloader)
        logging.info("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, \
            auc: {:.4f}".format(valid_loss, (valid_accuracy * 100), auc))

    # -------------------- Training epochs ------------------- #
    logging.info("\n" + 20 * "=" +
                 "Training Bert model on device: {}".format(device)+20 * "=")
    patience_counter = 0
    for i in range(start_epoch, epochs+1):
        logging.info("* starting training epoch {}".format(i))
        train_time, train_loss, train_acc = train(model=model,
                                                  dataloader=train_dataloader,
                                                  optimizer=optimizer,
                                                  epoch_number=i,
                                                  max_gradient_norm=max_grad_norm)
        train_losses.append(train_loss)
        logging.info("-> Training time: {:.4f}s, loss = {:.4f}, \
            accuracy: {:.4f}%".format(train_time, train_loss, (train_acc * 100)))

        logging.info("* Validation for epoch {}:".format(i))
        val_time, val_loss, val_acc, score = validate(model=model,
                                                      dataloader=dev_dataloader)
        valid_losses.append(val_loss)
        logging.info("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, \
            auc: {:.4f}\n".format(val_time, val_loss, (val_acc * 100), score))
        scheduler.step(val_acc)
        # Early stopping on validation accuracy.
        if val_acc < best_score:
            patience_counter += 1
        else:
            best_score = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": i,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses
                }, model_path)
        if patience_counter >= patience:
            logging.info("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    main(epochs=Config.epochs, batch_size=Config.batch_size, lr=Config.lr)

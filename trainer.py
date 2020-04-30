import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import os
import time
from glob import glob

import torch
import colored_traceback
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.corpus.corpus import ClassificationCorpus

from src.utils.logger import Logger
# from src.utils.ops import np_softmax

# from src.train import Trainer
# from src.optim.optim import OptimWithDecay
from src import config

# from src.models.classifier import Classifier

# from src.layers.pooling import PoolingLayer

from base_args import base_parser, CustomArgumentParser


class BaseTrainer:
    def __init__(self, model, corpus, optimizer, loss_function, print_period):
        self.model = model
        self.corpus = corpus
        self.model.cuda()
        self.model.eval()
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_data = corpus.train_batches
        self.test_data = corpus.test_batches
        self.print_period = print_period


class Trainer(BaseTrainer):
    def __init__(self, model, corpus, optimizer, loss_function, print_period=100):
        super(Trainer, self).__init__(model, corpus, optimizer, loss_function)

    def train(self, epoch, print_period=None):
        total_loss = 0
        start = time.time()
        self.train_data.shuffle_examples()
        for batchId, batchIter in enumerate(self.train_data.gen()):
            x = batchIter["sequences"]
            y = batchIter["labels"]
            x = torch.LongTensor(x)
            y = torch.LongTensor(y)
            x, y = Variable(x.cuda()), Variable(y.cuda())

            output = self.model(x)
            # 将数据转换为两列概率的形式
            output = output.view(-1, 2)
            y = y.view(-1)

            loss = self.loss_function(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.data
            if batchId % print_period:
                print('batchid:{} time:{}s loss:{}'.format(
                    batchId, time.time() - start, total_loss/print_period))
                total_loss = 0
                start = time.time()

    def test(self):
        self.test_data.shuffle_examples()
        start = time.time()
        total_loss = 0
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
        batch_counter = 0
        for batchIter in self.test_data.gen():
            x = batchIter["sequences"]
            y = batchIter["labels"]
            x = torch.LongTensor(x)
            y = torch.LongTensor(y)

            x, y = Variable(x.cuda()), Variable(y.cuda())
            prediction = self.model(x)
            loss = self.loss_function(prediction, y)
            total_loss += loss
            prediction[prediction[:, 1] > 0.5] = 1
            prediction[prediction[:, 1] <= 0.5] = 0
            prediction = prediction[:, 1].bool()
            # print(prediction)
            y = y.bool()
            true_pos += (y * prediction).sum()
            false_neg += (y * (~prediction)).sum()
            false_pos += ((~y) * prediction).sum()
            true_neg += ((~y) * (~prediction)).sum()
            # print(true_pos)

            batch_counter += 1
        self.evaluate(true_pos, false_pos, true_neg, false_neg,
                      start, total_loss, batch_counter)

    def evaluate(self, true_pos, false_pos, true_neg, false_neg, since, total_loss, batch_counter):
        print('-' * 15)
        print('TP: %d\nTN: %d\nFP: %d\nFN: %d\nmark_counter: \n' %
              (true_pos, true_neg, false_pos, false_neg))
        print('-' * 15)
        print('time: %.3f s\n loss: %.4f\n precision: %.4f\nrecall: %.4f\nf score: %.4f\naccuracy: %.4f'
              % (
                  time.time() - since,
                  total_loss / batch_counter,
                  true_pos / (true_pos + false_pos + 1e-6),
                  true_pos / (true_pos + false_neg + 1e-6),
                  2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6),
                (true_pos + true_neg) /
                  (true_neg + true_pos + false_neg + false_pos)
              ))
        print('-' * 15)
        return 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)

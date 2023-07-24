# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Training Config & Helper Classes  """
import math
import os
import json
from typing import NamedTuple

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torchmetrics.functional as tm
import tensorflow as tf

import checkpoint


class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, data_iter, optimizer, save_dir, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.data_iter = data_iter # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device # device name

    def train(self, get_loss, model_file=None, pretrain_file=None, data_parallel=True):
        """ Train Loop """
        self.model.train() # train mode
        self.load(model_file, pretrain_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        writer = SummaryWriter('./logs')
        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            accuracy_sum = 0.
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
                los, accuracy = get_loss(model, batch, global_step) # mean() for Data Parallelism
                loss = los.mean()
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                accuracy_sum += accuracy
                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())

                # 每个batch 一画
                writer.add_scalar("APIfunctionCall_loss", loss.item(), global_step)
                writer.add_scalar("APIfunctionCall_accuracy", accuracy, global_step)

                if global_step % self.cfg.save_steps == 0: # save
                    self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step) # save and finish when global_steps reach total_steps
                    return

            # 每个epoch 一画
            writer.add_scalar("APIfunctionCallLoss_for_epoch", loss_sum/(i+1), e+1)
            writer.add_scalar("APIfunctionCallAccuracy_for_epoch", accuracy_sum/(i+1), e+1)
            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
        self.save(global_step)

    def eval(self, evaluate, model_file, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file, None)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = [] # prediction results
        TP ,TN, FP, FN = 0, 0, 0, 0
        iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                accuracy, result, label_pred, label_ids = evaluate(model, batch) # accuracy to print
                (tn, fp, fn, tp) = confusion_matrix(label_pred, label_ids, task="binary").ravel()
                TP += tp
                TN += tn
                FP += fp
                FN += fn
            results.append(result)
            iter_bar.set_description('Iter(acc=%5.3f)'%accuracy)
        matric(TP ,TN, FP, FN)
        return results

    def load(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

        elif pretrain_file: # use pretrained transformer
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'): # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                ) # load only transformer parts


    def save(self, i):
        """ save current model """
        torch.save(self.model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))


def recall(TP, TN, FP, FN):
    return TP / (TP + FN)

def precision(TP, TN, FP, FN):
    return TP / (TP + FP)

# def auc(TP, TN, FP, FN):
#     return TP

# 当MCC=0时表明模型不比随机预测好
def MCC(TP, TN, FP, FN):
    fenzi = TP * TN - FP * FN
    fenmu = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return fenzi / fenmu

def F1_score(TP, TN, FP, FN):
    return 2*TP/(2*TP+FP+FN)

def FNR(TP, TN, FP, FN):
    return FN/(FN + TP)

def FPR(TP, TN, FP, FN):
    return FP/(FP + TN)


def matric(TP, TN, FP, FN):
    TP, TN, FP, FN = TP.item(), TN.item(), FP.item(), FN.item()
    reca = recall(TP, TN, FP, FN)
    pre = precision(TP, TN, FP, FN)
    mcc = MCC(TP, TN, FP, FN)
    f1_score = F1_score(TP, TN, FP, FN)
    fnr = FNR(TP, TN, FP, FN)
    fpr = FPR(TP, TN, FP, FN)

    print("TP:", TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN:", FN)
    print("recall:", reca)
    print("precision:", pre)
    print("FNR:", fnr)
    print("FPR:", fpr)
    print("f1_score:", f1_score)
    print("MCC:", mcc)
    # print("Accurancy:",(TP + TN)/(TP + TN + FP + FN))

if __name__ == '__main__':
    matric(1899, 5994, 259, 293)

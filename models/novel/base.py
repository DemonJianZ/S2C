import abc
import torch
import os.path as osp
from dataloader.data_utils import *

from utils import (
    ensure_path,
    Averager, Timer, count_acc,
)

class Buffer(object):
    def __init__(self,num_features,size=200):
        self.size = size
        self.num_features = num_features
        self.buffer = torch.empty(0,self.num_features).cuda()
        self.buffer_label = torch.empty(0).cuda()

    def updata_buffer(self,data,logits,label):
        if data.shape[0] <= self.size:
            self.buffer = data
            self.buffer_label = label
        else:
            onehot = torch.zeros(logits.shape[0], logits.shape[1]).to(logits.device)
            label_ = label.unsqueeze(-1)
            onehot.scatter_(1, label_, 1)
            mask = onehot > 0
            judge_logits = torch.masked_select(logits, mask)
            topk_value, topk_index = torch.topk(judge_logits, judge_logits.size(0), dim=0)
            self.buffer = data[topk_index[-size:]]
            self.buffer_label = label[topk_index[-size:]]

    def is_buffer_zero(self):
        if self.buffer.shape[0] > 0:
            return False
        else:
            return True

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = set_up_datasets(self.args)
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * args.sessions


    @abc.abstractmethod
    def train(self):
        pass
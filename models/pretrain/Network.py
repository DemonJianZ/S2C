import argparse
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
import numpy as np


class MYNET(nn.Module):

    def __init__(self, args,mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(True, args)
            self.prompt_num = 20
            self.gcn_fea_num = 20
            self.topk_num = 2
            self.prompt_feature = 64
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        nn.init.orthogonal_(self.fc.weight)

        self.dummy_orthogonal_classifier = nn.Linear(self.num_features, self.pre_allocate - self.args.pretrain_class,
                                                     bias=False)
        self.dummy_orthogonal_classifier.weight.requires_grad = False
        self.dummy_orthogonal_classifier.weight.data = self.fc.weight.data[self.args.pretrain_class:, :]
        print(self.dummy_orthogonal_classifier.weight.data.size())
        print('self.dummy_orthogonal_classifier.weight initialized over.')


    def forward(self, data):
        # get sample features

        x = self.get_feature(data)

        x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
        x2 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))

        x = torch.cat([x1[:, :self.args.pretrain_class], x2], dim=1)

        x = self.args.temperature * x

        return x



    def forpass_fc(self, data):
        x = self.get_feature(data)

        x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
        x = self.args.temperature * x

        return x

    def predict(self,data,test_class):
        with torch.no_grad():
            # get sample features
            x = self.get_feature(data)

            x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x2 = F.linear(F.normalize(x, p=2, dim=-1),
                          F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))

            x = torch.cat([x1[:, :self.args.pretrain_class], x2], dim=1)
            x = self.args.temperature * x

            return x

    def get_feature(self, data):

        # get sample features
        x = self.encoder(data)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)

        return x

    def get_feature_test(self, data):
        # get sample features

        x = self.get_feature(data)

        return x


    def pre_encode(self, x):

        if self.args.dataset in ['cifar100', 'manyshotcifar']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)

        elif self.args.dataset in ['mini_imagenet', 'manyshotmini', 'cub200']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)

        return x

    def post_encode(self, x):
        if self.args.dataset in ['cifar100', 'manyshotcifar']:

            x = self.encoder.layer3(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        elif self.args.dataset in ['mini_imagenet', 'manyshotmini', 'cub200']:

            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        logits = self.get_logits(x, self.fc.weight)

        return logits


    def update_fc(self, dataloader, class_list, session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            x = self.get_feature(data)

            data = x.detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

    def update_fc_avg(self, data, label, class_list):
        new_fc = []
        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index] = proto
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.pretrain_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.pretrain_class + self.args.way * (session - 1):self.args.pretrain_class + self.args.way * session, :].copy_(new_fc.data)

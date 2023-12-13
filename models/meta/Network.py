from torch.nn.parameter import  Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
import numpy as np

class doubleexp_loss(torch.nn.Module):
    def __init__(self,scale=5): 
        super(doubleexp_loss, self).__init__()
        self.scale = scale

    def forward(self,x): 
        return torch.exp(self.scale * x)+torch.exp(-self.scale * x)-2

class MLP(nn.Module):
    def __init__(self,in_dim,hidden = 96, ratio=[2,2,1,1]):
        super(MLP, self).__init__()
        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_dim,
                                              out_channels=hidden*ratio[0],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden*ratio[0]),
                                    nn.LeakyReLU())
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=hidden*ratio[0],
                                              out_channels=hidden*ratio[1],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden*ratio[1]),
                                    nn.LeakyReLU())
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=hidden * ratio[1],
                                              out_channels=hidden * ratio[2],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden * ratio[2]),
                                    nn.LeakyReLU())
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=hidden * ratio[2],
                                              out_channels=hidden * ratio[3],
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=hidden * ratio[3]),
                                    nn.LeakyReLU())
        self.conv_last = nn.Conv2d(in_channels=hidden * ratio[3],
                                              out_channels=1,
                                              kernel_size=1)

    def forward(self,X):

        dims = len(X.shape)
        if dims == 2:
            x_i = X.unsqueeze(1)
            x_j = torch.transpose(x_i, 0, 1)
            x_ij = (F.normalize(x_i, p=2, dim=-1) * F.normalize(x_j, p=2, dim=-1)).unsqueeze(0)
            x_ij = torch.transpose(x_ij, 1, 3).to(self.conv_last.weight.device)
            matrix = self.conv_last(self.conv_4(self.conv_3(self.conv_2(self.conv_1(x_ij))))).squeeze(1).squeeze(0)
            edge_mask = torch.eye(matrix.size(0)).to(matrix.device)
            matrix_adj = F.softmax(matrix * (1 - edge_mask), -1) + edge_mask
        elif dims == 3:
            x_i = X.unsqueeze(2)
            x_j = torch.transpose(x_i, 1, 2)
            x_ij = (F.normalize(x_i, p=2, dim=-1) * F.normalize(x_j, p=2, dim=-1))
            x_ij = torch.transpose(x_ij, 1, 3).to(self.conv_last.weight.device)
            matrix = self.conv_last(self.conv_4(self.conv_3(self.conv_2(self.conv_1(x_ij))))).squeeze(1)
            edge_mask = torch.eye(matrix.size(1)).to(matrix.device)
            matrix_adj = F.softmax(matrix * (1 - edge_mask), -1) + edge_mask

        return matrix_adj

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=True,bias=True,):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.matmul(input, self.weight)
        output = torch.matmul(adj, output)
        if self.bias is not None:
            output = output + self.bias

        if self.residual:
            output = F.relu(output) + input
            return output
        else:
            return F.relu(output)

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)
            self.prompt_num = 20
            self.gcn_fea_num = 20
            self.topk_num = 2
            self.prompt_feature = 2
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        nn.init.orthogonal_(self.fc.weight)

        self.prom_classifier = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        nn.init.orthogonal_(self.prom_classifier.weight)

        self.gcn_feature  = nn.Linear(self.num_features, self.gcn_fea_num, bias=False)
        nn.init.orthogonal_(self.gcn_feature.weight)
        self.act = doubleexp_loss()


        self.mlp_1 = MLP(self.num_features)
        self.instance_gcn = GraphConvolution(self.num_features, self.num_features)

        self.dummy_orthogonal_classifier = nn.Linear(self.num_features, self.pre_allocate - self.args.pretrain_class,
                                                     bias=False)
        self.dummy_orthogonal_classifier.weight.requires_grad = False
        self.dummy_orthogonal_classifier.weight.data = self.fc.weight.data[self.args.pretrain_class:, :]
        print(self.dummy_orthogonal_classifier.weight.data.size())
        print('self.dummy_orthogonal_classifier.weight initialized over.')



    def forward(self, data, train_label):
        # get sample features

        x = self.get_feature(data)

        x = x.unsqueeze(1)
        support_data = self.gcn_feature.weight.data
        support_data = support_data.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat([support_data, x], 1)
        x = self.ins_gcn(x)[:, -1, :].squeeze(1)


        x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.prom_classifier.weight, p=2, dim=-1))
        x2 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))

        x = torch.cat([x1[:, :self.args.meta_class], x2], dim=1)

        return x


    def predict(self,data,test_class):
        with torch.no_grad():
            x = self.get_feature(data)

            x = x.unsqueeze(1)
            support_data = self.gcn_feature.weight.data
            support_data = support_data.unsqueeze(0).repeat(x.size(0), 1, 1)
            x = torch.cat([support_data, x], 1)
            x = self.ins_gcn(x)[:, -1, :].squeeze(1)

            logits_x = self.get_logits(x, self.prom_classifier.weight)

            return logits_x


    def get_feature(self, data):
        # get sample features
        x = self.encoder(data)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def get_feature_test(self, data,test_class):
        # get sample features
        x = self.get_feature(data)

        x = x.unsqueeze(1)
        support_data = self.gcn_feature.weight.data
        support_data = support_data.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat([support_data, x], 1)
        x = self.ins_gcn(x)[:, -1, :].squeeze(1)

        return x


    def ins_gcn(self, x):

        x_edge = self.mlp_1(x)
        x = self.instance_gcn(x, x_edge)

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

    def post_encode(self,data, x):
        if self.args.dataset in ['cifar100', 'manyshotcifar']:

            x = self.encoder.layer3(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        elif self.args.dataset in ['mini_imagenet', 'manyshotmini', 'cub200']:

            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        x = x.unsqueeze(1)
        support_data = self.gcn_feature.weight.data
        support_data = support_data.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat([support_data, x], 1)
        x = self.ins_gcn(x)[:, -1, :].squeeze(1)


        logits = self.get_logits(x, self.prom_classifier.weight)

        return logits


    def update_fc(self, dataloader, class_list, session):

        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]

            x = self.get_feature(data)

            x = x.unsqueeze(1)
            support_data = self.gcn_feature.weight.data
            support_data = support_data.unsqueeze(0).repeat(x.size(0), 1, 1)
            x = torch.cat([support_data, x], 1)
            x = self.ins_gcn(x)[:, -1, :].squeeze(1).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(x, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,x,label,session)

    def update_fc_avg(self, data, label, class_list):
        new_fc = []
        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)
            self.prom_classifier.weight.data[class_index] = proto
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.prom_classifier.weight[:self.args.meta_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.prom_classifier.weight.data[self.args.meta_class + self.args.way * (session - 1):self.args.meta_class + self.args.way * session, :].copy_(new_fc.data)

    def get_logits(self,x,fc):
        return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))





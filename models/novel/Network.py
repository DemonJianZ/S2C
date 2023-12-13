from .base import Buffer
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *

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
            # x_ij = torch.abs(x_i - x_j).unsqueeze(0)
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
            self.num = 3
            self.prompt_num = 50
            self.prompt_feature = 32
            self.buffer_size = 200
            self.num_features = 512
            self.args.in_c = 512
            self.args.base_c = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, self.args.num_classes, bias=False)

        self.linear = nn.Linear(512, self.prompt_feature, bias=False)
        self.prompt = Parameter(torch.FloatTensor(self.prompt_num, self.prompt_feature))

        self.graph_node = Buffer(self.num_features,self.buffer_size)
        self.mlp_1 = MLP(self.num_features)
        self.mlp_2 = MLP(self.num_features+self.num*self.prompt_feature)
        self.proto_gcn = GraphConvolution(self.num_features+self.num*self.prompt_feature, self.num_features,False)
        # self.proto_gcn = GraphConvolution(self.num_features+self.num*self.prompt_feature, self.num_features+self.num*self.prompt_feature)
        self.instance_gcn = GraphConvolution(self.num_features,self.num_features)

    def forward(self, train_data, train_label, session):

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):

                # get sample features
                few_data = self.get_feature(train_data)

                data = torch.cat([few_data, self.graph_node.buffer], dim=0)

                # instance gcn
                data_features = self.ins_gcn(data)
                sample_feature = data_features[:few_data.shape[0]]
                sampel_label = train_label
                buffer_feature = data_features[few_data.shape[0]:]
                buffer_label = self.graph_node.buffer_label
                _,_,feature = self.pro_gcn(sample_feature, buffer_feature, sampel_label, buffer_label)

                # update buffer
                label = torch.cat([sampel_label, buffer_label], dim=0)
                logits = self.get_logits(data_features, self.fc.weight)
                self.graph_node.updata_buffer(data, logits, label)

                new_fc = self.fc.weight[self.args.meta_class + self.args.way * (session - 1):, :]
                # new_fc = feature[few_data.shape[0]:few_data.shape[0]+5, :]
                new_fc = new_fc.clone().detach()
                new_fc.requires_grad = True
                optimized_parameters = [{'params': new_fc}]
                optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new, momentum=0.9, dampening=0.9,
                                            weight_decay=0)

                few_feature = feature[:few_data.shape[0]].detach()

                old_fc = self.fc.weight[:self.args.meta_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                few_logits = self.get_logits(few_feature, fc)
                loss = F.cross_entropy(few_logits, sampel_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # self.fc.weight.data[self.args.meta_class + self.args.way * (session - 1):self.args.meta_class + self.args.way * (session), :].copy_(new_fc.data)
        self.fc.weight.data[self.args.meta_class + self.args.way * (session - 1):, :].copy_(new_fc.data)


    # def forward(self,data, label, session):
    #
    #     # get sample features
    #     few_data = self.get_feature(data)
    #
    #     data = torch.cat([few_data, self.graph_node.buffer], dim=0)
    #
    #     # instance gcn
    #     data_features = self.ins_gcn(data)
    #     sample_feature = data_features[:few_data.shape[0]]
    #     sampel_label = label
    #     buffer_feature = data_features[few_data.shape[0]:]
    #     buffer_label = self.graph_node.buffer_label
    #     feature = self.pro_gcn(sample_feature, buffer_feature, sampel_label, buffer_label)
    #
    #     # update buffer
    #     label = torch.cat([sampel_label, buffer_label], dim=0)
    #     logits = self.get_logits(data_features, self.fc.weight)
    #     self.graph_node.updata_buffer(data, logits, label)
    #
    #     new_fc = feature[few_data.shape[0]:]
    #     new_fc = new_fc.clone().detach()
    #     new_fc.requires_grad = True
    #     optimized_parameters = [{'params': new_fc}]
    #     optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new, momentum=0.9, dampening=0.9,
    #                                 weight_decay=0)
    #
    #     few_feature = feature[:few_data.shape[0]].detach()
    #
    #     with torch.enable_grad():
    #         for epoch in range(self.args.epochs_new):
    #             old_fc = self.fc.weight[:self.args.meta_class + self.args.way * (session - 1), :].detach()
    #             fc = torch.cat([old_fc, new_fc], dim=0)
    #             few_logits = self.get_logits(few_feature, fc)
    #             loss = F.cross_entropy(few_logits, sampel_label)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #
    #     self.fc.weight.data[
    #     self.args.meta_class + self.args.way * (session - 1):self.args.meta_class + self.args.way * session, :].copy_(
    #         new_fc.data)


    def predict(self,data,test_label):
        # get sample features
        few_data = self.get_feature(data)

        print("few_data",few_data.shape)


        x = few_data.unsqueeze(1)
        buffer_data = self.graph_node.buffer.unsqueeze(0).repeat(x.size(0), 1, 1)
        data = torch.cat([buffer_data, x], 1)

        print("x", x.shape)
        print("buffer_data", buffer_data.shape)
        print("data", data.shape)

        # data = torch.cat([few_data, self.graph_node.buffer], dim=0)
        # instance gcn
        data_features = self.ins_gcn(data)
        sample_feature = data_features[:few_data.shape[0]]
        sampel_label = test_label
        buffer_feature = data_features[few_data.shape[0]:]
        buffer_label = self.graph_node.buffer_label

        pro_logits , pro_loss, _ = self.pro_gcn(sample_feature, buffer_feature, sampel_label, buffer_label)

        logits = self.get_logits(data_features, self.fc.weight)
        logits_ = logits[:few_data.shape[0],:]
        loss = F.cross_entropy(logits_, sampel_label)

        total_loss = loss + pro_loss

        return pro_logits, total_loss
        # return feature_edge, total_loss

    def get_feature(self, data):

        # get sample features
        x = self.encoder(data)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)

        return x

    def ins_gcn(self, x):

        x_edge = self.mlp_1(x)
        x = self.instance_gcn(x, x_edge)

        return x

    def pro_gcn(self, x,buffer_features,x_label,buffer_label):

        class_list = torch.unique(buffer_label)
        proto_fc = []
        for class_index in class_list:
            data_index = (buffer_label == class_index).nonzero().squeeze(-1)
            embedding = buffer_features[data_index]
            proto = embedding.mean(0)
            proto_fc.append(proto)

        proto_fc = torch.stack(proto_fc, dim=0)

        x_label_list = torch.unique(x_label)
        x_label_proto_fc = []
        for class_index in x_label_list:
            data_index = (x_label == class_index).nonzero().squeeze(-1)
            embedding = x[data_index]
            proto = embedding.mean(0)
            x_label_proto_fc.append(proto)

        x_label_proto_fc = torch.stack(x_label_proto_fc, dim=0)

        feature = torch.cat([x,x_label_proto_fc,proto_fc],dim=0)
        feature_p = self.linear(feature)
        prompt_logits = self.get_logits(feature_p,self.prompt)
        topk_value, topk_index = torch.topk(prompt_logits, self.num, dim = 1)
        pro_feature = self.prompt[topk_index].view(feature.shape[0],-1)
        feature = torch.cat([pro_feature,feature],dim=-1)
        # pro_gcn
        feature_edge = self.mlp_2(feature)
        feature = self.proto_gcn(feature, feature_edge)

        logits = self.get_logits(feature,self.fc.weight)

        train_label = torch.cat([x_label,x_label_list,class_list],dim=0)
        loss = F.cross_entropy(logits, train_label)

        return logits[:x.shape[0]],loss,feature


    def get_logits(self,x,fc):
        return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))


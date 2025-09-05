from collections import OrderedDict
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric import data as DATA
import numpy as np

device = torch.device('cuda:0')

'''CAB+dtf'''

# class Conv1dReLU(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super().__init__()
#         self.inc = nn.Sequential(
#             nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
#             nn.ReLU()
#         )
    
#     def forward(self, x):

#         return self.inc(x)
    
# '''
# --StackCNN--
# 首先创建一个包含一个有序字典的实例self.inc。这个有序字典中的第一项是一个名为'conv_layer0'的Conv1dReLU层
# 接下来使用for循环创建了layer_num - 1个Conv1dReLU层,每个层的名称是'conv_layer'加上对应的索引号。
# 最后,通过add_module方法添加了一个名为'pool_layer'的nn.AdaptiveMaxPool1d层,它是一个自适应的1维最大池化层,将特征图池化成长度为1的张量。

# 前向传播方法 接受输入张量x,并通过self.inc对其进行卷积和池化操作。最后,使用squeeze(-1)方法去除张量的最后一个维度,返回结果。
# 这个StackCNN模型包含了多个卷积层和一个池化层,可以用于处理1维的输入数据并提取特征。
# '''
# class StackCNN(nn.Module):
#     def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super().__init__()

#         self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
#         for layer_idx in range(layer_num - 1):
#             self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

#         self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

#     def forward(self, x):

#         return self.inc(x).squeeze(-1)


# class ProteinEncoder(nn.Module):
#     def __init__(self, block_num=3, vocab_size=25+1, embedding_num=128):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
#         self.block_list = nn.ModuleList()
#         for block_idx in range(1, block_num + 1):
#             self.block_list.append(
#                 StackCNN(block_idx, embedding_num, 128, 3)
#             )
#         self.linear = nn.Linear(block_num * 128, 128)
#         self.conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.advpool = nn.AdaptiveMaxPool1d(1)
        
#     def forward(self, v):
#         v = self.embed(v)
#         v = v.transpose(2, 1)
#         v = self.bn1(F.relu(self.conv1(v)))
#         v = self.bn2(F.relu(self.conv2(v)))
#         v = self.bn3(F.relu(self.conv3(v)))
#         v = self.advpool(v).squeeze(-1)
#         # print("===v",v.shape)
#         return v
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch
import torch.nn as nn

from transformers import EsmModel, EsmTokenizer
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

class ESM2Encoder(nn.Module):
    def __init__(self, target_dim=64):
        super(ESM2Encoder, self).__init__()
        # 使用 Hugging Face 的 ESM2 模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        self.model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
        
        # PCA用于将768维降至128维
        self.pca = PCA(n_components=target_dim)

    def forward(self, sequences):
        # 将输入蛋白质序列的张量形式转换为氨基酸字符串
        sequences = [" ".join(list(seq)) for seq in sequences]  # 氨基酸分隔
        sequences = [f"<cls> {seq} <sep>" for seq in sequences]  # 添加特殊标记

        # 对序列进行编码
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取每个序列的 <cls> 标记嵌入作为序列的全局表示 (batch_size, 768)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # 使用 PCA 将 768 维的嵌入降维到 128 维
        reduced_embeddings = self.pca.fit_transform(cls_embeddings)  # (batch_size, 128)
        return torch.tensor(reduced_embeddings).to(device)


class ProteinEncoder(nn.Module):
    def __init__(self, block_num=3, target_dim=64):
        super(ProteinEncoder, self).__init__()
        self.esm2_encoder = ESM2Encoder(target_dim=target_dim)

        # 线性层处理后的嵌入维度保持一致
        self.linear = nn.Linear(target_dim, 128)

    def forward(self, v):
        # 调用 ESM2 生成 128 维的嵌入
        v = self.esm2_encoder(v)

        # 通过线性层将输出调整到所需的维度 (保持和CNN输出一致)
        v = self.linear(v)
        print("check", v.shape)
        return v


'''
num_features:输入特征的通道数
eps:分母中加入的小量,以避免除数为零
momentum:动量因子,用于计算运行时均值和方差的滑动平均值,默认为0.1
affine:Boolean值,决定是否使用可学习的缩放因子和偏置项,默认为True
track_running_stats:Boolean值,决定是否跟踪训练过程中的运行时均值和方差,默认为True

构造函数首先调用父类_BatchNorm的__init__方法进行初始化。
然后定义了一个名为_check_input_dim的私有方法,用于检查输入数据是否是2D张量,并引发ValueError异常('expected 2D input (got {}D input)')。

前向传播方法接受输入特征input作为参数,并使用_check_input_dim方法检查输入特征是否是2D张量。
如果self.momentum为None,则指数平均因子exponential_average_factor为0.0;
否则,exponential_average_factor为self.momentum。
如果当前处于训练模式并且self.track_running_stats为True,则更新num_batches_tracked计数器并更新exponential_average_factor的值。
最后,调用torch.functional.F.batch_norm函数对输入特征进行批归一化操作,并返回结果。

extra_repr方法返回一个字符串,表示模型的参数设置。它返回的字符串格式为'num_features={num_features}, eps={eps}, affine={affine}',其中num_features、eps和affine是从对象的__dict__属性中获取的。
'''
class NodeLevelBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)

'''这个GraphConvBn模型包含了一个图卷积层和一个节点级别的批归一化层, 可以用于处理图数据并提取特征。'''
class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        data.x = x

        return data

'''
创建了两个成员变量self.conv1和self.conv2。
self.conv1是一个GraphConvBn层,它接受num_input_features个输入通道和growth_rate * bn_size个输出通道。
self.conv2是一个GraphConvBn层, 它接受growth_rate * bn_size个输入通道和growth_rate个输出通道。

在前向传播过程中,首先判断输入特征x的类型是否为Tensor。如果是Tensor类型,说明只有一个输入特征,将其转换为列表[x]。
然后,使用torch.cat对输入特征x进行拼接操作,按照第1维度进行拼接,得到一个新的特征张量。
接下来,将新的特征张量data.x传递给self.conv1进行图卷积和批归一化操作。
然后,将输出结果data传递给self.conv2进行图卷积和批归一化操作。
最终,将更新后的数据对象data作为输出返回。

这个DenseLayer模型包含了两个GraphConvBn层,可以用于实现稠密连接的图卷积网络结构。通过拼接输入特征和多次进行图卷积和批归一化操作,可以提取更丰富的特征表示。
'''
class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]
        data.x = torch.cat(data.x, 1)

        data = self.conv1(data)
        data = self.conv2(data)

        return data

'''
DenseBlock类实现了一个由多个DenseLayer组成的稠密块。
在前向传播过程中,每一层的输出特征都会被保留,并与之前的输出特征进行拼接,形成稠密连接的特征图。这种设计可以增强模型的特征重用和信息流动,提高模型的表达能力。
'''
class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=2):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)

    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data

class DrugEncoder(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config=[8, 8, 8], bn_sizes=[2, 2, 2], channels=128, r=4):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i+1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)

            self.features.add_module("transition%d" % (i+1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

        inter_channels = int(channels // r)

        self.att1 = nn.Sequential(
            nn.Linear(228, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 228),
            nn.BatchNorm1d(228)
        )
        self.att2 = nn.Sequential(
            nn.Linear(228, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 228),
            nn.BatchNorm1d(228)
        )
        self.att3 = nn.Sequential(
            nn.Linear(228, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 228),
            nn.BatchNorm1d(228)
        )
        # self.att2 = nn.Sequential(
        #     nn.Linear(channels, inter_channels),
        #     nn.BatchNorm1d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(inter_channels, channels),
        #     nn.BatchNorm1d(channels)
        # )

        self.sigmoid = nn.Sigmoid()
        self.fc_mg = nn.Linear(2048, 228)
        self.fc_ava = nn.Linear(512, 228)
        self.fc_maccs = nn.Linear(167,228)
        # self.do = nn.Dropout(dropout)

    def forward(self, data):
        data = self.features(data)
        graph_feature = gnn.global_mean_pool(data.x, data.batch)
        # print("graph dtype: ",graph_feature.dtype) #tensor folat32
        # print("graph_feature",graph_feature.shape) #([64,228])
        morgan = torch.tensor(data.morgan).float().to('cuda:0')
        # print("morgan dtype: ",morgan.dtype) #tensor folat32
        morgan = self.fc_mg(morgan)
        # print("morgan",morgan.shape) #([64,228])
        Avalon = torch.tensor(data.Avalon).float().to('cuda:0')
        Avalon = self.fc_ava(Avalon)
        maccs = torch.tensor(data.maccs).float().to('cuda:0')
        maccs = self.fc_maccs(maccs)


        # print("Avalon",Avalon.shape) 

        w1 = self.sigmoid(self.att1(graph_feature + morgan))
        # print("w1shape: ",w1.shape)
        gm_f = graph_feature * w1 + morgan * (1 - w1)

        w2 = self.sigmoid(self.att2(gm_f + Avalon))
        
        Agm_f = gm_f * w2 + Avalon * (1 - w2)

        w3 = self.sigmoid(self.att3(Agm_f+maccs))
        print("maccs join sucess")

        ammg_f = Agm_f*w3 + maccs*(1-w3)



        # print("fout1shape: ",fout1.shape)
        # print("w2shape: ",w2.shape)
        # print("fout2shape: ",fout2.shape)
        fout = self.classifer(ammg_f)

        # print("==out",fout2.shape)
        return fout

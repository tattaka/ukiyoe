import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

from .commons import Flatten, AdaptiveConcatPool2d, ASPP, JPU

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine

class SimpleNetHead(nn.Module):
    
    def __init__(self, encoder_channels, p=0.2, metric_branch=False, last_activation='Softmax', num_classes=10):
        super(SimpleNetHead, self).__init__()

        self.acp = AdaptiveConcatPool2d()
        self.flatten = Flatten()
        self.dense_layers = nn.Sequential(
            nn.BatchNorm1d(encoder_channels[0]*2),
            nn.Dropout(p),
            nn.Linear(encoder_channels[0]*2, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p))
        if metric_branch:
            self.last_layer=ArcMarginProduct(1024, num_classes)
        else:
            self.last_layer = nn.Linear(1024, num_classes)
        if last_activation is None:
            self.activation = last_activation
        elif last_activation == 'LogSoftmax':
            self.activation = nn.LogSoftmax(dim=1)
        else:
            raise ValueError('Activation should be "LogSoftmax"/None')
            
    def forward(self, feats):
        x = self.flatten(self.acp(feats[0]))
        logits = self.dense_layers(x)
        logits = self.last_layer(logits)
        if self.activation:
            logits = self.activation(logits)
        return logits
    
class ACPNetHead(nn.Module):
    
    def __init__(self, encoder_channels, p=0.2, metric_branch=False, last_activation='Softmax', num_classes=10):
        super(ACPNetHead, self).__init__()
        
        self.acp1 = AdaptiveConcatPool2d()
        self.acp2 = AdaptiveConcatPool2d()
        self.acp3 = AdaptiveConcatPool2d()
        self.flatten = Flatten()
        self.dense_layers = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d((encoder_channels[0]+encoder_channels[1]+encoder_channels[3])*2),
            nn.Dropout(p),
            nn.Linear((encoder_channels[0]+encoder_channels[1]+encoder_channels[3])*2, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p))
        if metric_branch:
            self.last_layer=ArcMarginProduct(512, num_classes)
        else:
            self.last_layer = nn.Linear(512, num_classes)
        if last_activation is None:
            self.activation = last_activation
        elif last_activation == 'LogSoftmax':
            self.activation = nn.LogSoftmax(dim=1)
        else:
            raise ValueError('Activation should be "LogSoftmax"/None')
            
    def forward(self, feats):
        x1 = self.acp1(feats[0])
        x2 = self.acp2(feats[1])
        x3 = self.acp3(feats[3])
        x = self.flatten(torch.cat([x1, x2, x3], 1))
        logits = self.dense_layers(x)
        logits = self.last_layer(logits)
        if self.activation:
            logits = self.activation(logits)
        return logits
    
class JPUNetHead(nn.Module):
    
    def __init__(self, encoder_channels, mid_channel=128, p=0.2, metric_branch=False, last_activation='softmax', num_classes=10):
        super(JPUNetHead, self).__init__()
            
        self.jpu = JPU([encoder_channels[0],encoder_channels[1],encoder_channels[2]], mid_channel)
        self.aspp = ASPP(mid_channel*4, mid_channel, dilations=[1, (1, 4), (2, 8), (3, 12)])
        self.acp = AdaptiveConcatPool2d()
        self.flatten = Flatten()
        self.dense_layers = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(mid_channel*10),
            nn.Dropout(p),
            nn.Linear(mid_channel*10, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p))
        if metric_branch:
            self.last_layer=ArcMarginProduct(512, num_classes)
        else:
            self.last_layer = nn.Linear(512, num_classes)
        if last_activation is None:
            self.activation = last_activation
        elif last_activation == 'LogSoftmax':
            self.activation = nn.LogSoftmax(dim=1)
        else:
            raise ValueError('Activation should be "LogSoftmax"/None')
            
    def forward(self, feats):
        x = self.jpu(feats[0], feats[1], feats[2])
        x = self.aspp(x)
        x = self.flatten(self.acp(x))
        logits = self.dense_layers(x)
        logits = self.last_layer(logits)
        if self.activation:
            logits = self.activation(logits)
        return logits
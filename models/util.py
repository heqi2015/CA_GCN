import json
import math
import sys

import torch
import torch.nn as nn
from torch.nn import init


class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0] * size[1]))
        return out.view(-1, size[0], size[1])


class XavierLinear(nn.Module):
    '''
    Simple Linear layer with Xavier init

    Paper by Xavier Glorot and Yoshua Bengio (2010):
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class OrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(OrthogonalLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.orthogonal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class BottledLinear(Bottle, nn.Linear):
    pass


class BottledXavierLinear(Bottle, XavierLinear):
    pass


class BottledOrthogonalLinear(Bottle, OrthogonalLinear):
    pass


def log(*args, **kwargs):
    print(file=sys.stdout, flush=True, *args, **kwargs)


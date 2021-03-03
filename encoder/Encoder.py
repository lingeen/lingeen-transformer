# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 12:46 PM
# @Author  : Kevin

import torch
import math
import torch.nn.functional as F

from torch import nn
import math
import copy
from encoder.LayerNorm import LayerNorm

def clones(module,N):
    '''
    对一个模块复制多个,封装成nn.ModuleList
    :param module:
    :param N:
    :return:
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):

    def __init__(self,encoder_layer,encoder_layer_num):

        super(Encoder,self).__init__()

        self.encoder_layers=clones(encoder_layer,encoder_layer_num)

        self.layer_norm=LayerNorm(encoder_layer.embedding_dim)


    def forward(self,input,mask):

        for encoder_layer in self.encoder_layers:

            input=encoder_layer(input,mask)

        return self.layer_norm(input)



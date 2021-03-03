# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 11:20 AM
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

class Decoder(nn.Module):

    def __init__(self,decoder_layer,decoder_layer_num):

        super(Decoder,self).__init__()

        self.decoder_layers=clones(decoder_layer,decoder_layer_num)

        self.layer_norm=LayerNorm(decoder_layer.embedding_dim)


    def forward(self,target_input,encoder_outputs,source_mask,target_mask):
        '''

        :param target_input: 目标数据的embedding表示,decoder的输入数据,也是输出
        :param encoder_outputs:
        :param source_mask:
        :param target_mask:
        :return:
        '''
        # 数据经过每个decoder_layer
        for decoder_layer in self.decoder_layers:
            target_input=decoder_layer(target_input,encoder_outputs,source_mask,target_mask)

        return self.layer_norm(target_input)
# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 12:46 PM
# @Author  : Kevin


from torch import nn
import math
import copy
from encoder.SubleyerConnection import SubleyerConnection

def clones(module,N):
    '''
    对一个模块复制多个,封装成nn.ModuleList
    :param module:
    :param N:
    :return:
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(nn.Module):

    def __init__(self,embedding_dim,self_attention_layer,feed_forward_layer,drop_out_ratio=0.1):
        super(EncoderLayer,self).__init__()

        self.attention_layer=self_attention_layer

        self.feed_forward_layer=feed_forward_layer

        self.sublayer_connection_layers=clones(SubleyerConnection(embedding_dim,drop_out_ratio),2)

        self.embedding_dim=embedding_dim


    def forward(self,input,mask):

        # torch.Size([1, 5, 512])
        # torch.Size([8, 5, 5])

        input=self.sublayer_connection_layers[0](input,lambda x:self.attention_layer(x,x,x,mask))

        encoder_unit_output=self.sublayer_connection_layers[1](input,lambda x:self.feed_forward_layer(x))

        return encoder_unit_output






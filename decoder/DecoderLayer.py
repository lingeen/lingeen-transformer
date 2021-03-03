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


class DecoderLayer(nn.Module):

    def __init__(self,embedding_dim,self_attention_layer,source_attention_layer,feed_forward_layer,drop_out_ratio=0.1):
        super(DecoderLayer,self).__init__()

        self.embedding_dim=embedding_dim

        self.self_attention_layer=self_attention_layer

        self.source_attention_layer=source_attention_layer

        self.feed_forward_layer=feed_forward_layer
        # 三个子对象,要连接3次
        self.sublayer_connection_layers=clones(SubleyerConnection(embedding_dim,drop_out_ratio),3)



    def forward(self,target_input,encoder_outputs,source_mask,target_mask):
        '''

        :param target_input: decoder的输入数据,即目标数据
        :param encoder_outputs:
        :param source_mask: 源数据掩码张量>解码器输入部分?
        :param target__mask: 目标数据掩码张量>解码器输出部分?
        :return:
        '''
        # 第一个子层是注意力,qkv都是decoder的,mask是target的mask,遮盖的是目标词
        target_input=self.sublayer_connection_layers[0](target_input,lambda x:self.self_attention_layer(x,x,x,target_mask))
        # q是decoder的q,kv是encoder的kv,mask是decoder的输入mask
        target_input=self.sublayer_connection_layers[1](target_input,lambda x:self.source_attention_layer(x,encoder_outputs,encoder_outputs,source_mask))

        decoder_unit_output=self.sublayer_connection_layers[2](target_input,lambda x:self.feed_forward_layer(x))

        return decoder_unit_output






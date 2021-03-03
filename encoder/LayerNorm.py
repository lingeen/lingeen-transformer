# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 2:32 PM
# @Author  : Kevin

from torch import nn
import math
import torch


class LayerNorm(nn.Module):
    '''
    层规范化:接收[batch size,seq len,embedding dim]
    防止多层计算后的数值过大或过小
    '''

    def __init__(self,embedding_dim,eps=1e-6):
        # eps小数,分母用,防止除0
        super(LayerNorm,self).__init__()
        # 准备一个全0,一个全1
        self.a2=nn.Parameter(torch.ones(embedding_dim))
        self.b2=nn.Parameter(torch.zeros(embedding_dim))

        self.eps=eps


    def forward(self,input):
        '''
        层规范化规范的是embedding dim
        减均值 除标准差
        减均值是不改变位置
        除标准差是不改变分布
        :param input:
        :return:
        '''

        input_meaned=input.mean(-1,keepdim=True)
        input_stded=input.std(-1,keepdim=True)

        input_normed=self.a2*(input-input_meaned)/(input_stded+self.eps) +self.b2

        return input_normed


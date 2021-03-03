# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 2:26 PM
# @Author  : Kevin

from torch import nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    '''
    加入了位置编码的前馈全连接层,接收注意力机制的结果[batch size,seq len,embedding dim]
    '''

    def __init__(self,embedding_dim,feed_forward_middle_dim,drop_out_ratio=0.1):
        super(PositionwiseFeedForward,self).__init__()

        self.linear_1=nn.Linear(embedding_dim,feed_forward_middle_dim)

        self.linear_2=nn.Linear(feed_forward_middle_dim,embedding_dim)

        self.drop_out=nn.Dropout(drop_out_ratio)


    def forward(self,input):
        # input > linear1 > relu > dropout > linear2
        return self.linear_2(self.drop_out(F.relu(self.linear_1(input))))
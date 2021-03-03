# -*- coding: utf-8 -*-
# @Time    : 2021/3/2 10:39 PM
# @Author  : Kevin
from torch import nn
import torch.nn.functional as F


class Output(nn.Module):

    def __init__(self,embedding_dim,vocab_size):
        super(Output,self).__init__()
        self.linear=nn.Linear(embedding_dim,vocab_size)


    def forward(self,input):

        return F.log_softmax(self.linear(input),dim=-1)





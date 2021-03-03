# -*- coding: utf-8 -*-
# @Time    : 2021/3/2 10:39 PM
# @Author  : Kevin
from torch import nn
import math


class Embedding(nn.Module):

    def __init__(self,vocab_size,embedding_dim):
        super(Embedding,self).__init__()
        self.embedding_dim=embedding_dim
        self.embedding=nn.Embedding(vocab_size,embedding_dim)


    def forward(self,input):

        return self.embedding(input)*math.sqrt(self.embedding_dim)





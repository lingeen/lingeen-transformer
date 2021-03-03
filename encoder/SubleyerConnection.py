# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 2:49 PM
# @Author  : Kevin

from torch import nn
from encoder.LayerNorm import LayerNorm

class SubleyerConnection(nn.Module):

    def __init__(self,embedding_dim,drop_out_ratio=0.1):
        super(SubleyerConnection,self).__init__()

        self.norm=LayerNorm(embedding_dim=embedding_dim)

        self.drop_out=nn.Dropout(drop_out_ratio)



    def forward(self,input,sublayer_fn):

        return input+self.drop_out(sublayer_fn(self.norm(input)))


# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 3:34 PM
# @Author  : Kevin

from torch import nn
from input.Embedding import Embedding
from input.PositionEmbedding import PositionEmbedding


class Input(nn.Module):

    def __init__(self,vocab_size,embedding_dim,drop_out_ratio=0.1):
        super(Input,self).__init__()
        self.embedding_layer=Embedding(vocab_size,embedding_dim)
        self.position_embedding_layer=PositionEmbedding(embedding_dim,drop_out_ratio)

    def forward(self,input):

        return self.position_embedding_layer(self.embedding_layer(input))
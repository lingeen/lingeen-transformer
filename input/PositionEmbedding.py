# -*- coding: utf-8 -*-
# @Time    : 2021/3/2 10:44 PM
# @Author  : Kevin
import torch
from torch import nn
from torch.autograd import Variable
import math


class PositionEmbedding(nn.Module):

    def __init__(self,embedding_dim,drop_out_ratio,max_seq_len=5000):

        super(PositionEmbedding,self).__init__()

        self.drop_out=nn.Dropout(drop_out_ratio)
        # [max_seq_len,embedding_dim]
        self.position_embedding_context=torch.zeros(max_seq_len,embedding_dim)
        # [max_seq_len,1]
        self.position_content=torch.arange(0,max_seq_len).unsqueeze(1)
        # 位置信息position_vector赋值给position_embedding
        # [max_seq_len,1]*[1,embedding_dim]+[max_seq_len,embedding_dim]
        # sin([max_seq_len,1]*[1,embedding_dim/2])赋值给[max_seq_len,embedding_dim]偶数列
        # cos([max_seq_len,1]*[1,embedding_dim/2])赋值给[max_seq_len,embedding_dim]奇数列
        # 初始一半,两种变换
        transfer_matrix=torch.exp(torch.arange(0,embedding_dim,2)*-(math.log(10000.0)/embedding_dim))
        # [,0::2]行不管,列从0开始步长2>0也是偶数
        self.position_embedding_context[:,0::2]=torch.sin(self.position_content*transfer_matrix)
        self.position_embedding_context[:,1::2]=torch.cos(self.position_content*transfer_matrix)

        # self.position_encode_embedding[max_seq_len,embedding_dim]>[batch size,seq len,embedding dim]
        # PositionEmbedding接收Embedding的输出[batch size,seq len,embedding dim],并与之相加
        self.position_embedding_context=self.position_embedding_context.unsqueeze(0)
        # position_embedding不需要更新,只是给Embedding的输出打个位置标签>固化position_embedding
        self.register_buffer("position_embedding",self.position_embedding_context)
        

    def forward(self,embedding_output):
        # position_embedding的seq max len,要适配input的seq real len
        embedding_output_seq_len=embedding_output.size(1)

        position_embedding_output=embedding_output+Variable(self.position_embedding[:,:embedding_output_seq_len],requires_grad=False)

        return self.drop_out(position_embedding_output)

# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 1:18 PM
# @Author  : Kevin

from torch import nn
import math
import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def clones(module,N):
    '''
    对一个模块复制多个,封装成nn.ModuleList
    :param module:
    :param N:
    :return:
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query,key,value,mask=None,drop_out=None):
    # query[batch size,head_num,seq len,head_dim]
    embedding_dim=query.size(-1)
    # 1.q*kt / dim >softmax>*v
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(embedding_dim)
    # 2.scores使用mask
    if mask is not None:
        # mask盖住scores>太小就不可能被选中
        scores=scores.masked_fill(mask==0,value=-1e9)

    # 3.scores>softmax
    attention_weight=F.softmax(scores,dim=-1)
    # 4.*v前忽视一些>注意的时候两次忽视,整段mask softmax之前,部分drop softmax之后>m s d
    if drop_out is not None:
        attention_weight=drop_out(attention_weight)

    # 5.attention_weight[batch size,seq len,seq len]*value[batch size,seq len,embedding dim]
    attention_result=torch.matmul(attention_weight,value)

    return attention_result,attention_weight


class MultiHeadedAttention(nn.Module):

    def __init__(self,head_num,embedding_dim,drop_out_ratio=0.1):
        super(MultiHeadedAttention,self).__init__()
        # 确认词可被整分

        assert embedding_dim % head_num == 0,"embedding_dim不能被传入的head_num整除"

        self.head_dim=embedding_dim//head_num

        self.head_num=head_num

        self.linears=clones(nn.Linear(embedding_dim,embedding_dim), 4)

        self.attention_result=None

        self.drop_out=nn.Dropout(drop_out_ratio)


    def forward(self,query,key,value,mask=None):
        '''
        qkv>内部片段的qkv>融合成一个attention_result
        :param query:
        :param key:
        :param value:
        :param mask:
        :return:
        '''
        # mask[mask_num=head_num,size,size]>
        if mask is not None:
            # mask[size,1,size]>代表第n个头,即第几段
            mask=mask.unsqueeze(1)
        batch_size=query.size(0)
        # qkb和自己的lienar配对, 并且配对完就立即过渡成新的qkv
        # 过渡完就立即把embedding dim切割,并且原形状[batch size,seq len,embedding dim]变成[batch size,seq len,head_num,head_dim]
        # [batch size,seq len,head_num,head_dim]再变成[batch size,head_num,seq len,head_dim]
        # 原本研究的是seq和embedding dim的关系,变成研究seq和head dim的关系>整体变片段
        query,key,value=[linear(x).view(batch_size,-1,self.head_num,self.head_dim).transpose(1,2) for linear,x in  zip(self.linears,(query,key,value))]
        # 原本完整的query,key,value可以直接进入attention,整段自注意,现在内部变化成多段,然后同样进入attention
        attention_result,attention_weight=attention(query,key,value,mask=mask,drop_out=self.drop_out)
        # attention_result[batch size,head_num,seq len,head_dim]
        # 再转回来[batch size,seq len,head_num,head_dim]>[batch size,seq len,head_num*head_dim]
        # transpose后需要contiguous才能进行view
        attention_result=attention_result.transpose(1,2).contiguous().view(batch_size,-1,self.head_num*self.head_dim)
        # 最后还原的attention_result还要进入一个linear过渡
        return self.linears[-1](attention_result)









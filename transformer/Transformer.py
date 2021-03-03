# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 5:43 PM
# @Author  : Kevin

from torch import nn
import copy
from encoder.MultiHeadedAttention import MultiHeadedAttention
from encoder.PositionwiseFeedForward import PositionwiseFeedForward
from input.PositionEmbedding import PositionEmbedding
from transformer.Encoder2Decoder import Encoder2Decoder
from encoder.Encoder import Encoder
from encoder.EncoderLayer import EncoderLayer
from decoder.Decoder import Decoder
from decoder.DecoderLayer import DecoderLayer
from input.Input import Input
from output.Output import Output
import torch
from torch.autograd import Variable



class Transformer(nn.Module):

    def __init__(self,source_vocab_size
                 ,target_vocab_size
                 ,encoder_layer_num=6
                 ,decoder_layer_num=6
                 ,embedding_dim=512
                 ,feed_forward_middle_dim=2048
                 ,head_num=8
                 ,drop_out_ratio=0.1):

        super(Transformer,self).__init__()

        self.c=copy.deepcopy
        self.attention_layer=MultiHeadedAttention(head_num,embedding_dim,drop_out_ratio)
        self.feed_forward_layer=PositionwiseFeedForward(embedding_dim,feed_forward_middle_dim,drop_out_ratio)
        self.position_embedding_layer=PositionEmbedding(embedding_dim,drop_out_ratio)

        # source_embedding_layer, target_embedding_layer, encoder, decoder, output_layer
        self.encoder_to_decoder=Encoder2Decoder(
            source_embedding_layer=Input(source_vocab_size,embedding_dim,drop_out_ratio),
            target_embedding_layer=Input(target_vocab_size,embedding_dim,drop_out_ratio),
            encoder=Encoder(EncoderLayer(embedding_dim,self.c(self.attention_layer),self.c(self.feed_forward_layer),drop_out_ratio),encoder_layer_num),
            decoder=Decoder(DecoderLayer(embedding_dim,self.c(self.attention_layer),self.c(self.attention_layer),self.c(self.feed_forward_layer),drop_out_ratio),decoder_layer_num),
            output_layer=Output(embedding_dim,target_vocab_size)
        )
        # 初始化transformer参数,遍历模型里的所有参数,如果维度大于1,则初始化成均匀分布矩阵 -1 - 1
        # 维度是1的话不是矩阵
        for params in  self.encoder_to_decoder.parameters():
            if params.dim()>1:
                nn.init.xavier_uniform(params)


    def forward(self,source_input, target_input, source_mask, target_mask):
        '''
        :param source_input:源数据,encoder的输入数据
        :param target_input:目标数据,decoder的输入数据
        :param source_mask:
        :param target_mask:
        :return:
        '''
        output=self.encoder_to_decoder(source_input, target_input, source_mask, target_mask)

        return output

if __name__ == '__main__':
    source_vocab_size=11
    target_vocab_size=11
    head_num=8
    source_input=torch.LongTensor()
    transformer=Transformer(source_vocab_size=source_vocab_size,target_vocab_size=target_vocab_size)

    # source_input, target_input, source_mask, target_mask
    source_input=torch.LongTensor([[1,2,3,4,5]])
    target_input=torch.LongTensor([[1,2,3,4,5]])

    source_mask=Variable(torch.zeros(head_num,5,5))
    target_mask=Variable(torch.zeros(head_num,5,5))

    result=transformer(source_input,target_input,source_mask,target_mask)

    print(result)
    print(result.size())

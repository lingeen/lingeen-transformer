# -*- coding: utf-8 -*-
# @Time    : 2021/3/3 6:05 PM
# @Author  : Kevin

from torch import nn
import math


class Encoder2Decoder(nn.Module):

    def __init__(self, source_embedding_layer, target_embedding_layer, encoder, decoder, output_layer):
        super(Encoder2Decoder, self).__init__()
        self.source_embedding_layer = source_embedding_layer
        self.target_embedding_layer = target_embedding_layer
        self.encoder = encoder
        self.decoder = decoder
        self.output_layer = output_layer

    def forward(self, source_input, target_input, source_mask, target_mask):
        '''

        :param source_input:源数据,encoder的输入数据
        :param target_input:目标数据,decoder的输入数据
        :param source_mask:
        :param target_mask:
        :return:
        '''
        return self.decode(self.encode(source_input, source_mask), source_mask, target_input, target_mask)

    def encode(self, source_input, source_mask):
        # source_input>source_embedding_layer>+source_mask>encoder
        return self.encoder(self.source_embedding_layer(source_input), source_mask)

    def decode(self, encoder_outputs, source_mask, target_input, target_mask):
        # target_input, encoder_outputs, source_mask, target_mask

        return self.decoder(self.target_embedding_layer(target_input), encoder_outputs, source_mask, target_mask)
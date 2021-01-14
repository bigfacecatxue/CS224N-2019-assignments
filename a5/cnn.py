#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, char_embed, word_embed, max_word_len, k=5):
        """
        @param k(int): kernel size
        @param word_embed(int): filter size, here is word embedding size
        @param char_embed(int): character embedding size
        @param max_word_len(int): max word length
        """
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(char_embed, word_embed, k)
        # self.relu = nn.ReLU()
        self.max_pool_1d = nn.MaxPool1d(max_word_len - k + 1)

    def forward(self, x_reshaped: torch.Tensor):
        """
        @param x_reshaped(Tensor): shape(max_sentence_length*batch_size, character_embedding_size, max_word_length)
        @return conv_out(Tensor): shape(max_sentence_length*batch_size, word_embedding_size)
        """

        x_conv = self.conv1d(x_reshaped) # shape(max_sentence_length*batch_size, word_embedding_size, max_word_length_k+1)
        # print(x_conv)
        # x_conv_out, _ = torch.max(F.relu(x_conv), 2)
        x_conv_out = self.max_pool_1d(F.relu(x_conv)).squeeze()


        return x_conv_out


# cnn = CNN(6, 3, 7)
# x_reshaped = torch.randn(2,6,7)
# conv_out = cnn.forward(x_reshaped)
# print(conv_out)



### END YOUR CODE


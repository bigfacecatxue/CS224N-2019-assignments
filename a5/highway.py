#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):

    def __init__(self, embed_size):
        """
        Init Highway Model
        @param embed_size(int): word embedding size (dimensionality)
        """

        super(Highway, self).__init__()
        self.projection = nn.Linear(embed_size, embed_size)
        self.gate = nn.Linear(embed_size, embed_size)
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, source: torch.Tensor):
        """
        @param source(Tensor): tensor of conv_out parameters, shape (batch, word_embedding_size)
        @return highway(Tensor): shape(batch, word_embedding_size)
        """
        x_proj = F.relu(self.projection(source))
        x_gate = F.sigmoid(self.gate(source))
        x_highway = x_gate * x_proj + (1 - x_gate) * source

        return x_highway


# test = Highway(embed_size=5)
# source = torch.randn(2, 5)
# print(test.forward(source).shape)


### END YOUR CODE 


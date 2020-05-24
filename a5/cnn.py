#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, in_channels, out_channels):
        """
        Init 1D convolutional layers.
        @param in_channels(int): the size of input dimension.
        @param out_channels(int): number of output channels.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=1)
    def forward(self, x):
        """
        Forward passing for the convolutional layers.
        @param x(Tensor): input Tensor, size: (batch_size, embedding_size)
        """
        conv_output = self.conv1(x)

        # The window size of max pooling layer of CNN depends on the dimension of conv1d output.
        # Since padding size is 1 and kernal size is 5, so the output of conv1d is with dimension
        # length_of_input_sequence - 2 + 5 - 1 = length_of_input_sequence - 2
        x_conv = F.max_pool1d(F.relu(conv_output), x.size()[-1] - 2)
        return x_conv



    ### END YOUR CODE


def main():
    """ Main func.
    """
    # seed the random number generators			
    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    cnn = CNN(2, 3)
    print(cnn.conv1.weight)
    print(cnn.conv1.bias)
    # batch size: 3, char embedding size: 2(input channel num), padded word length: 10
    result = cnn(torch.ones(3, 2, 10))
    print(result)

if __name__ == '__main__':
    main()
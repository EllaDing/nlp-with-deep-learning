#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, input_size):
    	"""
		Init the Highway layers. 
		@param input_size(int): The size of input Tensor.
    	"""
    	super(Highway, self).__init__()
    	self.input_size = input_size
    	self.W_proj = nn.Linear(
    		in_features=self.input_size,
    		out_features=self.input_size,
    		bias=True)
    	self.W_gate = nn.Linear(
    		in_features=self.input_size,
    		out_features=self.input_size,
    		bias=True)

    def forward(self, x_conv):
        """
        @param x_conv(Tensor): Input tensor, with size (batch_size, embedding_size)
        """
        x_proj = F.relu(self.W_proj(x_conv))
        x_gate = torch.sigmoid(self.W_gate(x_conv))
        return x_gate * x_proj + (1-x_gate) * x_conv
    ### END YOUR CODE

def main():
    """ Main func.
    """
    # seed the random number generators			
    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    highway = Highway(3)
    print(highway.W_proj.weight)
    print(highway.W_proj.bias)
    print(highway.W_gate.weight)
    print(highway.W_gate.bias)
    result = highway(torch.tensor([
    	[0.1, 0.2, 0.3],
    	[0.4, 0.5, 0.6]]))
    print(result)

if __name__ == '__main__':
    main()

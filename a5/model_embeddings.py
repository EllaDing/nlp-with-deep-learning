#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        # the output word embedding size is same as the CNN output channel's size.
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.embedding = nn.Embedding(len(self.vocab.char2id), 50, padding_idx=0)

        # char embedding size is 50.
        # The sequence length is self.word_embed_size which is used to construct max_pooling layer.
        print('word_embed_size:', self.word_embed_size)
        self.cnn = CNN(in_channels=50, out_channels=self.word_embed_size)
        self.dropout = nn.Dropout(0.3)
        self.highway = Highway(self.word_embed_size)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        highway_output = torch.Tensor(input.size()[0], input.size()[1], self.word_embed_size)
        embedding = self.embedding(input)
        embedding = embedding.contiguous().view(-1, embedding.size()[-2], embedding.size(-1))
        conv_output = self.cnn(embedding.permute(0, 2, 1))
        highway_output = self.highway(torch.squeeze(conv_output, 2))
        return highway_output.contiguous().view(input.size()[0], input.size()[1], self.word_embed_size)
        ### END YOUR CODE


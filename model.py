import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import torch.optim as optim


class TaggerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        super(TaggerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = self.init_hidden()

        # define layers
        ## word to embedding
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        ## embedding to hidden
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        ## hidden to output
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.cell = self.lstm(embeds.view(len(sentence), 1, -1), self.cell)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores
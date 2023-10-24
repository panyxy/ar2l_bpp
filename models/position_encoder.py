import sys
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

class PositionalEncoder(nn.Module):
    def __init__(self, sequence_len, embedding_size):
        super(PositionalEncoder, self).__init__()
        self.sequence_len = sequence_len
        self.embedding_size = embedding_size

        pos = torch.arange(0, sequence_len, 1).float().unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() / embedding_size * (-math.log(10000.0)))

        PE = torch.zeros(sequence_len, embedding_size)
        PE[:, 0::2] = torch.sin(pos*div_term)
        PE[:, 1::2] = torch.cos(pos*div_term)
        PE = PE.unsqueeze(dim=0)

        self.register_buffer('PE', PE)

    def forward(self, x,):
        x = x + Variable(self.PE[:, :x.size(1)], requires_grad=False)
        #x = x + self.PE[:, :x.size(1)].detach()
        return x


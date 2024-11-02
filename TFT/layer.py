import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.parameter import UninitializedParameter



class GRU(nn.Module):
    def __init__(self, input_size, 
                       hidden_size,
                       output_size = None,
                       context_size = None,
                       dropout_rate = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)


        self.GlU = GLU(hidden_size, hidden_size)

        self.droput = nn.Dropout(dropout_rate)

        self.layernorm = nn.LayerNorm()

        if output_size:
            self.linear_out = nn.Linear(hidden_size, output_size)

        if context_size:
            self.linear_conext = nn.Linear(context_size, hidden_size, bias=False)
    
    def forward(self, x, context = None):
        linear1 = self.linear1(x)
        if context:
            linear1 = linear1 + self.linear_conext(context).unsqueeze(1)
        linear2 = self.linear2(linear1)
        ELU = F.elu(linear2)
        output = self.layernorm(x + self.GLU(linear2))
        if self.output_size:
            output = self.linear_out(output)
        return output
    



class GLU(nn.Module):
    def __init__(self, input_size , hidden_size):
        super().__init__()
        self.linear4 = nn.Linear(input_size, hidden_size)
        self.linear5 = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        linear4 = self.linear4(x)
        linear5 = self.linear5(x)
        x = F.sigmoid(linear4) * linear5
        return x


    
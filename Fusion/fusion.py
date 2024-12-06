import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from .layer import *

class CrossFusion(nn.Module):
    def __init__(self, n_head, num_model, seq_list, hidden_size ,device):
        super().__init__()

        self.cross_list = nn.ModuleList(
                            [CrossAttention(n_head = n_head, q_seq_len= seq_list[0] , k_v_seq_len=seq_list[i+1]  ,hidden_size = hidden_size, device= device     )
                             for i in range(num_model-1)
                             ])
        
    def forward(self, model_output_list):
        q = model_output_list[0]

        for i, cross in enumerate(self.cross_list):
            q = cross(q, k=model_output_list[i+1], v=model_output_list[i+1])
        return q    
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class CrossAttention(nn.Module):
    def __init__(self, hidden_size,n_head,q_seq_len, k_v_seq_len, device):
        super().__init__()
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.attention_hidden_size = self.hidden_size//self.n_head
        self.div = self.attention_hidden_size**-0.5
        self.mask = torch.triu(torch.full((q_seq_len, k_v_seq_len), float('-inf')), 1).to(device)
        self.q_linear = nn.ModuleList(
                            [nn.Linear(hidden_size, self.attention_hidden_size)
                             for i in range(self.n_head)])

        self.k_linear = nn.ModuleList(
                            [nn.Linear(hidden_size, self.attention_hidden_size)
                             for i in range(self.n_head)])
        self.v_linear = nn.Linear(hidden_size, q_seq_len)

        self.attention_output_linear = nn.Linear(q_seq_len, self.hidden_size)
        self.q_residual = nn.Linear(hidden_size, self.hidden_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, q, k, v):

        q_list = [q_linear(q) for q_linear in self.q_linear]
        k_list = [torch.transpose(k_linear(k),1,2) for k_linear in self.k_linear]
        v = self.v_linear(v)
        attention_QK = [torch.matmul(q, k) for q,k in zip(q_list,k_list)]

        attention_QK = [F.softmax(torch.mul(attention, self.div) + self.mask, dim =-1) 
                        for i, attention in enumerate(attention_QK)]
        attention_QKV = [torch.matmul(attention, v) for attention in attention_QK]
        attention_QKV = torch.stack(attention_QKV, dim=-1)
        attention_QKV = torch.mean(attention_QKV, dim=-1) 


        q_residual = self.q_residual(q)
        attention_output = self.attention_output_linear(attention_QKV)
        q_sigmoid = self.sigmoid(q_residual)

        output = attention_output * q_sigmoid
        return output

class AttentionFusion(nn.Module):
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

class GLU(nn.Module):
    def __init__(self, input_size, 
                       hidden_size):
        super().__init__()

        self.linear4 = nn.Linear(input_size, hidden_size)
        self.linear5 = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        linear4 = self.linear4(x)
        linear5 = self.linear5(x)
        x = F.sigmoid(linear4) * linear5
        return x
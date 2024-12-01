import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.parameter import UninitializedParameter

class TFT_embedding(nn.Module):
    def __init__(self,  static_cont_feature_num = None ,                
                        static_cat_feature_num_list = None , 
                        history_cont_feature_num = None, 
                        history_cat_feature_num_list = None,
                        future_feature_list= None,

                        hidden_size = 64):
        super().__init__()

        self.static_cont = nn.Linear(static_cont_feature_num , hidden_size )
        self.static_cat = nn.ModuleList([
                nn.Embedding(i , hidden_size ) for i in static_cat_feature_num_list ])
        
        self.history_cont = nn.Linear(history_cont_feature_num , hidden_size )
        self.history_cat = nn.ModuleList([
                nn.Embedding(i , hidden_size ) for i in history_cat_feature_num_list ])
        
        self.future_feature = nn.ModuleList([
                nn.Embedding(i , hidden_size ) for i in future_feature_list ])


    def forward(self, 
                static_cont_input, 
                static_cat_input, 
                history_cont_input, 
                history_cat_input, 
                future_input):
        # Static continuous embedding
        static_cont_emb = self.static_cont(static_cont_input)

        # Static categorical embeddings
        static_cat_embs = [emb(static_cat_input[:, i]) 
                        for i, emb in enumerate(self.static_cat)]
        static_cat_emb = torch.stack(static_cat_embs, dim=-1)

        # History continuous embedding
        history_cont_emb = self.history_cont(history_cont_input)

        # History categorical embeddings
        history_cat_embs = [emb(history_cat_input[:, :, i]) 
                                for i, emb in enumerate(self.history_cat)]
        history_cat_emb = torch.stack(history_cat_embs, dim=-1)
        
        # Future features embedding
        future_emb =  [emb(future_input[:, :, i]) 
                                for i, emb in enumerate(self.future_feature)]
        future_emb = torch.stack(future_emb, dim=-1)
        
        static_cont_emb= static_cont_emb.unsqueeze(-1)
        history_cont_emb =history_cont_emb.unsqueeze(-1)

        # Concatenate embeddings
        static_input = torch.cat([static_cont_emb , static_cat_emb], dim=-1)
        history_input = torch.cat([history_cont_emb, history_cat_emb], dim=-1)


        return static_input, history_input, future_emb


class GRN(nn.Module):
    def __init__(self, input_size, 
                       hidden_size,
                       output_size = None,
                       context_size = None,
                       dropout_rate = 0.0):
        super().__init__()
        self.input_size =  input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, hidden_size)


        self.GLU = GLU(hidden_size, input_size if output_size else hidden_size)

        self.droput = nn.Dropout(dropout_rate)

        self.layernorm = nn.LayerNorm( input_size if output_size else hidden_size)

        if output_size:
            self.linear_out = nn.Linear(input_size, output_size)

        if context_size:
            self.linear_conext = nn.Linear(context_size, input_size, bias=False)
    
    def forward(self, x, context = None):

        linear1 = self.linear1(x)

        if context is not None:
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

class Gate_Add_Norm(nn.Module):
    def __init__(self, input_size , hidden_size):
        super().__init__()

        self.layernorm = nn.LayerNorm( hidden_size)
        self.GLU = GLU(input_size , hidden_size)
    
    def forward(self, x, residual):
        gated = self.GLU(x)

        output = self.layernorm(gated + residual)
        return output




class VariationSelection(nn.Module):
    def __init__(self, hidden_size , input_dim, context_size= None):
        super().__init__()
        self.group_GRN = GRN(hidden_size*input_dim, hidden_size, output_size= input_dim, context_size = context_size)
        self.individual_GRN = nn.ModuleList(
                    [GRN(hidden_size, hidden_size) for i in range(input_dim)])
    
    def forward(self, x, context = None):
        flat_x = torch.flatten(x, start_dim=-2)
        group_output = self.group_GRN(flat_x, context)
        group_weighted = F.softmax(group_output, dim=-1).unsqueeze(-1)
        individual_gru = [gru(x[...,i]) for i, gru in enumerate(self.individual_GRN)]
        individual_gru = torch.stack(individual_gru, dim=-1)
        if individual_gru.dim() == 3:
            output =  torch.einsum('aik, akj -> aij', individual_gru, group_weighted).squeeze(-1)
        
        else:
            output =  torch.einsum('abik, ackj -> abij', individual_gru, group_weighted).squeeze(-1)
        return output
    

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size,n_head,v_size, device):
        super().__init__()
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.attention_hidden_size = self.hidden_size//self.n_head
        self.div = self.attention_hidden_size**-0.5
        self.mask = torch.triu(torch.full((v_size, v_size), float('-inf')), 1).to(device)
        self.q_linear = nn.ModuleList(
                            [nn.Linear(hidden_size, self.attention_hidden_size)
                             for i in range(self.n_head)]

        )

        self.k_linear = nn.ModuleList(
                            [nn.Linear(hidden_size, self.attention_hidden_size)
                             for i in range(self.n_head)]
        )
        self.v_linear = nn.Linear(hidden_size, v_size)



    def forward(self, q, k, v):

        q_list = [q_linear(q)  for  q_linear in self.q_linear  ]
        k_list = [torch.transpose(k_linear(k),1,2)  for k_linear in self.k_linear  ]
        v = self.v_linear(v)
        attention_QK = [torch.matmul(q, k) for q,k in zip(q_list,k_list)]

        attention_QK = [ F.softmax(  torch.mul(attention, self.div) + self.mask, dim =-1 )       for i, attention in enumerate(attention_QK)   ]
        attention_QKV = [   torch.matmul(attention, v)    for  attention in  attention_QK      ]
        attention_QKV = torch.stack(attention_QKV, dim=-1)
        attention_QKV = torch.mean(attention_QKV, dim=-1) 
        return attention_QKV




import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.parameter import UninitializedParameter


class TFT_embeding(nn.Module):
    def __init__(self,  static_cont_feature_num = None ,                
                        static_cat_feature_num_list = None , 
                        history_cont_feature_num = None, 
                        history_cat_feature_num_list = None,
                        future_feature_list= None,

                        hidden_size = 64):
        super().__init__()

        self.static_cont = nn.Embedding(static_cont_feature_num , hidden_size )
        self.static_cat = nn.ModuleList([
                nn.Embedding(i , hidden_size ) for i in static_cat_feature_num_list ])
        
        self.history_cont = nn.Embedding(history_cont_feature_num , hidden_size )
        self.history_cat = nn.ModuleList([
                nn.Embedding(i , hidden_size ) for i in history_cat_feature_num_list ])
        
        self.future_feature = nn.Embedding(future_feature_list , hidden_size )


        
        

    def forward(self, static_cont_input, static_cat_input,history_cont_input, history_cat_input, future_input ):
        static_cont_input = self.static_cont(static_cont_input)
        static_cat_input = self.static_cat(static_cat_input)

        history_cont_input = self.history_cont(history_cont_input)
        history_cat_input = self.history_cat (history_cat_input)
        future_input= self.future_feature(future_input)

        static_input = torch.cat((static_cont_input,static_cat_input), dim=2)
        history_input = torch.cat((history_cont_input,history_cat_input), dim=2)

        

        return static_input , history_input, future_input






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


    
class CatEmbedding(nn.Module):
    def __init__(self, cat_size , hidden_size):
        super().__init__()
        self.embed = nn.Embedding(cat_size,hidden_size)

    def forward(self, x):
        embed = self.embed(x)
        return embed
    



    
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from .layer import *

class Model_Fusion(nn.Module):
    def __init__(self, tft_model, autoformer_model, n_head,hidden_size,device,output = 3):

        super().__init__()
        self.tft_model = tft_model
        self.autoformer_model = autoformer_model

        self.hook_outputs = {}
        self.tft_model.layernorm.register_forward_hook(self.save_hook_output("tft_model"))
        self.autoformer_model.decoder.layers[-1].dropout.register_forward_hook(self.save_hook_output("autoformer_model"))
        self.attentionfusion =  AttentionFusion(n_head, num_model = 2, seq_list =[15,7], hidden_size=hidden_size ,device=device)
        
        self.layernorm = nn.LayerNorm(hidden_size)

        self.glu = GLU(hidden_size,hidden_size)
        self.linear_output = nn.Linear(hidden_size,output)
    def save_hook_output(self, layer_name):
        def hook_fn(module, input, output):
            self.hook_outputs[layer_name] = output
        return hook_fn

    def forward(self,  static_cont_input, static_cat_input,history_cont_input, history_cat_input,future_input, tft_prediction, autoformer_feature_input, autoformer_prediction):

        tft_output = self.tft_model(static_cont_input, static_cat_input,history_cont_input, history_cat_input, future_input)  # Pass through Model A
        autoformer_output = self.autoformer_model(autoformer_feature_input, autoformer_prediction )  # Pass through Model B
        self.hook_outputs['autoformer_model'] = self.hook_outputs['autoformer_model'].permute(1, 0, 2)
        model_hidden_list =  list(self.hook_outputs.values())
        fusion = self.attentionfusion(model_hidden_list)
        glu = self.glu(fusion)
        output = self.layernorm(self.hook_outputs['tft_model']  + glu)
        output = self.linear_output(output)
        return output
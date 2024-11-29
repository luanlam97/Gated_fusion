from torch import nn
from .layer import *

class TFT(nn.Module):
    def __init__(self,  static_cont_feature_num = None ,                
                        static_cat_feature_num_list = None , 
                        history_cont_feature_num = None, 
                        history_cat_feature_num_list = None,
                        future_feature_list= None,
                        history_len = None,
                        future_len = None,
                        dropout = 0,
                        num_head = 1,

                        hidden_size = 64):
        super().__init__()
        self.future_len = future_len
        self.seq_len = future_len + history_len

        self.tft_embed =  TFT_embedding(static_cat_feature_num_list= static_cat_feature_num_list,
             static_cont_feature_num=static_cont_feature_num,
             history_cat_feature_num_list= history_cat_feature_num_list,
             history_cont_feature_num=history_cont_feature_num,
             future_feature_list=future_feature_list,
             hidden_size=hidden_size)

        static_variation_nums_dim = 1 + len(static_cat_feature_num_list)
        history_variation_nums_dim = 1 + len(history_cat_feature_num_list)
        future_variation_nums_dim =  len(future_feature_list)

        self.cs =  VariationSelection(hidden_size = hidden_size,input_dim = static_variation_nums_dim )
        self.ce =  VariationSelection(hidden_size = hidden_size,input_dim = static_variation_nums_dim )
        self.cc = VariationSelection(hidden_size = hidden_size,input_dim = static_variation_nums_dim )
        self.ch = VariationSelection(hidden_size = hidden_size,input_dim = static_variation_nums_dim )

        self.history_variation = VariationSelection(hidden_size = hidden_size,input_dim = history_variation_nums_dim, context_size= hidden_size)
        self.future_variation = VariationSelection(hidden_size = hidden_size,input_dim = future_variation_nums_dim, context_size= hidden_size)
        self.gate_add_norm_history = Gate_Add_Norm(hidden_size, hidden_size)
        self.gate_add_norm_future = Gate_Add_Norm(hidden_size, hidden_size)

        
        self.GRN = GRN(hidden_size, hidden_size, context_size = hidden_size)

        self.history_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True, batch_first=True)
        self.future_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True, batch_first=True)

        self.history_layernorm = nn.LayerNorm(hidden_size, eps = dropout )
        self.future_layernorm = nn.LayerNorm(hidden_size, eps = dropout )

        self.InterpAttention =  InterpretableMultiHeadAttention(hidden_size,num_head,self.seq_len)
        self.attention_layernorm = nn.LayerNorm(self.seq_len, eps = dropout )

        self.gate_add_norm_attention = Gate_Add_Norm(self.seq_len, hidden_size)
        self.attention_GRN = GRN(hidden_size, hidden_size)
        self.gate_add_norm_last = Gate_Add_Norm(future_len, hidden_size)

        self.GLU = GLU(hidden_size, hidden_size)

        self.layernorm = nn.LayerNorm(hidden_size)

        self.output_linear = nn.Linear(hidden_size, 3 )
    def forward(self, static_cont_input, static_cat_input,history_cont_input, history_cat_input, future_input):
        static_input, history_input, future_input = self.tft_embed (static_cont_input, 
                                                                static_cat_input,
                                                                history_cont_input, 
                                                                history_cat_input, 
                                                                future_input)
        
        cs  = self.cs(static_input)  
        ce  = self.ce(static_input)  
        cc  = self.cc(static_input) 
        ch  = self.ch(static_input) 


        history_variation = self.history_variation(history_input, cs)
        future_variation = self.future_variation(future_input, cs)

        hidden = (ch.unsqueeze(0) ,cc.unsqueeze(0))
 
        history_lstm, (ch,cc) =  self.history_lstm( history_variation ,hidden)
        hidden = (ch,cc)
        future_lstm, (ch,cc) =  self.future_lstm( future_variation ,hidden)

        history_norm = self.history_layernorm(history_lstm)
        future_norm = self.future_layernorm(future_lstm)

        gate_add_norm_history = self.gate_add_norm_history(x = history_norm, residual = history_variation)
        gate_add_norm_future = self.gate_add_norm_future(x = future_norm, residual = future_variation)
        combined = torch.cat([gate_add_norm_history,gate_add_norm_future], dim = 1)
        static_enriched = self.GRN(combined,ce)

        attention = self.InterpAttention( static_enriched,static_enriched,static_enriched       )
        attention_norm = self.attention_layernorm(attention)
        attention_gated =  self.gate_add_norm_attention(attention_norm,static_enriched )
        attention_GRN = self.attention_GRN(attention_gated)

        future = future_lstm[:, -self.future_len:, :] 
        future_attention = attention_GRN[:, -self.future_len:, :] 

        output =self.layernorm(self.GLU(future_attention) + future)

        output = self.output_linear(output)
        return output
# initialization code taken from https://huggingface.co/docs/transformers/en/model_doc/vit
# CITE: https://arxiv.org/pdf/2010.11929
# CITE: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
from transformers import ViTConfig, ViTModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(item):
  return item if isinstance(item,tuple) else (item,item)


# for use in transformer encoder
class FF(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, FF_dropout = 0.0):
    super().__init__()
    self.FFlayers = nn.Sequential(
      nn.Linear(embedding_dim, hidden_dim),
      nn.LayerNorm(embedding_dim),
      nn.ReLU(),
      nn.Dropout(FF_dropout),
      nn.Linear(hidden_dim, embedding_dim)
    )
    def forward(self, x):
      self.FFLayers(x)

# self attention model
class Attention(nn.Module):
    def __init__(self, embedding_dim, heads=8, dropout=0.0):
        super().__init__()
        
        self.pre_attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=heads, 
            dropout=dropout, 
            bias=True,
            add_bias_kv=False, 
            add_zero_attn=False,
            batch_first=False
        ) 

    def forward(self, x):
        x = self.pre_attention_norm(x)  # norm before attention!
        attn_output, _ = self.attention(x, x, x)  # self-attention (Q=K=V=x)
        return attn_output

# can't just use Transformer Encoder due to paper specifications
class TransformerEncoder_ViT(nn.Module):
  def __init__(self, embedding_dim, depth, heads, dim_heads, mlp_dim, dropout = 0.0):
      super().__init__()
      self.norm_final = nn.LayerNorm(embedding_dim)
      self.depth = depth

      # attention layer has norm and attention
      # FF has norm and MLP
      self.layers = nn.ModuleList([])

      for i in range(depth):
        self.layers.append(nn.ModuleList([
        Attention(embedding_dim),
        FF(embedding_dim, mlp_dim , FF_dropout = dropout)  # mlp dim, since this FF layer has Norm + MLP after attention
      ]))

  def forward(self, x):
    for layer in self.layers:
      if isinstance(layer, Attention):
      # have to add embedded patches before and attention layer together
        x = layer(x) + x
      # have to add ff layer (norm and mlp) to previous attention layer
      elif isinstance(layer, FF):
          x = layer(x) + x

      return self.norm_final(x) # this norm might be incorrect - check


class ViT_Model(nn.Module):
    def __init__(self, image_size, patch_size, embedding_dim, num_heads, num_classes, num_layers, mlp_dim, embedding_dropout = 0.0,  pool = 'cls'):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # FIXME: remove later
        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width

        # embedding
        # take input image, make into sequence of patch embeddings to feed into ViT
        self.make_into_patch_embedding = nn.Sequential(
          # takes care of reshaping the input image into patches, flattening each patch
          # for use in transformer 
          Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
          nn.LayerNorm(patch_dim),  # normalize features of each flattened vector of each patch 
          nn.Linear(patch_dim, embedding_dim),  # projects flattened patch vector into dim, which is model's dimensionality
          nn.LayerNorm(embedding_dim), # ensures output features are stabilized
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embedding_dim)) # positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1,1,embedding_dim)) # account for cls token
        self.dropout_embedding = nn.Dropout(embedding_dropout)

        # transformer
        self.transformer = TransformerEncoder_ViT(
            embedding_dim= embedding_dim,
            depth= num_layers,
            heads=num_heads,
            dim_heads = 256,
            mlp_dim=mlp_dim, # extra mlp layer within the encoder
        )
        self.pool = pool
        self.placeholder = nn.Identity()

        # outputs class - will take care of 1 or 0 isssue
        self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, image):

        # embedding first
        x = self.make_into_patch_embedding(image)
        batch_size, num_patches, _ = x.size()

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed[:, :(num_patches+1)] # n + 1 for extra patch for cls token / position embedding as well
        x = self.dropout_embedding(x)


        # transformer
        x = self.transformer(x)

        x = x[:, 0] # account for cls
        x = self.placeholder(x)

        return self.mlp_head(x)











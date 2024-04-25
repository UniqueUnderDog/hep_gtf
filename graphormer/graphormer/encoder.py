import torch.nn as nn
from typing import List
import torch
from gcn.layers import Transformer_Layer

class TransformerEncoder(nn.Module):
    def __init__(self, 
                 input_dim:int,
                 num_layers: int=8, 
                 hidden_dim: int=768, 
                 num_heads: int=8, 
                 ffn_hidden_dim: int=512, 
                 dropout: float = 0.1, 
                 attn_dropout: float = 0.1, 
                 temperature:int=1,
                 activation_fn: str = "GELU"):
        super().__init__()
        self.layers = nn.ModuleList([
            Transformer_Layer(input_dim=input_dim,
                              num_heads=num_heads,
                              hidden_dim=hidden_dim,
                              ffn_hidden_dim=ffn_hidden_dim,
                              dropout=dropout,
                              attn_dropout=attn_dropout,
                              temperature=temperature,
                              activation_fn=activation_fn)
            for _ in range(num_layers)
        ])

    def forward(self, x) -> torch.Tensor:
        output = x
        for layer in self.layers:
            output,_ = layer(output)
        return output
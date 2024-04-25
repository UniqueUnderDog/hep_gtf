import torch
import math
from torch.nn import init
import json
import torch.nn as nn
import torch.nn.functional as F


def get_activation_function(activation: str='PReLU') -> nn.Module:
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "Linear":
        return lambda x: x
    elif activation == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')



class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self, 
        num_atoms, 
        hidden_dim
    ):
        super(GraphNodeFeature, self).__init__()

        self.num_atoms = num_atoms

        # 1 for graph token
        self.atom_encoder = nn.Embedding(num_atoms, hidden_dim)
        self.linear_degree=nn.Linear(1,hidden_dim)

        self.graph_token = nn.Embedding(1, hidden_dim)

    def forward(self, batched_data):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )

        # node feature + graph token
        node_feature = self.atom_encoder(x)  # [ n_node, n_hidden]

        # if self.flag and perturb is not None:
        #     node_feature += perturb
        in_degree_feature=self.linear_degree(in_degree)
        out_degree_feature=self.linear_degree(out_degree)
        node_feature = (
            node_feature
            + in_degree_feature
            + out_degree_feature
        )

        return node_feature


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self,hidden_dim , ffn_hidden_dim, dropout=0.1,activation_fn="GELU"):
        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, hidden_dim)
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn_act_func = get_activation_function(activation_fn)

    def forward(self, x):
        residual=x
        x = self.dropout(self.fc2(self.act_dropout(self.ffn_act_func(self.fc1(x)))))
        x+=residual
        x = self.ffn_layer_norm(x)
        return x



class MultiheadAttention(nn.Module):
    """
    Compute 'Scaled Dot Product SelfAttention
    """
    def __init__(self,
                 input_dim,
                 num_heads,
                 hidden_dim,
                 dropout=0.1,
                 attn_dropout=0.1,
                 temperature = 1):
        super().__init__()
        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads  # number of heads
        self.temperature =temperature
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.a_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim,eps=1e-6)
        self.reset_parameters()
        print(self.q_proj.weight.dtype)
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.a_proj.weight)

    def forward(self, x):
        residual = x
        self.x=x

        query = self.q_proj(self.x)
        key = self.k_proj(self.x)
        value = self.v_proj(self.x)

        query = query.view(-1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view( -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(-1, self.num_heads, self.d_k).transpose(1, 2)

        #ScaledDotProductAttention

        scores = torch.matmul(query/self.temperature, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))


        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        #ScaledDotProductAttention

        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).contiguous().view( -1, self.num_heads * self.d_k)
        out = self.dropout(self.a_proj(out))
        out += residual
        out = self.layer_norm(out)

        return out, attn


class Transformer_Layer(nn.Module):
    def __init__(self,
                 input_dim,
                 num_heads,
                 hidden_dim,
                 ffn_hidden_dim,
                 dropout=0.1,
                 attn_dropout=0.1,
                 temperature = 1,
                 activation_fn='GELU'):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.attention = MultiheadAttention(input_dim,
                                            num_heads,
                                            hidden_dim, 
                                            dropout, 
                                            attn_dropout,
                                            temperature)
        self.ffn_layer = PositionwiseFeedForward(hidden_dim,ffn_hidden_dim,activation_fn=activation_fn)


    def forward(self, x):
        x, attn = self.attention(x)
        x = self.ffn_layer(x)

        return x, attn
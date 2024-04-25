import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hgcn import hyp_layer
import utils.math_utils as pmath
from manifold import poincare




class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output


class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(
                 self, 
                 c=1,
                 num_layers=12, 
                 dropout=0.1,
                 act="relu",
                 use_bias=True,
                 in_dim=1,
                 device=torch.device('cuda'),
                 cuda=0,
                 hidden_dim=32):
        super(HGCN, self).__init__(c)
        self.in_dim=in_dim
        self.num_layers=num_layers

        self.dropout=dropout,
        self.act=act,
        self.act=self.act[0]
        self.use_bias=use_bias
        self.device=device
        self.cuda=cuda
        self.hidden_dim=hidden_dim
        self.manifold=poincare.PoincareBall()
        assert num_layers > 1
        dims, acts, self.curvatures = hyp_layer.get_dim_act_curv(
                                                                feat_dim=self.in_dim,
                                                                act=self.act,
                                                                num_layers=self.hidden_dim,
                                                                dim=self.hidden_dim,
                                                                device=self.device,
                                                                cuda=self.cuda,
                                                                c=c,
                                                                )
        self.curvatures.append(self.c) 
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layer.HyperbolicGraphConvolution(
                                                        in_dim,
                                                        out_dim, 
                                                        c_in, 
                                                        c_out, 
                                                        dropout, 
                                                        act=self.act,
                                                        use_bias=self.use_bias 
                                                        )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)




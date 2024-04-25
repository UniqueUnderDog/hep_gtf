import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from manifold import poincare
from utils.graph_utils import GraphUtils
from torch_geometric.nn import GCNConv

class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.in_features = in_features

    def forward (self, x, adj):
        n = x.size(0)
        # n x 1 x d
        x_left = torch.unsqueeze(x, 1)
        x_left = x_left.expand(-1, n, -1)
        # 1 x n x d
        x_right = torch.unsqueeze(x, 0)
        x_right = x_right.expand(n, -1, -1)

        x_cat = torch.cat((x_left, x_right), dim=2)
        att_adj = self.linear(x_cat).squeeze()
        att_adj = F.sigmoid(att_adj)
        att_adj = torch.mul(adj.to_dense(), att_adj)
        return att_adj

def get_dim_act_curv(
                     feat_dim:int,
                     act:str,
                     num_layers:int,
                     dim:int,
                     device,
                     cuda:int,
                     c:float):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not act:
        act = lambda x: x
    else:
        act = getattr(F, act)
    acts = [act] * (num_layers - 1)
    dims = [feat_dim] + ([dim] * (num_layers - 1))
    n_curvatures = num_layers - 1
    if c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([c]) for _ in range(n_curvatures)]
        if not cuda == -1:
            curvatures = [curv.to(device) for curv in curvatures]
    return dims, acts, curvatures


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(
                 self,  
                 in_features, 
                 out_features, 
                 c_in, 
                 c_out, 
                 dropout, 
                 act, 
                 use_bias
                 ):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(in_features, 
                                out_features, 
                                c_in, 
                                dropout, 
                                use_bias)
        self.a1gg = HypAgg(c_in, 
                          out_features, 
                          dropout)
        self.a2gg = HypAgg(c_out,
                           c_out,
                           dropout)
        self.agg_pyg=HypAgg_pyg(c_in,
                                out_features,
                                dropout)
        self.hyp_act = HypAct(c_in, 
                              c_out, 
                              act)

    def forward(self, input):
        
        x= input[0]
        adj=input[1]
        x=x.to(torch.device('cuda:0'))
        adj=adj.to(torch.device('cuda:0'))
        h = self.linear.forward(x)
        h = self.a1gg.forward(h, adj)
        h=self.a2gg.forward(h,adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output





class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(
                 self,  
                 in_features, 
                 out_features, 
                 c=0.1, 
                 dropout=0.1, 
                 use_bias=False):
        super(HypLinear, self).__init__()
        self.manifold = poincare.PoincareBall()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.normal_(self.bias, mean=0.0, std=0.01)

    def forward(self, x):
        x = x.to(torch.device('cuda:0'))
        
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        drop_weight=drop_weight.to(torch.device('cuda:0'))
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, 
                 c,
                 in_features, 
                 dropout):
        super(HypAgg, self).__init__()
        self.manifold = poincare.PoincareBall()
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
    
    def forward(self, x, adj):
        
        
        x_tangent = self.manifold.logmap0(x, c=self.c)
        with torch.no_grad():
            adj= GraphUtils.convert_adj_to_matrix(adj,x.shape[0])
        
        x_tangent = x_tangent.to(torch.device('cuda:0'))
        adj = adj.to(torch.device('cuda:0'))
        support_t = torch.spmm(adj.to(torch.float32), x_tangent)
        adj.zero_()
        del adj
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)
    
    



class HypAct(Module):
    """
    Hyperbolic activation layer.
    """
    def __init__(self,  c_in, c_out, act="relu"):
        super(HypAct, self).__init__()
        self.manifold = poincare.PoincareBall()
        self.c_in = c_in
        self.c_out = c_out
        if act.lower() == "relu":
            self.act = nn.ReLU()
        elif act.lower() == "sigmoid":
            self.act = nn.Sigmoid()
        elif act.lower() == "tanh":
            self.act = nn.Tanh()

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(p=x, c=self.c_in))
        xt = self.manifold.proj_tan0(u=xt, c=self.c_out)
        return self.manifold.proj(x=self.manifold.expmap0(u=xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class HypAgg_pyg(Module):
    """
    Hyperbolic aggregation layer with pyg.
    """

    def __init__(self, 
                 c,
                 in_features, 
                 dropout):
        super(HypAgg_pyg, self).__init__()
        self.manifold = poincare.PoincareBall()
        self.c = c
        
        self.in_features = in_features
        self.conv1 = GCNConv(in_features, in_features)
        self.dropout = dropout
    
    def forward(self, x, adj):
        
        
        x_tangent = self.manifold.logmap0(x, c=self.c)
        x_tangent = x_tangent.to(torch.device('cuda:0'))
        adj = adj.to(torch.device('cuda:0'))
        
        support_t = self.conv1(x, adj)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)
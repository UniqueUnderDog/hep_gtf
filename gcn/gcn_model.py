import torch
import torch.nn as nn
from gcn.gcn_encoder import GCN
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp

class GCNModel(nn.Module):
    def __init__(self,
                 task="gr",
                 task_dim=1,
                 in_dim=1,
                 out_dim=32
                 ):
        super(GCNModel, self).__init__()
        # 实例化HGCN类
        self.task=task
        self.task_dim=task_dim
        self.device=torch.device('cuda:0')
        self.gcn = GCN( 
                 in_dim=in_dim,
                 out_dim=out_dim,
                 dropout=0.1
        )
        self.classifier = nn.Linear(2*out_dim, 1) 
        # 添加一个线性层来映射到输出类别数
        

    def forward(self, data):
        # 通过HGCN层进行编码
        x, adj = data.x.to(self.device), data.edge_index.to(self.device)
        x = self.gcn.forward(x, adj)
        # 应用分类器
        if self.task == "gr":
            # 对编码后的表示执行全局平均池化，并调整形状为 (batch_size, 1)
            x_gap_pooled = gap(x, batch=data.batch)
            x_gmp_pooled=gmp(x,batch=data.batch)
            x_pooled = torch.cat((x_gap_pooled, x_gmp_pooled), dim=1)
            x_out= self.classifier(x_pooled)
        else:
            # 如果不是 "gr" 任务，直接返回编码结果
            x_out = x
        
        return x_out

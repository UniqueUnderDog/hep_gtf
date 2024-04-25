import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC
import torch_geometric.utils as utils



class HyperDataset(object):
    def __init__(self, 
                 dataset):
        # 将位置编码列表转换为一个大张量
        self.dataset = []  # 创建一个空列表来存放带有全局节点编号的数据
        self.n_features = dataset[0].x.shape[-1]
        self.global_node_counter = 0
        self.graph_counter = 0
        
        for data in dataset:
            graph_id = len(self.dataset)  # 获取当前图的ID（即dataset中的索引）
            global_node_ids = torch.arange(self.global_node_counter,
                                           self.global_node_counter + data.num_nodes,
                                           dtype=torch.long)
            
            # 组合全局节点ID和所属图ID，形成双元素元组
            data.global_node_ids = torch.stack((graph_id * torch.ones_like(global_node_ids), global_node_ids), dim=-1)  # 调整列顺序

            self.global_node_counter += data.num_nodes
            self.dataset.append(data) 
        
    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        data = self.dataset[index]
        start_id = sum([d.num_nodes for d in self.dataset[:index]])  # 计算累计节点数
        end_id = start_id + data.num_nodes

        

        
        return data
    
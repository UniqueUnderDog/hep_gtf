import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC
import torch_geometric.utils as utils

position_encode_path = "./data/ZINC/subset/processed/last_epoch_position_encodes.pt"
position_encode_data = torch.load(position_encode_path)

class GraphDataset(object):
    def __init__(self, 
                 dataset,
                 position_encode_data_list,  # 更改变量名，表明这是一个列表
                 degree=True,  ):
        # 将位置编码列表转换为一个大张量
        self.position_encode_tensor = torch.cat(position_encode_data_list, dim=0)
        self.dataset = []  # 创建一个空列表来存放带有全局节点编号的数据
        self.n_features = dataset[0].x.shape[-1]
        self.in_degree = degree  # 新增参数，控制是否计算入度
        self.out_degree = degree
        self.global_node_counter = 0
        
        for data in dataset:
            data.global_node_ids = torch.arange(self.global_node_counter,
                                               self.global_node_counter + data.num_nodes,
                                               dtype=torch.long)
            self.global_node_counter += data.num_nodes
            self.dataset.append(data) 
        self.compute_degree()
        max_degree = 0
        for data in dataset:
            max_degree = max(max_degree, data.edge_index.size(1))
        self.max_degree = max_degree
        
    
    def compute_degree(self):
        if not self.in_degree and not self.out_degree:
            return

        self.in_degree_list = []
        self.out_degree_list = []

        for g in self.dataset:
            in_degrees = utils.degree(g.edge_index[0], g.num_nodes)  # 计算入度
            out_degrees = utils.degree(g.edge_index[1], g.num_nodes)  # 计算出度

        # 如果只需要入度或出度，可以在这里进行筛选
            if self.in_degree:
                self.in_degree_list.append(in_degrees)
            if self.out_degree:
                self.out_degree_list.append(out_degrees)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        start_id = sum([d.num_nodes for d in self.dataset[:index]])  # 计算累计节点数
        end_id = start_id + data.num_nodes

        data.position_encode = self.position_encode_tensor[start_id:end_id].view(data.num_nodes, -1)
        
        if data.x.dim() > 1 and data.x.size(1) == 1:
            data.x = data.x.squeeze(1)  # 移除第二维

        data.degree = None
        if self.in_degree:
            data.in_degree = self.in_degree_list[index].unsqueeze(-1)
        if self.out_degree:
            data.out_degree = self.out_degree_list[index].unsqueeze(-1)
        
        return data
    

import torch
import torch.nn as nn
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from dataset.Graphdataset import GraphDataset
from hgcn.hgcn_dataset import HyperDataset
from utils.graph_utils import GraphUtils



dataset = HyperDataset(ZINC("./data/ZINC", subset=True, split='train'))
print(dataset[0].global_node_ids)
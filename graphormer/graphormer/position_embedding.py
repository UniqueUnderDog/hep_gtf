import torch

class PositionEmbedding(torch.nn.Module):
    def __init__(self, position_embedding_dim):
        super(PositionEmbedding, self).__init__()
        self.position_embedding_dim = position_embedding_dim

    def forward(self, data):
        # 获取节点的位置编码、入度和出度
        position_encodes = data.position_encode
        in_degree = data.in_degree
        out_degree = data.out_degree

        # 确保度信息为 torch.long 类型
        in_degree = in_degree.to(torch.long)
        out_degree = out_degree.to(torch.long)

        # 计算节点的位置嵌入
        position_embedding = position_encodes / torch.norm(position_encodes, dim=-1, keepdim=True)
        position_embedding *= (in_degree + out_degree).unsqueeze(-1)  # 正比于入度和出度之和

        return position_embedding
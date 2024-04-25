import torch

class GraphUtils:
    @staticmethod
    def convert_adj_to_matrix(adj, num_nodes, fill_value=1):
        """
        Converts a PyG-style adjacency tensor [2, num_edges] to a dense [num_nodes, num_nodes] matrix.

        Args:
        adj (torch.Tensor): PyG-style adjacency tensor of shape [2, num_edges].
        num_nodes (int): The number of nodes in the graph.
        fill_value (float, optional): Value to fill in the adjacency matrix. Defaults to 1.

        Returns:
        torch.Tensor: Dense adjacency matrix of shape [num_nodes, num_nodes].
        """
        adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=adj.dtype)
        
        src, dst = adj.unbind(0)  # Unpack source and destination node indices
        adj_matrix[src, dst] = fill_value  # 使用fill_value填充邻接矩阵
        
        return adj_matrix
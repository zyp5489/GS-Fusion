import torch
import torch.nn as nn
from torch_geometric.nn import DenseGCNConv
from torch_geometric.utils import dropout_adj


class DenseGCNBlock(nn.Module):
    """密集图卷积块"""
    def __init__(self, layer_dims, dropout_rate=0.0, relu_indices=None, dropout_indices=None):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.convs.append(DenseGCNConv(layer_dims[i], layer_dims[i+1]))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_indices = relu_indices or []
        self.dropout_indices = dropout_indices or []

    def forward(self, x, adj):
        h = x
        outputs = []
        for idx, conv in enumerate(self.convs):
            h = conv(h, adj, add_loop=False)
            if idx in self.relu_indices:
                h = self.relu(h)
            if idx in self.dropout_indices:
                h = self.dropout(h)
            outputs.append(torch.squeeze(h, dim=0))
        return outputs


class AffinityGraphEncoder(nn.Module):
    """在药物-靶标亲和力矩阵上运行的GCN编码器"""
    def __init__(self, layer_dims, edge_dropout_rate=0.0):
        super().__init__()
        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layer_dims) - 1
        self.gcn_block = DenseGCNBlock(
            layer_dims,
            dropout_rate=0.1,
            relu_indices=list(range(self.num_layers)),
            dropout_indices=list(range(self.num_layers))
        )

    def forward(self, graph):
        x, adj = graph.x, graph.adj
        num_d, num_t = graph.num_drug, graph.num_target

        indices = torch.where(adj != 0)
        edge_index = torch.stack([indices[0], indices[1]], dim=0)
        edge_index_drop, edge_weight_drop = dropout_adj(
            edge_index, edge_attr=adj[indices],
            p=self.edge_dropout_rate, force_undirected=True,
            num_nodes=num_d + num_t, training=self.training
        )
        adj_drop = torch.zeros_like(adj)
        adj_drop[edge_index_drop[0], edge_index_drop[1]] = edge_weight_drop

        embeddings = self.gcn_block(x, adj_drop)
        return embeddings
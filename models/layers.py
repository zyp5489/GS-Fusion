import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout, LayerNorm as TorchLayerNorm
from torch_geometric.nn import GATConv, GCNConv, SAGPooling, global_add_pool
from torch_geometric.nn import LayerNorm as GeoLayerNorm, BatchNorm
from torch_geometric.utils import to_dense_batch


class ResidualGraphAttentionLayer(nn.Module):
    """
    带残差连接和图注意力卷积的模块，用于分子图或蛋白质图处理
    """
    def __init__(self, n_heads, in_features, head_out_feats,
                 use_conv1d=False, window_size=5,
                 dropout_att=0.4, dropout_out=0.1,
                 use_layer_norm=True, edge_feat_dim=0, extra_dim=None):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.head_out_feats = head_out_feats
        self.hidden_dim = n_heads * head_out_feats
        self.extra_dim = extra_dim
        self.use_conv1d = use_conv1d

        self.residual_proj = nn.Linear(in_features, self.hidden_dim)
        if self.use_conv1d:
            self.conv1d = nn.Conv1d(in_features, self.hidden_dim, kernel_size=window_size, padding=window_size//2)
            gat_in_dim = self.hidden_dim
        else:
            gat_in_dim = in_features

        self.gat = GATConv(gat_in_dim, head_out_feats, n_heads, dropout=dropout_att)
        self.pool = SAGPooling(self.hidden_dim, min_score=-1)
        self.norm = GeoLayerNorm(self.hidden_dim) if use_layer_norm else BatchNorm(self.hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_out)

    def forward(self, graph_data):
        residual = self.residual_proj(graph_data.x)

        if self.use_conv1d:
            graph_data.x = self.conv1d(graph_data.x.t().unsqueeze(0)).squeeze(0).t()

        graph_data.x, att_weights = self.gat(graph_data.x, graph_data.diff_edge_index,
                                              return_attention_weights=True)

        pooled_x, pooled_edge_index, pooled_edge_attr, pooled_batch, perm, scores = \
            self.pool(graph_data.x, graph_data.diff_edge_index, batch=graph_data.batch)
        global_emb = global_add_pool(pooled_x, pooled_batch)

        graph_data.x = graph_data.x + residual
        graph_data.x = self.dropout(self.activation(self.norm(graph_data.x)))

        return graph_data, global_emb, att_weights, scores


class BidirectionalCrossAttention(nn.Module):
    """双向交叉亲和变换（基于AFT），用于药物和蛋白质特征交互"""
    def __init__(self, config, drug_dim, protein_dim):
        super().__init__()
        self.config = config
        self.pos_bias = nn.Parameter(
            torch.zeros(config['max_seq_len'], config['max_seq_len']),
            requires_grad=config.get('cross_pos_bias_switch', False)
        )
        self.sigmoid = nn.Sigmoid()

        self.query_drug = nn.Linear(drug_dim, config['n_embd'])
        self.key_prot = nn.Linear(protein_dim, config['n_embd'])
        self.value_prot = nn.Linear(protein_dim, config['n_embd'])

        self.query_prot = nn.Linear(protein_dim, config['n_embd'])
        self.key_drug = nn.Linear(drug_dim, config['n_embd'])
        self.value_drug = nn.Linear(drug_dim, config['n_embd'])

        self.proj_drug = nn.Linear(config['n_embd'], config['n_embd'])
        self.proj_prot = nn.Linear(config['n_embd'], config['n_embd'])

    def forward(self, drug_emb, drug_mask, prot_emb, prot_mask):
        B_d, T_d, C_d = drug_emb.size()
        B_p, T_p, C_p = prot_emb.size()
        assert B_d == B_p

        drug_emb = drug_emb.permute(1, 0, 2)
        prot_emb = prot_emb.permute(1, 0, 2)

        if self.config.get('masking', False):
            comb_mask = drug_mask.unsqueeze(2) * prot_mask.unsqueeze(1)
            comb_mask = comb_mask.permute(1, 2, 0)

        # Drug side
        q_d = self.query_drug(drug_emb)
        k_p = self.key_prot(prot_emb)
        v_p = self.value_prot(prot_emb)

        pos_bias = self.pos_bias[:T_d, :T_p].unsqueeze(-1)
        if self.config.get('masking', False):
            pos_bias = pos_bias.masked_fill(comb_mask == 0, -1e9)

        max_k_p = k_p.max(dim=0, keepdim=True)[0]
        max_pos = pos_bias.max(dim=1, keepdim=True)[0]

        exp_k = torch.exp(k_p - max_k_p)
        exp_pos = torch.exp(pos_bias - max_pos)

        num = torch.einsum('ijb,jbd->ibd', exp_pos, exp_k * v_p)
        den = torch.einsum('ijb,jbd->ibd', exp_pos, exp_k)
        y_d = self.sigmoid(q_d) * num / den
        y_d = self.proj_drug(y_d).permute(1, 0, 2)

        # Protein side
        q_p = self.query_prot(prot_emb)
        k_d = self.key_drug(drug_emb)
        v_d = self.value_drug(drug_emb)

        pos_bias = self.pos_bias[:T_d, :T_p].transpose(0, 1).unsqueeze(-1)
        if self.config.get('masking', False):
            pos_bias = pos_bias.masked_fill(comb_mask.permute(1, 0, 2) == 0, -1e9)

        max_k_d = k_d.max(dim=0, keepdim=True)[0]
        max_pos = pos_bias.max(dim=1, keepdim=True)[0]

        exp_k = torch.exp(k_d - max_k_d)
        exp_pos = torch.exp(pos_bias - max_pos)

        num = torch.einsum('ijb,jbd->ibd', exp_pos, exp_k * v_d)
        den = torch.einsum('ijb,jbd->ibd', exp_pos, exp_k)
        y_p = self.sigmoid(q_p) * num / den
        y_p = self.proj_prot(y_p).permute(1, 0, 2)

        if self.config.get('masking', False):
            y_d = y_d.masked_fill(drug_mask.unsqueeze(-1) == 0, -1e9)
            y_p = y_p.masked_fill(prot_mask.unsqueeze(-1) == 0, -1e9)

        y_d, _ = torch.max(y_d, dim=1)
        y_p, _ = torch.max(y_p, dim=1)

        return y_d, y_p


class LayerNormDropout(nn.Module):
    """层归一化 + Dropout"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = TorchLayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.norm(x))


class MLPBlock(nn.Module):
    """多层感知机块，可指定哪些层应用ReLU和Dropout"""
    def __init__(self, layer_dims, dropout_rate=0.0, relu_indices=None, dropout_indices=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(Linear(layer_dims[i], layer_dims[i+1]))
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_indices = relu_indices or []
        self.dropout_indices = dropout_indices or []

    def forward(self, x):
        outputs = [x]
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.relu_indices:
                x = self.relu(x)
            if idx in self.dropout_indices:
                x = self.dropout(x)
            outputs.append(x)
        return outputs


class LightGraphConvEncoder(nn.Module):
    """轻量级GCN编码器，用于扩散视图特征提取"""
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid_dim))
        self.bns.append(nn.BatchNorm1d(hid_dim))
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hid_dim, hid_dim))
            self.bns.append(nn.BatchNorm1d(hid_dim))
        if n_layers >= 2:
            self.convs.append(GCNConv(hid_dim, out_dim))
            self.bns.append(nn.BatchNorm1d(out_dim))
        self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        hs = []
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight)
            h = self.bns[i](h)
            h = self.activation(h)
            hs.append(h)
        return hs


class DynamicGateFusion(nn.Module):
    """动态门控融合多分支特征"""
    def __init__(self, branch_dim, num_branches):
        super().__init__()
        self.num_branches = num_branches
        self.gate_net = nn.Sequential(
            nn.Linear(branch_dim, branch_dim // 2),
            nn.ReLU(),
            nn.Linear(branch_dim // 2, num_branches)
        )

    def forward(self, *branches):
        B = branches[0].shape[0]
        gate_logits = self.gate_net(branches[0])
        gates = torch.softmax(gate_logits, dim=-1)

        weighted = []
        for i, feat in enumerate(branches):
            w = gates[:, i:i+1]
            weighted.append(w * feat)
        return torch.cat(weighted, dim=-1)


def masked_pooling(x, mask, dim=1):
    """带掩码的平均池化"""
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.to(dtype=torch.float32).unsqueeze(-1)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-9)


def compute_cosine_similarity(x, y=None):
    """计算两两余弦相似度"""
    x_norm = F.normalize(x, dim=-1)
    if y is not None:
        y_norm = F.normalize(y, dim=-1)
        return torch.mm(x_norm, y_norm.T)
    return torch.mm(x_norm, x_norm.T)
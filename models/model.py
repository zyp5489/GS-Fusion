import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

# 导入自定义层
from layers import (
    ResidualGraphAttentionLayer,
    BidirectionalCrossAttention,
    LayerNormDropout,
    MLPBlock,
    LightGraphConvEncoder,
    DynamicGateFusion,
    masked_pooling
)
from affinity_encoder import AffinityGraphEncoder


class AffinityNet(nn.Module):
    """
    综合药物分子图、蛋白质图、指纹和亲和力二部图的端到端亲和力预测模型
    """
    def __init__(self, **params):
        super().__init__()
        self.params = params
        self.num_drugs = params.get("num_drugs", 68)   # 数据集中药物数量
        self.num_targets = params.get("num_targets", 442)  # 数据集中靶标数量

        # ---------- 化学图模块 ----------
        self.chem_blocks = nn.ModuleList()
        chem_in = params["chem_in_features"]
        for i, (head_out, n_heads) in enumerate(
                zip(params["chem_heads_out_feat_params"],
                    params["chem_blocks_params"])):
            block = ResidualGraphAttentionLayer(
                n_heads, chem_in, head_out,
                use_conv1d=False,
                dropout_att=params["dropout_1"],
                dropout_out=params["dropout_2"],
                use_layer_norm=True
            )
            self.chem_blocks.append(block)
            chem_in = head_out * n_heads
        self.chem_final_dim = chem_in

        # ---------- 蛋白质图模块 ----------
        self.prot_blocks = nn.ModuleList()
        prot_in = params["prot_in_features"]
        for i, (head_out, n_heads, win_size) in enumerate(
                zip(params["prot_heads_out_feat_params"],
                    params["prot_blocks_params"],
                    params["prot_windows_params"])):
            block = ResidualGraphAttentionLayer(
                n_heads, prot_in, head_out,
                use_conv1d=False,
                window_size=win_size,
                dropout_att=params["dropout_1"],
                dropout_out=params["dropout_2"],
                use_layer_norm=True
            )
            self.prot_blocks.append(block)
            prot_in = head_out * n_heads
        self.prot_final_dim = prot_in

        # 配置标志
        self.use_diff_raw = params.get("using_diff_raw", False)
        self.use_lstm = params.get("using_lstm", False)
        self.use_contrastive = params.get("using_constractive", False)

        # 特征投影
        self.fp_proj = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.1)
        )
        self.chem_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.1)
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(2560, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.1)
        )

        # 交叉注意力
        self.cross_att1 = BidirectionalCrossAttention(params, 256, 256)
        self.cross_att2 = BidirectionalCrossAttention(params, 256, 256)

        # 初始归一化
        self.chem_norm = nn.LayerNorm(params["chem_in_features"])
        self.prot_norm = nn.LayerNorm(params["prot_in_features"])

        # 可选的扩散编码器
        if self.use_diff_raw:
            self.prot_diff_enc = LightGraphConvEncoder(in_dim=256, hid_dim=256, out_dim=256, n_layers=2)
            self.chem_diff_enc = LightGraphConvEncoder(in_dim=256, hid_dim=256, out_dim=256, n_layers=2)

        # LSTM
        if self.use_lstm:
            self.chem_lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, dropout=0.0)
            self.prot_lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, dropout=0.0)

        # 亲和力图编码器
        affinity_dim = self.num_drugs + self.num_targets + 2
        self.affinity_gcn = AffinityGraphEncoder([affinity_dim, 512, 256], edge_dropout_rate=0.2)
        self.affinity_drug_proj = MLPBlock([256], 0.1, relu_indices=[0], dropout_indices=[0])
        self.affinity_target_proj = MLPBlock([256], 0.1, relu_indices=[0], dropout_indices=[0])

        # 动态门控融合（共11个分支）
        self.fusion_gate = DynamicGateFusion(branch_dim=256, num_branches=11)

        # 最终预测MLP
        fusion_dim = 256 * 11
        self.predictor = nn.Sequential(
            LayerNormDropout(fusion_dim, 0.1),
            nn.Linear(fusion_dim, 1024),
            nn.ELU(),
            LayerNormDropout(1024, 0.1),
            nn.Linear(1024, 512),
            nn.ELU(),
            LayerNormDropout(512, 0.1),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, chem_fp, chem_graph, prot_graph,
                batch_drug_ids=None, batch_target_ids=None,
                full_affinity_train=None, drug_pos=None, target_pos=None):
        # 初始归一化
        chem_graph.x = self.chem_norm(chem_graph.x)
        prot_graph.x = self.prot_norm(prot_graph.x)

        # 投影额外特征
        chem_graph.x2 = self.chem_proj(chem_graph.x2)
        prot_graph.x2 = self.prot_proj(prot_graph.x2)

        fp_repr = self.fp_proj(chem_fp)

        # ---------- 化学图处理 ----------
        chem_reprs = []
        for block in self.chem_blocks:
            chem_graph, r_chem, _, _ = block(chem_graph)
            chem_reprs.append(r_chem)
        chem_reprs = torch.stack(chem_reprs, dim=1)

        if self.use_diff_raw:
            chem_diff_hs = self.chem_diff_enc(
                chem_graph.x2,
                chem_graph.diff_edge_index,
                chem_graph.diff_edge_weight
            )
            chem_diff_repr = chem_diff_hs[-1]
        else:
            chem_diff_repr = chem_graph.x2

        chem_dense, chem_mask = to_dense_batch(chem_diff_repr, chem_graph.batch)

        if self.use_lstm:
            chem_lstm_out, _ = self.chem_lstm(chem_dense)
            chem_diff_pool = masked_pooling(chem_lstm_out, mask=chem_mask, dim=1)
        else:
            chem_diff_pool = masked_pooling(chem_dense, mask=chem_mask, dim=1)

        # ---------- 蛋白质图处理 ----------
        prot_reprs = []
        for block in self.prot_blocks:
            prot_graph, r_prot, _, _ = block(prot_graph)
            prot_reprs.append(r_prot)
        prot_reprs = torch.stack(prot_reprs, dim=1)

        if self.use_diff_raw:
            prot_diff_hs = self.prot_diff_enc(
                prot_graph.x2,
                prot_graph.diff_edge_index,
                prot_graph.diff_edge_weight
            )
            prot_diff_repr = prot_diff_hs[-1]
        else:
            prot_diff_repr = prot_graph.x2

        prot_dense, prot_mask = to_dense_batch(prot_diff_repr, prot_graph.batch)

        if self.use_lstm:
            prot_lstm_out, _ = self.prot_lstm(prot_dense)
            prot_diff_pool = masked_pooling(prot_lstm_out, mask=prot_mask, dim=1)
        else:
            prot_diff_pool = masked_pooling(prot_dense, mask=prot_mask, dim=1)

        # ---------- 交叉注意力 ----------
        chem_fused1, prot_fused1 = self.cross_att1(
            chem_dense, chem_mask,
            prot_dense, prot_mask
        )

        device = batch_drug_ids.device
        chem_fused2, prot_fused2 = self.cross_att2(
            chem_reprs,
            torch.ones(chem_reprs.shape[0], chem_reprs.shape[1], device=device),
            prot_reprs,
            torch.ones(prot_reprs.shape[0], prot_reprs.shape[1], device=device)
        )

        # ---------- 亲和力图特征 ----------
        aff_embeddings = self.affinity_gcn(full_affinity_train)[-1]
        aff_drug_emb = aff_embeddings[:self.num_drugs]
        aff_target_emb = aff_embeddings[self.num_drugs:]
        aff_drug_f = self.affinity_drug_proj(aff_drug_emb)[-1]
        aff_target_f = self.affinity_target_proj(aff_target_emb)[-1]

        aff_drug_batch = aff_drug_f[batch_drug_ids]
        aff_target_batch = aff_target_f[batch_target_ids]

        # 收集所有分支特征
        branches = [
            aff_drug_batch,
            aff_target_batch,
            prot_reprs[:, -1, :],
            chem_reprs[:, -1, :],
            chem_diff_pool,
            prot_diff_pool,
            chem_fused1,
            prot_fused1,
            chem_fused2,
            prot_fused2,
            fp_repr
        ]

        combined = self.fusion_gate(*branches)
        pred = self.predictor(combined).squeeze(-1)

        # 此处 contrast_loss 作为占位，可替换为实际对比损失实现
        return pred, 0.0
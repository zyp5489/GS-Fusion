import torch.nn as nn

params = {
    "num_drugs": 68,                # 数据集中药物数量
    "num_targets": 442,             # 数据集中靶标数量
    "chem_in_features": 32,
    "prot_in_features": 41,
    "chem_heads_out_feat_params": [32, 32, 32, 32, 32, 32],
    "chem_blocks_params": [8, 8, 8, 8, 8, 8],
    "prot_heads_out_feat_params": [32, 32, 32, 32],
    "prot_blocks_params": [8, 8, 8, 8],
    "prot_windows_params": [7, 7, 7, 7],
    "dropout_1": 0.4,
    "dropout_2": 0.1,
    "batch_size": 256,
    "lr": 1e-4,
    "criterion": nn.MSELoss(),
    "max_seq_len": 1200,
    "n_embd": 256,
    "cross_pos_bias_switch": False,
    "masking": False,
    "using_diff_raw": False,
    "using_lstm": True,
    "using_constractive": False,
    "contrast_weight": 1.0,
}
import time
import pickle
import torch as t
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gzip
import time
from torch_geometric.data import Data, Batch
from torch_geometric import data as DATA
import torch
import pytorch_lightning as pl
import json
import scipy.sparse as sp
from torch_geometric.utils import to_dense_adj


def minMaxNormalize(Y, Y_min=None, Y_max=None):
    if Y_min is None:
        Y_min = np.min(Y)
    if Y_max is None:
        Y_max = np.max(Y)
    normalize_Y = (Y - Y_min) / (Y_max - Y_min)
    return normalize_Y


class DiffusionMatrix:
    @staticmethod
    def compute_ppr(adj, alpha=0.2):
        """计算 Personalized PageRank 扩散矩阵"""
        deg = adj.sum(dim=1, keepdim=True)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_adj = deg_inv_sqrt * adj * deg_inv_sqrt.T
        n = adj.shape[0]
        I = torch.eye(n, device=adj.device)
        S = alpha * torch.inverse(I - (1 - alpha) * norm_adj)
        return S

    @staticmethod
    def to_sparse_edge_index(S, threshold=0.01):
        """将稠密扩散矩阵转为稀疏边索引和边权重"""
        mask = S > threshold
        row, col = torch.where(mask)
        edge_weight = S[mask]
        return torch.stack([row, col], dim=0), edge_weight


def denseAffinityRefine(adj, k):
    refine_adj = np.zeros_like(adj)
    indexs1 = np.tile(np.expand_dims(np.arange(adj.shape[0]), 0), (k, 1)).transpose()
    indexs2 = np.argpartition(adj, -k, 1)[:, -k:]
    refine_adj[indexs1, indexs2] = adj[indexs1, indexs2]
    return refine_adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class pygFpDataSet(Dataset):
    """自定义数据集，加载药物/蛋白质图、指纹，并添加扩散边"""
    def __init__(self, dataset, root_dir):
        super().__init__()
        self.dataset = dataset
        self.protein_graph = pickle.load(open(f"{root_dir}/davis_protein_graphs_with_esm6.pkl", "rb"))
        self.ligand_graph = pickle.load(open(f"{root_dir}/davis_drug_to_pyg_atom_aligned2.pkl", "rb"))
        self.ligand_fp = pickle.load(open(f"{root_dir}/ligand_to_fp.pkl", "rb"))

        self._add_diffusion_edges(self.ligand_graph)
        self._add_diffusion_edges(self.protein_graph)

        self.batch_ligand_graph = Batch.from_data_list(self.ligand_graph)
        self.batch_protein_graph = Batch.from_data_list(self.protein_graph)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_fn(self, batch):
        batch_ligand_graph = []
        batch_protein_graph = []
        batch_ligand_fp = []
        ys = []
        drug_ids = []
        target_ids = []

        for ligand_index, protein_index, y in batch:
            batch_protein_graph.append(self.protein_graph[protein_index])
            batch_ligand_graph.append(self.ligand_graph[ligand_index])
            batch_ligand_fp.append(self.ligand_fp[ligand_index])
            ys.append(y)
            drug_ids.append(ligand_index)
            target_ids.append(protein_index)

        ys = torch.Tensor(ys)
        drug_ids = torch.tensor(drug_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)

        batch_ligand_graph = Batch.from_data_list(batch_ligand_graph)
        batch_protein_graph = Batch.from_data_list(batch_protein_graph)
        batch_ligand_fp = torch.stack(batch_ligand_fp)

        return ys, batch_ligand_fp, batch_ligand_graph, batch_protein_graph, drug_ids, target_ids

    def _add_diffusion_edges(self, graph_list):
        for g in graph_list:
            if not hasattr(g, 'diff_edge_index'):
                adj = to_dense_adj(g.edge_index)[0]
                S = DiffusionMatrix.compute_ppr(adj, alpha=0.2)
                mask = S > 0.01
                row, col = torch.where(mask)
                edge_weight = S[mask]
                g.diff_edge_index = torch.stack([row, col], dim=0)
                g.diff_edge_weight = edge_weight


class testFpDataModule(pl.LightningModule):
    """PyTorch Lightning DataModule，封装数据集和亲和力图构建"""
    def __init__(self, **args):
        super().__init__()
        self.dataset_name = args["dataset_name"]
        self.batch_size = args["batch_size"]
        self.train_fold_index = 0
        self.full_affinity = None
        self.args = args

    def read_data(self, dataset):
        dataset_path = 'data/' + dataset + '/'
        affinity = pickle.load(open(dataset_path + 'davis_affinities', 'rb'), encoding='latin1')
        if dataset in ['davis_cold_target', 'davis_cold_drug', 'davis_cold', 'davis']:
            affinity = [-np.log10(y / 1e9) for y in affinity]
        affinity = np.asarray(affinity)
        return affinity

    def process_data(self, affinity_mat, dataset, num_pos, pos_threshold):
        dataset_path = 'data/' + dataset + '/'
        train_file = json.load(open(dataset_path + '/folds/train_fold_setting1.txt'))
        train_index = []
        for i in range(len(train_file)):
            train_index += train_file[i]
        test_index = json.load(open(dataset_path + '/folds/test_fold_setting1.txt'))

        rows, cols = np.where(np.isnan(affinity_mat) == False)
        train_rows, train_cols = rows[train_index], cols[train_index]
        train_Y = affinity_mat[train_rows, train_cols]
        test_rows, test_cols = rows[test_index], cols[test_index]
        test_Y = affinity_mat[test_rows, test_cols]

        train_affinity_mat = np.zeros_like(affinity_mat)
        train_affinity_mat[train_rows, train_cols] = train_Y
        affinity_graph, drug_pos, target_pos = self.get_affinity_graph(dataset, train_affinity_mat, num_pos, pos_threshold)

        return affinity_graph, drug_pos, target_pos

    def get_affinity_graph(self, dataset, adj, num_pos, pos_threshold):
        dataset_path = 'data/' + dataset + '/'
        num_drug, num_target = adj.shape[0], adj.shape[1]

        dt_ = adj.copy()
        dt_ = np.where(dt_ >= pos_threshold, 1.0, 0.0)
        dtd = np.matmul(dt_, dt_.T)
        dtd = dtd / dtd.sum(axis=-1).reshape(-1, 1)
        dtd = np.nan_to_num(dtd)
        dtd += np.eye(num_drug, num_drug)
        dtd = dtd.astype("float32")
        d_d = np.loadtxt(dataset_path + 'drug-drug-sim.txt', delimiter=',')
        dAll = dtd + d_d
        drug_pos = np.zeros((num_drug, num_drug))
        for i in range(len(dAll)):
            one = dAll[i].nonzero()[0]
            if len(one) > num_pos:
                oo = np.argsort(-dAll[i, one])
                sele = one[oo[:num_pos]]
                drug_pos[i, sele] = 1
            else:
                drug_pos[i, one] = 1
        drug_pos = sp.coo_matrix(drug_pos)
        drug_pos = sparse_mx_to_torch_sparse_tensor(drug_pos)

        td_ = adj.T.copy()
        td_ = np.where(td_ >= pos_threshold, 1.0, 0.0)
        tdt = np.matmul(td_, td_.T)
        tdt = tdt / tdt.sum(axis=-1).reshape(-1, 1)
        tdt = np.nan_to_num(tdt)
        tdt += np.eye(num_target, num_target)
        tdt = tdt.astype("float32")
        t_t = np.loadtxt(dataset_path + 'target-target-sim.txt', delimiter=',')
        tAll = tdt + t_t
        target_pos = np.zeros((num_target, num_target))
        for i in range(len(tAll)):
            one = tAll[i].nonzero()[0]
            if len(one) > num_pos:
                oo = np.argsort(-tAll[i, one])
                sele = one[oo[:num_pos]]
                target_pos[i, sele] = 1
            else:
                target_pos[i, one] = 1
        target_pos = sp.coo_matrix(target_pos)
        target_pos = sparse_mx_to_torch_sparse_tensor(target_pos)

        if dataset in ["davis", "davis_cold_target", "davis_cold_drug", "davis_cold"]:
            adj[adj != 0] -= 5
            adj_norm = minMaxNormalize(adj, 0)
        elif dataset == "kiba":
            adj_refine = denseAffinityRefine(adj.T, 150)
            adj_refine = denseAffinityRefine(adj_refine.T, 40)
            adj_norm = minMaxNormalize(adj_refine, 0)

        adj_1 = adj_norm
        adj_2 = adj_norm.T
        adj = np.concatenate((
            np.concatenate((np.zeros([num_drug, num_drug]), adj_1), 1),
            np.concatenate((adj_2, np.zeros([num_target, num_target])), 1)
        ), 0)

        train_row_ids, train_col_ids = np.where(adj != 0)
        edge_indexs = np.concatenate((
            np.expand_dims(train_row_ids, 0),
            np.expand_dims(train_col_ids, 0)
        ), 0)
        edge_weights = adj[train_row_ids, train_col_ids]
        node_type_features = np.concatenate((
            np.tile(np.array([1, 0]), (num_drug, 1)),
            np.tile(np.array([0, 1]), (num_target, 1))
        ), 0)
        adj_features = np.zeros_like(adj)
        adj_features[adj != 0] = 1
        features = np.concatenate((node_type_features, adj_features), 1)

        affinity_graph = DATA.Data(
            x=torch.Tensor(features),
            adj=torch.Tensor(adj),
            edge_index=torch.LongTensor(edge_indexs)
        )
        affinity_graph.__setitem__("edge_weight", torch.Tensor(edge_weights))
        affinity_graph.__setitem__("num_drug", num_drug)
        affinity_graph.__setitem__("num_target", num_target)

        return affinity_graph, drug_pos, target_pos

    def setup(self, stage):
        dataset_name = self.args["dataset_name"]
        self.dataset = pickle.load(open(f'./data/{dataset_name}/DTAs_new.pkl', 'rb'))
        root_dir = f'./data/{self.dataset_name}'

        train_fold = json.load(open(f"data/{dataset_name}/folds/train_fold_setting1.txt"))
        test_fold = json.load(open(f"data/{dataset_name}/folds/test_fold_setting1.txt"))

        trainset_fold = [self.dataset[index] for fold_index in set([0, 1, 2, 3, 4]) for index in train_fold[fold_index]]
        testset_fold = [self.dataset[index] for index in test_fold]

        self.train_dataset = pygFpDataSet(trainset_fold, root_dir)
        self.test_dataset = pygFpDataSet(testset_fold, root_dir)

        affinity = self.read_data(self.args["dataset_name"])
        dataset = self.args["dataset_name"]
        num_pos = 5
        pos_threshold = 8.0
        affinity_graph, drug_pos, target_pos = self.process_data(affinity, dataset, num_pos, pos_threshold)

        self.full_affinity = affinity_graph
        self.drug_pos = drug_pos
        self.target_pos = target_pos

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.test_dataset.collate_fn,
            batch_size=self.batch_size,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.test_dataset.collate_fn,
            batch_size=self.batch_size,
            pin_memory=(torch.cuda.is_available())
        )
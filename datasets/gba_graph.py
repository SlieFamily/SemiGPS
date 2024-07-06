import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import numpy as np
import pandas as pd

class GbaDataset(Dataset):
    def __init__(self, root="datasets/", transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

        self.node_feature = torch.load(root+"poi_feature.pt").float()
        self.move = pd.read_csv(root+"gba_move_cnt.csv", index_col=0)
        self.gdp = pd.read_csv(root+"gba_3gdp.csv", index_col=0)

    def len(self):
        return 1

    def get(self, idx):
        edge_index = torch.tensor(self.move[["O_JD_id","D_JD_id"]].values.tolist(), dtype=torch.long)
        edge_weight = torch.tensor(self.move["num_total"].values.tolist(), dtype=torch.float)
        y = torch.tensor(self.gdp[["_1stIn", "_2ndIn", "_3rdIn"]].values.tolist(), dtype=torch.float)
        return Data(x=self.node_feature, edge_index=edge_index.t().contiguous(), edge_weight=edge_weight, y=y)
# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : intro.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/04/01 16:18
# @Description: 

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
# edge_index.t().contiguous()

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index.t().contiguous())

data.validate()

data.keys()

data.has_isolated_nodes()
data.has_self_loops()
data.is_directed()
data.is_undirected()
data.stores

root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/tudatasets'
dataset = TUDataset(root=root, name='ENZYMES')
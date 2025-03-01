# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : datasplit.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/02/14 20:34
@Description: 
"""

from torch_geometric.data import Data
from torch_geometric.datasets import PPI
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit

import torch

def get_data():
    x = torch.randn(8, 32)
    y = torch.randint(0, 4, (8,))

    edge_index = torch.tensor([
        [2, 3, 3, 4, 5, 6, 7],
        [0, 0, 1, 1, 2, 3, 4]],
    )
    return x, y, edge_index

def node_prediction():
    """Node Prediction"""
    
    x, y, edge_index = get_data()

    data = Data(x=x, y=y, edge_index=edge_index)
    node_transform = RandomNodeSplit(num_val=2, num_test=2)
    node_splits = node_transform(data)

    return node_splits

def link_prediction():
    """Link Prediction"""

    x, y, edge_index = get_data()

    edge_y = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    data = Data(x=x, y=y, edge_index=edge_index, edge_y=edge_y)
    link_transform = RandomLinkSplit(num_val=.2, num_test=.2, key='edge_y', 
                                     is_undirected=False, add_negative_train_samples=False)
    train_data, val_data, test_data = link_transform(data)
    return train_data, val_data, test_data

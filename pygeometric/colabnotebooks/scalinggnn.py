# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : scalinggnn.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/04/14 17:26
# @Description: 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.optim import Adam

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data.data import BaseData
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import ClusterData, ClusterLoader


def process():

    root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/planetoid'
    dataset = Planetoid(root=root, name='PubMed', transform=NormalizeFeatures())
    data = dataset.get(0)

    torch.manual_seed(12345)
    cluster_data = ClusterData(data, num_parts=32)
    train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)

    total_num_nodes = 0
    for step, sub_data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
        print(sub_data)
        print()
        total_num_nodes += sub_data.num_nodes

class GCN(nn.Module):

    def __init__(self, dataset: Planetoid, hidden: int = 16) -> None:
        super().__init__()

        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.conv2 = GCNConv(hidden, dataset.num_classes)
    
    def forward(self, x: Tensor, edge_index: Tensor):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train(model: GCN, loader: ClusterLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer):
    model.train()

    for sub in loader:
        pred = model(sub.x, sub.edge_index)
        loss = criterion(pred[sub.train_mask],sub.y[sub.train_mask], )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    
def test(model: GCN, data: BaseData):
    model.eval()

    pred = model(data.x, data.edge_index).argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        accs.append(correct.sum() / int(mask.sum()))
    return accs


def main():

    root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/planetoid'
    dataset = Planetoid(root=root, name='PubMed', transform=NormalizeFeatures())
    data = dataset.get(0)

    torch.manual_seed(12345)
    cluster_data = ClusterData(data, num_parts=32)
    train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)

    model = GCN(dataset)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=.01, weight_decay=5e-4)

    for epoch in range(1, 51):
        train(model, train_loader, criterion, optimizer)
        train_acc, val_acc, test_acc = test(model, data)
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')



def demo():
    root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/planetoid'
    dataset = Planetoid(root=root, name='PubMed', transform=NormalizeFeatures())
    data = dataset.get(0)

    dataset.num_classes
    dataset.num_features
    dataset.num_edge_features

    data.num_edges
    data.num_nodes
    
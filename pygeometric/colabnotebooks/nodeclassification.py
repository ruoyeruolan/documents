# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : nodeclassification.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/04/09 14:32
# @Description: 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.data.data import BaseData
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures



def visualize(h: Tensor, color):

    z = TSNE().fit_transform(h.detach().cpu().numpy())
    
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

class MLP(nn.Module):

    def __init__(self, dataset: Planetoid, hidden) -> None:
        super().__init__()

        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
    
    def forward(self, x: Tensor) -> Tensor:

        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=.5, training=self.training)
        return self.lin2(x)

def train(model: nn.Module, data: BaseData, criterion: nn.Module, optimizer: torch.optim.Optimizer):
    model.train()

    optimizer.zero_grad()
    pred = model(data.x)
    loss = criterion(pred[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test(model: nn.Module, data: BaseData):
    model.eval()

    pred = model(data.x).argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) 
    return test_acc

class GCN(nn.Module):

    def __init__(self, dataset: Planetoid, hidden: int = 16, heads: int = 8, type: str = 'gcn') -> None:
        super().__init__()
        torch.manual_seed(1234567)

        if type == 'gcn':
            self.conv1 = GCNConv(dataset.num_features, hidden)
            self.conv2 = GCNConv(hidden, dataset.num_classes)
        elif type == 'gat':
            self.conv1 = GATConv(dataset.num_edge_features, hidden, heads=heads)
            self.conv2 = GATConv(hidden * 8, dataset.num_classes, heads=heads)
    
    def forward(self, x: Tensor, edge_index: Tensor):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
def train_(model: GCN, data: BaseData, criterion: nn.Module, optimizer: torch.optim.Optimizer):
    model.train()

    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    loss = criterion(pred[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test_(model: GCN, data: BaseData):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc

def mlp():
    root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/planetoid'
    dataset = Planetoid(root, name='Cora', transform=NormalizeFeatures())
    data = dataset.get(0)

    model = MLP(dataset=dataset, hidden=16)
    data.y[data.train_mask]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2, weight_decay=5e-4)
    for epoch in range(1, 201):
        loss = train(model, data, criterion=criterion, optimizer=optimizer)
        print(f'Epoch: {epoch: 03d}, Loss: {loss: .4f}')

def gcn(model: GCN, data: BaseData):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2, weight_decay=5e-4)
    for epoch in range(1, 201):
        loss = train_(model=model, data=data, criterion=criterion, optimizer=optimizer)
        print(f'Epoch: {epoch: 03d}, Loss: {loss: .4f}')
    
    acc = test_(model, data)
    print(f'Test Acc: {acc}')

    # model.eval()

    # pred = model(data.x, data.edge_index)
    # visualize(pred, color=data.y)
    return model

def main():
    root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/planetoid'
    dataset = Planetoid(root, name='Cora', transform=NormalizeFeatures())
    data = dataset.get(0)

    model = GCN(dataset, type='gat')
    model = gcn(model=model, data=data)

    model.eval()
    pred = model(data.x, data.edge_index)
    visualize(pred, color=data.y)



def demo():

    root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/planetoid'
    dataset = Planetoid(root, name='Cora', transform=NormalizeFeatures())
    data = dataset.get(0)
    
    data.num_edges
    data.num_nodes
    data.num_features

    data.is_directed()
    data.has_isolated_nodes()
    data.has_self_loops()

    dataset.num_classes
    dataset.num_edge_features
    dataset.num_features

    
    model = GCN(dataset)
    model.eval()

    pred = model(data.x, data.edge_index)
    visualize(pred, color=data.y)

    test_(model, data)


if __name__ == '__main__':

    main()
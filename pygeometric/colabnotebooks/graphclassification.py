# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : graphclassification.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/04/11 15:12
# @Description: 

from ast import main
from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.optim import Adam


from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import global_add_pool
from torch_geometric.data.data import BaseData
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.data import Dataset, InMemoryDataset


def get_data() -> Tuple[Any, DataLoader, DataLoader]:

    root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/tudatasets'
    dataset = TUDataset(root=root, name='MUTAG')
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # type: ignore
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)  # type: ignore
    return dataset, train_loader, test_loader

class GCN(nn.Module):

    def __init__(self, dataset: TUDataset, hidden: int = 16, type: str = 'gcn'):
        super().__init__()
        
        torch.manual_seed(12345)
        if type == 'gcn':
            self.conv1 = GCNConv(dataset.num_features, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.conv3 = GCNConv(hidden, hidden)
        elif type == 'graph':
            self.conv1 = GraphConv(dataset.num_features, hidden)
            self.conv2 = GraphConv(hidden, hidden)
            self.conv3 = GraphConv(hidden, hidden)

        self.lin = nn.Linear(hidden, dataset.num_classes)
    
    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_add_pool(x, batch=batch)
        
        x = F.dropout(x, p=.5, training=self.training)
        return self.lin(x)

def train(model: GCN, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer):
    model.train()
    
    for data in loader:
        pred = model(data.x, data.edge_index, data.batch)
        loss = criterion(pred, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(model: GCN, loader: DataLoader):
     model.eval()

     correct = 0
     total = 0
     for data in loader:
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1) 
         correct += int((pred == data.y).sum())
         total += data.num_graphs
     return correct / total

def mian():

    dataset, train_loader, test_loader = get_data()
    model = GCN(dataset=dataset)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=.01)

    for epoch in range(1, 201):
        train(model, train_loader, criterion, optimizer)

        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        print(f'Epoch: {epoch: 03d}, Trian Acc: {train_acc: .4f}, Test Acc: {test_acc: .4f}')


def demo():


    dataset, train_loader, test_loader = get_data()
    
    
    dataset.len()
    dataset.num_classes
    dataset.num_features
    dataset.num_edge_labels

    for step,data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)


if __name__ == '__main__':
    
    main()
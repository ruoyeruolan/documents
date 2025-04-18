# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : handsongnn.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/04/06 19:01
# @Description: 

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import KarateClub


import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(G: nx.Graph, color):

    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    nx.draw_networkx(
        G=G, 
        pos=nx.spring_layout(G=G, seed=42),
        with_labels=False,
        node_color=color,
        cmap='Set2'
    )
    plt.show()


def visualize_embedding(h: Tensor, color, epoch: int|None = None, loss = None):

    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()

def data_preprocess():

    dataset = KarateClub()
    data = dataset.get(0)
    data.num_nodes
    data.num_edges
    data.is_directed()
    data.node_attrs()
    data.train_mask.sum()
    data.has_self_loops()
    data.has_isolated_nodes()

    data.edge_index.t()

    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y)

    model = GCN(dataset=dataset)
    model.parameters()
    _, x = model(dataset.x, dataset.edge_index)
    visualize_embedding(x, color=dataset.y)

class GCN(nn.Module):

    def __init__(self, dataset: Data | InMemoryDataset) -> None:
        super().__init__()

        torch.manual_seed(1234)
        self.conv1 = GCNConv(in_channels=dataset.num_features, out_channels=4)
        self.conv2 = GCNConv(in_channels=4, out_channels=4)
        self.conv3 = GCNConv(in_channels=4, out_channels=2)
        self.classifier = Linear(in_features=2, out_features=dataset.num_classes)
    
    def forward(self, x: Tensor, edge_index: Tensor):

        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = self.conv3(x, edge_index)
        x = x.tanh()
        return self.classifier(x), x

def train(model: nn.Module, data):
    # model.train()  # here the `model.train()` is not necessary, becasuae the model not have layers such `Dropout` and `BatchNoramlization`
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    optimizer.zero_grad()
    pred, x = model(data.x, data.edge_index)
    loss = criterion(pred[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, x

if __name__ == '__main__':

    dataset = KarateClub()
    data = dataset.get(0)

    model = GCN(dataset)
    for epoch in range(1, 401):
        loss, x = train(model, data)
        print(f'Epoch: {epoch}, Loss: {loss}')
    
    visualize_embedding(x, color=data.y)

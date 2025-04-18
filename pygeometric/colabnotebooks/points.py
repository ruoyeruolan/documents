# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : points.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/04/15 18:50
# @Description: 

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch import Tensor
from torch_cluster import knn_graph, fps
from torch.nn import Sequential, Linear, ReLU

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import GeometricShapes
from torch_geometric.nn import MessagePassing, global_max_pool, PPFConv
from torch_geometric.transforms import SamplePoints, Compose, RandomRotate


def visualize_mesh(pos: Tensor, face: Tensor):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=face.t(), antialiased=False)
    plt.show()

def visualize_points(pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.axis('off')
    plt.show()


class PointNetLayer(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(aggr='max')

        torch.manual_seed(12345)
        self.mlp = Sequential(
            Linear(in_channels + 3, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )
    
    def forward(self, h: Tensor, pos: Tensor, edge_index: Tensor):
        return self.propagate(edge_index=edge_index, h=h, pos=pos)
    
    def message(self, h_j: Tensor, pos_j: Tensor, pos_i: Tensor):

        input = pos_j - pos_i

        if h_j is not None:
            input = torch.cat([h_j, input], dim=-1)
        return self.mlp(input)

class PointNet(nn.Module):

    def __init__(self, num_classes: int = 40) -> None:
        super().__init__()

        self.conv1 = PointNetLayer(3, 32)
        self.conv2 = PointNetLayer(32, 32)
    
        self.classifier = Linear(32, num_classes)
    
    def forward(self, pos: Tensor, batch: Tensor):

        edge_index = knn_graph(pos, k=16, loop=True, batch=batch)

        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        h = global_max_pool(h, batch)
        return self.classifier(h)


class PPFNet(nn.Module):
    
    def __init__(self, num_classses: int = 40):
        super().__init__()

        torch.manual_seed(12345)
        
        self.conv1 = PPFConv(
            local_nn=Sequential(
                Linear(4, 32),
                ReLU(),
                Linear(32, 32),
            )
        )

        self.conv2 = PPFConv(
            local_nn=Sequential(
                Linear(32 + 4, 32),
                ReLU(),
                Linear(32, 32),
            )
        )

        self.classifier = Linear(32, num_classses)

    def forward(self, pos: Tensor, batch: Tensor, normal):

        edge_index = knn_graph(pos, k=16, loop=False, batch=batch)

        h = self.conv1(h=pos, pos=pos, edge_index=edge_index, normal=normal)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index, normal=normal)
        h = h.relu()

        h = global_max_pool(h, batch=batch)
        return self.classifier(h)



def train(model: PointNet | PPFNet, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer):
    model.train()

    total = 0
    for data in loader:
        optimizer.zero_grad()
        pred = model(data.pos, data.batch)
        loss = criterion(pred, data.y)
        loss.backward()
        optimizer.step()
        total += loss.item() * data.num_graphs
    return total / len(loader.dataset) # type: ignore

@torch.no_grad()
def test(model: PointNet | PPFNet, loader: DataLoader):
    model.eval()

    correct = 0
    num_dataset = 0
    for data in loader:
        pred = model(data.pos, data.batch).argmax(dim=-1)
        correct += int((pred == data.y).sum())
        num_dataset += 1
    return correct / len(loader.dataset) # type: ignore


def main():

    root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/geometricShapes'
    train_dataset = GeometricShapes(root=root, train=True, transform=SamplePoints(num=128))
    test_dataset = GeometricShapes(root=root, train=False, transform=SamplePoints(num=128))

    train_loader = DataLoader(dataset=train_dataset, batch_size=10,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

    model = PointNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=.01)

    for epoch in range(1, 51):
        loss = train(model, train_loader, criterion, optimizer)
        test_acc = test(model, test_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')


if __name__ == '__main__':

    main()
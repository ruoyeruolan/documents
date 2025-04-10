# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : intro.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/04/01 16:18
# @Description: 

import torch
import torch.nn.functional as F


import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import BaseData
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.datasets import TUDataset, Planetoid,ShapeNet
from torch_geometric.nn import GCNConv


def demo():
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


    root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/planetoid'
    dataset = Planetoid(root=root, name='Cora')
    dataset[0]['train_mask']
    dataset[0]['y']

    root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/tudatasets'
    dataset = TUDataset(root=root, name='ENZYMES')

    dataset.shuffle()

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        # print(batch)
        print(batch['batch'])

    root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/shapenet'
    dataset = ShapeNet(
        root=root, 
        categories=['Airplane'], 
        pre_transform=T.KNNGraph(k=6), 
        transform=T.RandomJitter(.01),
    )

    ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])

class GCN(torch.nn.Module):

    def __init__(self, dataset: Dataset|InMemoryDataset) -> None:
        super().__init__()

        self.conv1 = GCNConv(in_channels=dataset.num_node_features, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=dataset.num_classes)
    
    def forward(self, data: Data):

        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def main():

    root = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/planetoid'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid(root=root, name='Cora')
    data = dataset[0]
    
    model = GCN(dataset).to(device)
    data = data.to(device.type)  # type: ignore
    optimizer = torch.optim.Adam(params=model.parameters(), lr=.01, weight_decay=5e-4)

    model.train()
    for epoch in range(1, 101):

        optimizer.zero_grad()
        pred = model(data)
        loss = F.nll_loss(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss}')
    
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')


if __name__ == '__main__':
    main()
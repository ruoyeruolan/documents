# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : graphtransformer.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/02/15 20:09
@Description: 
"""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool


def load_dataset(path: str | None = None):
    if path is None:
        # path = '/public/workspace/ryrl/data/pygeometrics/zinc'
        path = '.'
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    train_dataset = ZINC(root=path, subset=False, split='train', pre_transform=transform)  # List[Data]
    val_dataset = ZINC(root=path, subset=False, split='val', pre_transform=transform)
    test_dataset = ZINC(root=path, subset=False, split='test', pre_transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    return train_loader, val_loader, test_loader

class GPS(nn.Module):

    def __init__(self, channels: int, pe_dim: int, num_layers: int, attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()

        self.node_emb = nn.Embedding(num_embeddings=28, embedding_dim=channels - pe_dim)
        self.pe_lin = nn.Linear(in_features=20, out_features=pe_dim)
        self.pe_norm = nn.BatchNorm1d(num_features=20)
        self.edge_emb = nn.Embedding(num_embeddings=4, embedding_dim=channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            pipeline = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, channels)
            )

            conv = GPSConv(
                channels=channels,
                conv=GINEConv(nn=pipeline),
                heads=4,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs
            )

            self.convs.append(conv)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, 1)
        )

        self.redraw_projection = RedrawProjection(
            self.convs, redraw_interval=1000 if attn_type == 'performer' else None
        )

    def forward(self, x: Tensor, pe, edge_index, edge_attr, batch):

        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        x = global_add_pool(x, batch)
        return self.mlp(x)

class RedrawProjection:

    def __init__(self, model: nn.Module, redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0
    
    def redraw_projections(self):

        if not self.model.training or self.redraw_interval is None:
            return
        
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules() if isinstance(module, PerformerAttention)
            ]

            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return 
        self.num_last_redraw += 1


def train(model: GPS, train_loader, optimizer: torch.optim.Optimizer, device):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.redraw_projection.redraw_projections()
        out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)

        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(model: GPS, loader, device):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)

        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)


def run():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_kwargs = {'dropout': 0.5}

    model = GPS(channels=64, pe_dim=8, num_layers=10, attn_type='performer', attn_kwargs=attn_kwargs).to(device)
    optimizer = Adam(params=model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)

    train_loader, val_loader, test_loader = load_dataset()

    for epoch in range(1, 500):
        loss = train(model, train_loader,optimizer=optimizer, device=device)
        val_mae = test(model, val_loader, device=device)
        test_mae = test(model, test_loader, device=device)

        scheduler.step(val_mae)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, Test: {test_mae:.4f}')
    torch.save(model.state_dict(), '/public/workspace/ryrl/IdeaProjects/Projects/torch/pyGenometrics/Cases/model.pth')


if __name__ == '__main__':
    run()

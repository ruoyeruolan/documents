# -*- encoding: utf-8 -*-
# @Introduce  : 
# @File       : messagepassing.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/04/04 18:51
# @Description: 

from cv2 import norm
import torch
import torch.nn as nn

from torch import Tensor

from torch.nn import Linear, Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(aggr='add')

        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        
        self.lin.reset_parameters()
        self.bias.data.zero_()
    
    def forward(self, x: Tensor, edge_index: Tensor):
        
        edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=x.size(0))

        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        y = self.propagate(edge_index, x=x, norm=norm)
        y += self.bias
        return y
    
    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j
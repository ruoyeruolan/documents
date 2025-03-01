# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : DiseaseGeneInteractions.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/02/24 23:30
@Description: 
"""

from sympy import degree
import torch
import numpy as np
import pandas as pd

from typing import Tuple

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree


# _cached_data_: pd.DataFrame | None = None

def load_data(offset: int = 0, index_col: int | str = 0):
    
    data= pd.read_csv('./Cases/DiseaseGene/data.tsv', sep='\t', header=0, index_col=index_col)
    mapping = {index_id: i + offset for i, index_id in enumerate(data.index.unique())}
    return mapping

def load_edge_list(
        src_col: str, src_mapping: dict, 
        dst_col: str, dst_mapping: dict,
) -> Tensor:

        data = pd.read_csv('./Cases/DiseaseGene/data.tsv', sep='\t', header=0) 
        src_nodes = [src_mapping[idx] for idx in data[src_col]]
        dst_nodes = [dst_mapping[idx] for idx in data[dst_col]]
        edge_index = torch.tensor([src_nodes, dst_nodes])
        return edge_index

def initialize_data(num_features: int = 1) -> Tuple[Data, dict, dict]:

    dz, gene_col = '# Disease ID', 'Gene ID'
    dz_mapping = load_data(index_col=dz)
    gene_mapping = load_data(index_col=gene_col, offset=519)

    edge_index = load_edge_list(dz, dz_mapping, gene_col, gene_mapping)
    reverse_edge_index = load_edge_list(gene_col, gene_mapping, dz, dz_mapping)

    # if edge_index is None or reverse_edge_index is None:
    #     raise ValueError("Edge indices not initialized properly")

    data = Data()
    data.num_nodes = len(dz_mapping) + len(gene_mapping)
    data.edge_index = torch.cat((edge_index, reverse_edge_index), dim=1)
    data.x = torch.ones((data.num_nodes, num_features))
    return data, dz_mapping, gene_mapping


def main():
     data, dz_mapping, gene_mapping = initialize_data()
     reverse_dz_mapping = {val: key for key, val in dz_mapping.items()}
     reverse_gene_mapping = {val: key for key, val in gene_mapping.items()}

     degree(data.edge_index[0])

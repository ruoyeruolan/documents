# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : DiseaseGeneInteractions.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/02/24 23:30
@Description: 
"""


import torch
import numpy as np
import pandas as pd

from torch import dropout
import torch.nn as nn
import networkx as nx
import plotly.express as px
import matplotlib.pyplot as plt
import torch_geometric.transforms as T

from typing import Tuple

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree

from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from torch_geometric.nn import GCNConv, GAE


# _cached_data_: pd.DataFrame | None = None

def load_data(offset: int = 0, index_col: int | str = 0):

    path = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/Cases/DiseaseGene/data.tsv'
    data= pd.read_csv(path, sep='\t', header=0, index_col=index_col)
    mapping = {index_id: i + offset for i, index_id in enumerate(data.index.unique())}
    return mapping

def load_edge_list(
        src_col: str, src_mapping: dict, 
        dst_col: str, dst_mapping: dict,
) -> Tensor:

        path = '/Users/wakala/IdeaProjects/Projects/documents/pygeometric/Cases/DiseaseGene/data.tsv'
        data = pd.read_csv(path, sep='\t', header=0) 
        src_nodes = [src_mapping[idx] for idx in data[src_col]]
        dst_nodes = [dst_mapping[idx] for idx in data[dst_col]]
        edge_index = torch.tensor([src_nodes, dst_nodes])
        return edge_index

def initialize_data(num_features: int = 20) -> Tuple[Data, dict, dict]:

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
    
    if data.num_nodes is None:
        raise ValueError("Number of nodes is not initialized properly")
    
    data.x = torch.ones((int(data.num_nodes), num_features))
    return data, dz_mapping, gene_mapping


def analysis_graph():
    df = pd.read_csv('/Users/wakala/IdeaProjects/Projects/documents/pygeometric/Cases/DiseaseGene/data.tsv', sep='\t')
    data, dz_mapping, gene_mapping = initialize_data()
    reverse_dz_mapping = {val: key for key, val in dz_mapping.items()}
    reverse_gene_mapping = {val: key for key, val in gene_mapping.items()}

    if data.edge_index is not None:
        degrees = degree(data.edge_index[0]).numpy()
        sorted_degree_i = np.argsort(-1 * degrees)  # sort the tensor in descending and return the index of sorted tensor
    else:
        raise ValueError("Edge index is not initialized properly")
    
    print('Disease Nodes with highest degree')
    top_k = 10
    num_dz_nodes = 0

    for i in sorted_degree_i:
        if i < 519:
            node_id = reverse_dz_mapping[i]
            node_description = df[df['# Disease ID'] == node_id]['Disease Name'].drop_duplicates().values
            node_degree = degrees[i]
            print(
                "node_index=" + str(i), 
                "node_degree=" + str(int(node_degree)),
                node_id, 
                node_description
            )

            num_dz_nodes += 1
            if num_dz_nodes >= top_k:
                break
    
    print("\nGene Nodes of highest degree")
    num_gene_nodes = 0
    for i in sorted_degree_i:
        if i > 519:
            node_id = reverse_gene_mapping[i]
            node_degree = degrees[i]

            print("node_index=" + str(i), "node_degree=" + str(int(node_degree)), node_id)
            num_gene_nodes += 1
            if num_gene_nodes >= top_k:
                break
    return None

def data_preparation(data: Data, num_features: int = 20, device: str = 'cpu') -> Tuple[Data, Data, Data]:

    data.x = torch.ones((data.num_nodes if data.num_nodes else 0, num_features))

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(
            num_val=.05, 
            num_test=.15, 
            split_labels=True,
            is_undirected=True,
            add_negative_train_samples=True, 
        ),
    ])

    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data


# Helper functions
def get_mapping():
    data_path = "./DG-AssocMiner_miner-disease-gene.tsv"
    df = pd.read_csv(data_path, index_col="Disease Name", sep="\t")
    disease_mapping = [index_id for index_id in enumerate(df.index.unique())]
    df = pd.read_csv(data_path, index_col="Gene ID", sep="\t")
    gene_mapping = [index_id[1] for index_id in enumerate(df.index.unique())]
    mapping = disease_mapping + gene_mapping
    return mapping

def visualize_tsne_embeddings(
        model, 
        data, 
        title, 
        perplexity=30.0, 
        labeled=False, 
        labels=[]
):
    """Visualizes node embeddings in 2D space with t-SNE.

    Args: model, pass in the trained or untrained model
            data, Data object, where we assume the first 519 datapoints are disease
            nodes and the rest are gene nodes
            title, title of the plot
            perplexity, t-SNE hyperparameter for perplexity
    """
    model.eval()
    x = data.x
    z = model.encode(x, data.edge_index)
    ax1, ax2 = zip(*TSNE(n_components=2, learning_rate='auto', perplexity=perplexity,
                        init='random').fit_transform(z.detach().cpu().numpy()))

    fig = px.scatter(x=ax1, y=ax2, color=['r']*519 + ['g']*7294,
                    hover_data=[get_mapping()],
                    title=title)

    if labeled:
        for i in labels:
            fig.add_annotation(x=ax1[i], y=ax2[i], text=str(i), showarrow=False)
    fig.show()

def visualize_pca_embeddings(model, data, title, labeled=False, labels=[]):
    """Visualizes node embeddings in 2D space with PCA (components=2)

    Args: model, pass in the trained or untrained model
            data, Data object, where we assume the first 519 datapoints are disease
            nodes and the rest are gene nodes
            title, title of the plot
    """
    
    model.eval()
    x = data.x
    z = model.encode(x, data.edge_index)

    pca = PCA(n_components=2)
    components = pca.fit_transform(z.detach().cpu().numpy())
    fig = px.scatter(components, x=0, y=1, color=['r']*519 + ['g']*7294,
                    hover_data=[get_mapping()], title=title)

    if labeled:
        for i in labels:
            fig.add_annotation(x=components[:,0][i], y=components[:,1][i], text=str(i), showarrow=False)
    fig.show()

def plot_roc_curve(model, data):
    """Visualizes ROC curve of model predictions

    Args: model, pass in the trained or untrained model
            data, Data object, where we assume the first 519 datapoints are disease
    """
    model.eval()

    x = data.x
    z = model.encode(x, data.edge_index)

    pos_preds = model.decode(z, data.pos_edge_label_index, sigmoid=True)
    neg_preds = model.decode(z, data.neg_edge_label_index, sigmoid=True)
    preds = torch.cat([pos_preds, neg_preds], dim=0)
    preds = preds.detach().cpu().numpy()

    labels = torch.cat((data.pos_edge_label, data.neg_edge_label), dim=0)
    labels = labels.detach().cpu().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)

    # Using J-statistic: https://en.wikipedia.org/wiki/Youden%27s_J_statistic
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    print('Best Threshold=%f' % (best_thresh))

    roc_auc = metrics.roc_auc_score(labels, preds)

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--') # diagonal roc curve of a random classifier
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best=%0.2f' % best_thresh)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = 'lower right')
    plt.title('ROC curve for model predictions')
    plt.show()

def plot_training_stats(title, losses, test_auc, test_ap, train_auc, train_ap):
    """Plots evolution of loss and metrics during training

    Args: losses, test_auc, test_ap, train_auc, and train_ap should be lists
        outputted by the training process.
    """
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.set_xlabel("Training Epochs")
    ax2.set_ylabel("Performance Metric")
    ax.set_ylabel("Loss")

    plt.title(title)
    p1, = ax.plot(losses, "b-", label="training loss")
    p2, = ax2.plot(test_auc, "r-", label="test AUC")
    p3, = ax2.plot(test_ap, "g-", label="test AP")
    p4, = ax2.plot(train_auc, "o-", label="train AUC")
    p5, = ax2.plot(train_ap, "v-", label="train AP")
    plt.legend(handles=[p1, p2, p3, p4, p5])
    plt.show()

def get_edge_dot_products(data, model, num_dz_nodes=519):
  """
  A pair of nodes (u,v) is predicted to be connected with an edge if the dot
  product between the learned embeddings of u and v is high. This function
  computes and returns the dot product of all pairs of (dz_node, gene_node).

  Args:
    data, the data_object containing the original node featues
    model, the model that will be used to encode the data
    num_dz_nodes, the number of disease nodes; used to differentiate between
      disease and gene node embeddings
  Returns:
    dot_products, a numpy 2D array of shape (num_dz_nodes, num_gene_nodes)
      containing the dot product between each (dz_node, gene_node) pair.
  """
  model.eval()
  x = data.x
  z = model.encode(x, data.edge_index).detach().numpy()
  dz_z = z[:num_dz_nodes, :]
  gene_z = z[num_dz_nodes:, :]

  dot_products = np.einsum('ai,bi->ab', dz_z, gene_z)
  return dot_products   # numpy array of shape (num_dz_nodes, num_gene_nodes)


def get_ranked_edges(data: Data, model, num_dz_nodes: int = 519):
    """
    Ranks all potential edges as predicted by the model.

    Args:
        data, the data_object containing the original node featues
        model, the model that will be used to encode the data
        num_dz_nodes, the number of disease nodes; used to differentiate between
        disease and gene node embeddings
    Returns:
        ranked_edge_list, a full edge list ranked by the likelihood of the edge
        being a positive edge, in decreasing order
        ranked_dot_products, a list of the dot products of each edge's node
        embeddings, ranked in decreasing order
    """
    # Get dot products
    edge_dot_products = get_edge_dot_products(data, model, num_dz_nodes=num_dz_nodes)
    num_possible_edges = edge_dot_products.shape[0] * edge_dot_products.shape[1]

    # Get indeces, ranked by dot product in descending order. This is a tuple (indeces[0], indeces[1]).
    ranked_edges = np.unravel_index(np.argsort(-1 * edge_dot_products, axis=None), edge_dot_products.shape)
    assert len(ranked_edges[0]) == num_possible_edges

    # Get the corresponding, ranked edge list and ranked dot products. Note that
    # we need to add an offset for the gene_node indeces.
    offset = np.array([np.zeros(num_possible_edges, dtype=int), num_dz_nodes + np.ones(num_possible_edges, dtype=int)]).T
    ranked_edge_list = np.dstack(ranked_edges)[0] + offset
    assert ranked_edge_list.shape[0] == num_possible_edges

    # Get the corresponding ranked dot products
    ranked_dot_products = edge_dot_products[ranked_edges]
    assert ranked_dot_products.shape[0] == num_possible_edges

    return ranked_edge_list, ranked_dot_products

class GCNEncoder(nn.Module):

    def __init__(self, in_channels: int = 20, hidden_channels: int = 200, out_channels: int = 20, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels, cached=True)
        self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=out_channels, cached=True)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        return self.conv2(x, edge_index)

gae = GAE(encoder=GCNEncoder()).to(device='cpu')


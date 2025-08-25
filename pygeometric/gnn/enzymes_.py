# -*- encoding: utf-8 -*-
# @Introduce  :
# @File       : enzymes_.py
# @Author     : ryrl
# @Email      : ryrl970311@gmail.com
# @Time       : 2025/08/25 16:19
# @Description: https://medium.com/@a.r.amouzad.m/how-to-get-state-of-the-art-result-on-graph-classification-with-gnn-73afadff5d49

import random
import networkx as nx
import matplotlib.pyplot as plt

from typing import Tuple
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_mean_pool

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def draw_random_graph_samples(
    dataset, num_classes, num_samples=20, name: str = "ENZYMES"
):
    num_rows = 4
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    fig.suptitle(f"Random Samples of {name} Graphs")

    # Define a colormap based on the number of unique classes
    cmap = plt.get_cmap("tab10", num_classes)

    # Get random indices for sampling
    random_indices = random.sample(range(len(dataset)), num_samples)

    for i, index in enumerate(random_indices):
        row = i // num_cols
        col = i % num_cols

        data = dataset[index]
        edge_index, y = data.edge_index, data.y
        G = nx.Graph()
        for src, dst in edge_index.t().tolist():
            G.add_edge(src, dst)

        # Map class labels to colors
        nodes_color = cmap(y[0])

        pos = nx.spring_layout(G)  # Positions for all nodes
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=nodes_color,
            node_size=100,
            edge_color="k",
            linewidths=1,
            font_size=5,
            ax=axs[row, col],  # type: ignore
        )
        axs[row, col].set_title(f"Graph Sample {index}")  # type: ignore

    # Custom legend for class labels
    handles = [
        Line2D(
            [],
            [],
            color=cmap(i),
            marker="o",
            linestyle="",
            markersize=10,
            label=f"Class {i}",
        )
        for i in range(num_classes)
    ]
    plt.legend(
        handles=handles,
        title="Class Labels",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    # plt.close()
    plt.show()

    # # Save the plot
    # plt.savefig(
    #     os.path.join(model_save_path, f"{dataset_name}/random_samples_of_graphs.png")
    # )


def preprocess():
    dataset = TUDataset(
        "/Users/wakala/IdeaProjects/Projects/documents/pygeometric/data/tudatasets",
        name="ENZYMES",
        use_node_attr=True,
    )

    # Split dataset into training and test sets
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42
    )

    # Draw graphs from the test dataset before training
    # draw_random_graph_samples(test_dataset, dataset.num_classes)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)
    return dataset, train_loader, test_loader


class GINClassification(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
        )

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
        )

        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )
        )

        self.fc = nn.Linear(32, out_channels)

    def forward(self, x, edge_index, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch=batch)
        classification = self.fc(x)
        return classification, x


def model_train(
    model: GINClassification, loader: DataLoader, optimizer: torch.optim.Optimizer
):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        pred, _ = model(data.x, data.edge_index, data.batch)
        loss = torch.nn.CrossEntropyLoss()(pred, data.y)
        loss.backward()
        optimizer.step()


def model_test(model: GINClassification, loader: DataLoader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        labels = []
        preds = []
        for data in loader:
            pred, _ = model(data.x, data.edge_index, data.batch)
            pred = pred.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs

            labels.extend(data.y.numpy())
            preds.extend(pred.numpy())
        f1 = f1_score(labels, preds, average="weighted")
    return correct / total, f1


def main():
    dataset, train_loader, test_loader = preprocess()

    model = GINClassification(dataset.num_node_features, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    log_data = {"Epoch": [], "Test Accuracy": [], "F1 Score": []}
    best_acc = 0.0
    best_f1 = 0.0
    for epoch in range(1, 101):
        optimizer.zero_grad()
        model_train(model, train_loader, optimizer)

        train_acc, train_f1 = model_test(model, train_loader)
        test_acc, test_f1 = model_test(model, test_loader)
        print(
            f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
            f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}"
        )

        log_data["Epoch"].append(epoch)
        log_data["Test Accuracy"].append(test_acc)
        log_data["F1 Score"].append(test_f1)

        if test_acc > best_acc or (test_acc == best_acc and test_f1 > best_f1):
            best_acc = test_acc
            best_f1 = test_f1
    return log_data, best_acc, best_f1


if __name__ == "__main__":
    log_data, best_acc, best_f1 = main()
    print(f"Best Test Accuracy: {best_acc:.4f}, Best F1 Score: {best_f1:.4f}")

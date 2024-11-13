import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import to_networkx

from utils import *


class ModelTypeError(ValueError):
    def __init__(self, message="Invalid model type. Please use '1-input' or '2-input'."):
        super().__init__(message)


# model 1
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, out_channels))

    def forward(self, graph):
        # x, edge_index = graph.x, graph.edge_index
        x, edge_index = graph.pos, graph.edge_index
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.silu(x)
        x = self.layers[-1](x, edge_index)
        x = F.sigmoid(x)
        return x


# model 2
class MaxAggConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MaxAggConv, self).__init__(aggr="max")  # Use 'max' as the aggregation method
        # Add a linear transformation (learnable weights)
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Apply the linear transformation before propagating
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Apply a function to the message (incoming node features)
        # return F.sigmoid(x_j)
        return x_j

    def update(self, aggr_out):
        # Return the aggregated values (max values from neighbors)
        return aggr_out


class GCN_2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(GCN_2, self).__init__()
        self.layers = nn.ModuleList()
        # self.batch_norms = nn.ModuleList()
        self.layers.append(MaxAggConv(in_channels, hidden_channels))
        # self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.layers.append(MaxAggConv(hidden_channels, hidden_channels))
            # self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        self.layers.append(MaxAggConv(hidden_channels, out_channels))

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.silu(x)
            # x = self.batch_norms[i](x)
        # x = F.relu(x)
        x = self.layers[-1](x, edge_index)
        # x = torch.sigmoid(x)
        return x


class BaseModel:
    def __init__(self, args, path, graph, device):
        self.name = "BaseModel"
        self.args = args
        self.path = path
        self.graph = graph
        self.model = GCN(*args.channels, args.num_layers).to(device)
        # self.model = GCN_2(*args.channels, args.num_layers).to(device)
        self.optimizer = self._set_optimizer()
        self.criterion = self._set_loss_function()

    def _set_optimizer(self):
        if self.args.opt == "adam":
            return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=5e-4)
        elif self.args.opt == "adamax":
            return optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=5e-4)
        elif self.args.opt == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=5e-4)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.opt}")

    def _set_loss_function(self):
        if self.args.loss == "MSE":
            return nn.MSELoss()
        elif self.args.loss == "CrossEntropy":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {self.args.loss}")

    def show_all_attributes(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

    def train(self):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.graph).squeeze(dim=-1)
        loss = self.criterion(out[self.graph.train_mask], self.graph.y[self.graph.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self):
        """Evaluate the model on test nodes"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.graph).squeeze(dim=-1)
            # Calculate MSE only for test nodes
            test_loss = self.criterion(out[self.graph.test_mask], self.graph.y[self.graph.test_mask])
            print(f"Test MSE: {test_loss.item()}")
            return test_loss.item()

    def fit(self):
        """Train the model"""
        for epoch in range(self.args.epochs):
            loss = self.train()
            if epoch == 0 or (epoch + 1) % 300 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss}")
        torch.save(self.model.state_dict(), f"{self.path}/{self.name}_model.pth")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def load(self, modelPath=None):
        Path = modelPath if modelPath else self.path
        self.model.load_state_dict(torch.load(f"{Path}/{self.name}_model.pth"))
        print("Model loaded.")


def set_up_model(args, query_set, column_intervals, modelPath, table_size):
    if args.model == "1-input":
        pass
    elif args.model == "2-input":
        pass
    else:
        raise ModelTypeError


def Visualize_compare_Graph_2D(
    graph,
    out,
    args,
    save_path,
    figsize=(10, 8),
    to_undirected=True,
    with_labels=False,
    font_size=24,
    colorbar_label="Node Values",
    node_size = 100,
):
    fig, axs = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'hspace': 0.15, 'wspace': 0.15})
    G = to_networkx(graph, to_undirected=to_undirected)
    pos = {i: v for i, v in enumerate(graph.pos)}
    vmin, vmax = 0, 1  # Set color range (adjust as needed)

    # Row 1: Ground Truth
    # Plot 1.1: Train data (Ground Truth)
    train_indices = graph.train_mask.nonzero(as_tuple=True)[0]
    train_colors = torch.full_like(graph.y, float("nan"))
    train_colors[train_indices] = graph.y[train_indices].cpu()

    nodes = nx.draw(
        G,
        pos,
        with_labels=with_labels,
        node_color=train_colors,
        cmap=plt.get_cmap("coolwarm"),
        vmin=vmin,
        vmax=vmax,
        ax=axs[0, 0],
        node_size=node_size
    )
    axs[0, 0].set_title("Train Data (Ground Truth)", fontsize=font_size)

    # Plot 1.2: Test data (Ground Truth)
    test_indices = graph.test_mask.nonzero(as_tuple=True)[0]
    test_colors = torch.full_like(graph.y, float("nan"))
    test_colors[test_indices] = graph.y[test_indices].cpu()

    nx.draw(
        G,
        pos,
        with_labels=with_labels,
        node_color=test_colors,
        cmap=plt.get_cmap("coolwarm"),
        vmin=vmin,
        vmax=vmax,
        ax=axs[0, 1],
        node_size=node_size
    )
    axs[0, 1].set_title("Test Data (Ground Truth)", fontsize=font_size)

    # Row 2: Model Predictions
    # Plot 2.1: Train data (Model Prediction)
    train_pred_colors = torch.full_like(out, float("nan"))
    train_pred_colors[train_indices] = out[train_indices].cpu()

    nx.draw(
        G,
        pos,
        with_labels=with_labels,
        node_color=train_pred_colors,
        cmap=plt.get_cmap("coolwarm"),
        vmin=vmin,
        vmax=vmax,
        ax=axs[1, 0],
        node_size=node_size
    )
    axs[1, 0].set_title("Train Data (Model Prediction)", fontsize=font_size)

    # Plot 2.2: Test data (Model Prediction)
    test_pred_colors = torch.full_like(out, float("nan"))
    test_pred_colors[test_indices] = out[test_indices].cpu()

    plot = nx.draw(
        G,
        pos,
        with_labels=with_labels,
        node_color=test_pred_colors,
        cmap=plt.get_cmap("coolwarm"),
        vmin=vmin,
        vmax=vmax,
        ax=axs[1, 1],
        node_size=node_size
    )
    axs[1, 1].set_title("Test Data (Model Prediction)", fontsize=font_size)

    # Add labels if required
    if args.plot_labels:
        train_labels = {i: f"{val:.2f}" for i, val in enumerate(graph.y[train_indices].cpu()) if not torch.isnan(val)}
        test_labels = {i: f"{val:.2f}" for i, val in enumerate(graph.y[test_indices].cpu()) if not torch.isnan(val)}
        train_pred_labels = {i: f"{val:.2f}" for i, val in enumerate(out[train_indices].cpu()) if not torch.isnan(val)}
        test_pred_labels = {i: f"{val:.2f}" for i, val in enumerate(out[test_indices].cpu()) if not torch.isnan(val)}
        
        nx.draw_networkx_labels(G, pos, labels=train_labels, ax=axs[0, 0], font_size=font_size)
        nx.draw_networkx_labels(G, pos, labels=test_labels, ax=axs[0, 1], font_size=font_size)
        nx.draw_networkx_labels(G, pos, labels=train_pred_labels, ax=axs[1, 0], font_size=font_size)
        nx.draw_networkx_labels(G, pos, labels=test_pred_labels, ax=axs[1, 1], font_size=font_size)

    # Add a shared color bar using the last plot in the array
    cbar = fig.colorbar(axs[1, 1].collections[0], ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label(colorbar_label, fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    # Adjust layout and save
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.15, hspace=0.15, wspace=0.15)
    plt.tight_layout()
    plt.savefig(f"{save_path}/train_test_2x2_compare_plot.png", dpi=300)
    plt.show()
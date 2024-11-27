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


class NodeDataLoader:
    def __init__(self, graph, batch_size):
        self.graph = graph
        self.batch_size = batch_size
        self.train_idx = torch.where(graph.train_mask)[0]  # Only nodes with train_mask=True

    def __iter__(self):
        indices = self.train_idx[torch.randperm(len(self.train_idx))]  # Shuffle indices each epoch
        for start in range(0, len(indices), self.batch_size):
            batch_nodes = indices[start : start + self.batch_size]
            yield batch_nodes

    def __len__(self):
        return (len(self.train_idx) + self.batch_size - 1) // self.batch_size


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
        # self.node_loader = NodeDataLoader(graph, batch_size=args.bs)

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
        # self.model.train()
        # total_loss = 0
        # for batch_nodes in self.node_loader:
        #     self.optimizer.zero_grad()
        #     # Forward pass only for the batched nodes and their neighbors
        #     out = self.model(self.graph).squeeze(dim=-1)
        #     loss = self.criterion(out[batch_nodes], self.graph.y[batch_nodes])
        #     loss.backward()
        #     self.optimizer.step()
        #     total_loss += loss.item() * len(batch_nodes)

        # avg_loss = total_loss / len(self.node_loader.train_idx)
        # return avg_loss

    def evaluate(self):
        """Evaluate the model on test nodes"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.graph).squeeze(dim=-1)
            # Calculate MSE only for test nodes
            test_loss = self.criterion(
                out[self.graph.test_mask], self.graph.y[self.graph.test_mask]
            )
            print(f"Test MSE: {test_loss.item()}")
            return test_loss.item()

    def fit(self):
        """Train the model"""
        best_loss = np.inf
        for epoch in range(self.args.epochs):
            loss = self.train()
            if epoch == 0 or (epoch + 1) % 300 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss}")
            # if loss < best_loss:
            #     best_loss = loss
            #     torch.save(self.model.state_dict(), f"{self.path}/{self.name}_model.pth")
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
    figsize=(12, 6),
    to_undirected=True,
    with_labels=False,
    font_size=16,
    colorbar_label="Node Values",
    node_size=20,
    edge_linewidth=0.1,
):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    # Adjust the grid layout to reduce spacing
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.02)
    G = to_networkx(graph, to_undirected=to_undirected)
    pos = {i: v for i, v in enumerate(graph.pos)}
    vmin, vmax = 0, 1  # Set color range (adjust as needed)

    # Plot 1: Train data (Ground Truth)
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
        ax=axs[0],
        node_size=node_size,
        width=edge_linewidth,
    )
    axs[0].set_title("Ground Truth", fontsize=font_size)

    # Plot 2: Train data (Model Prediction)
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
        ax=axs[1],
        node_size=node_size,
        width=edge_linewidth,
    )
    axs[1].set_title("Model Prediction", fontsize=font_size)

    # Add labels if required
    if args.plot_labels:
        train_labels = {
            i: f"{val:.2f}"
            for i, val in enumerate(graph.y[train_indices].cpu())
            if not torch.isnan(val)
        }
        train_pred_labels = {
            i: f"{val:.2f}"
            for i, val in enumerate(out[train_indices].cpu())
            if not torch.isnan(val)
        }

        nx.draw_networkx_labels(G, pos, labels=train_labels, ax=axs[0], font_size=font_size)
        nx.draw_networkx_labels(G, pos, labels=train_pred_labels, ax=axs[1], font_size=font_size)

    # Add a shared color bar
    cbar = fig.colorbar(
        axs[1].collections[0], ax=axs, orientation="horizontal", fraction=0.05, pad=0.05
    )
    cbar.set_label(colorbar_label, fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size - 4)

    # Adjust layout and save
    # plt.tight_layout()
    plt.savefig(f"{save_path}/train_compare_plot.png", dpi=300)
    plt.show()

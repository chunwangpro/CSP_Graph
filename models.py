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
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.sigmoid(x)
        x = torch.max(x, dim=1)[0]  # torch.max(x, dim=1, keepdim=True)[0]
        return x


# model 2
class MaxAggConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MaxAggConv, self).__init__(aggr="max")  # Use 'max' as the aggregation method
        # Add a linear transformation (learnable weights)
        self.lin = nn.Linear(in_channels, out_channels)  #

    def forward(self, x, edge_index):
        # Apply the linear transformation before propagating
        x = self.lin(x)  #
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Apply a function to the message (incoming node features)
        # return F.sigmoid(x_j)
        return x_j

    def update(self, aggr_out):
        # Return the aggregated values (max values from neighbors)
        return aggr_out


class GCN_2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_2, self).__init__()
        self.conv1 = MaxAggConv(in_channels, hidden_channels)
        self.conv2 = MaxAggConv(hidden_channels, out_channels)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class BaseModel:
    def __init__(self, args, path, graph, device):
        self.name = "BaseModel"
        self.args = args
        self.path = path
        self.graph = graph
        self.model = GCN_2(*args.channels).to(device)
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


def set_up_model(args, query_set, unique_intervals, modelPath, table_size):
    if args.model == "1-input":
        pass
    elif args.model == "2-input":
        pass
    else:
        raise ModelTypeError


def Visualize_compare_Graph_2D(
    graph,
    column_interval_number,
    out,
    figsize=(15, 6),
    to_undirected=False,
    with_labels=True,
):
    # compare graph we learned with the initial graph
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    G = to_networkx(graph, to_undirected=to_undirected)
    pos = {
        i: np.array(np.unravel_index(i, column_interval_number)) + 1
        for i in range(graph.x.shape[0])
    }
    # ground truth
    nx.draw(
        G,
        pos,
        with_labels=with_labels,
        node_color=graph.y.cpu(),
        cmap=plt.get_cmap("coolwarm"),
        ax=axs[0],
    )
    axs[0].set_title("Ground Truth")

    # model output
    masked_out = torch.full(out.shape, float("nan"))
    masked_out[graph.train_mask] = out[graph.train_mask]
    nx.draw(
        G,
        pos,
        with_labels=with_labels,
        node_color=masked_out.detach().cpu(),
        cmap=plt.get_cmap("coolwarm"),
        ax=axs[1],
    )
    axs[1].set_title("Model output")

    plt.colorbar(axs[1].collections[0], ax=axs[1])
    plt.tight_layout()
    # plt.savefig("./images/2D_compare.png", dpi=300)
    plt.show()
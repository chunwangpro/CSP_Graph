import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import to_networkx

np.random.seed(42)


# hyperparameters
x_grid_size, y_grid_size = 4, 3
unique_number = [x_grid_size, y_grid_size]
num_nodes = np.product(unique_number)


# define node and edge
node_features = torch.tensor(
    [[i, j] for j in range(y_grid_size) for i in range(x_grid_size)], dtype=torch.float
)

edges = []
for j in range(y_grid_size):
    for i in range(x_grid_size):
        node_id = j * x_grid_size + i
        # out edge: up, right
        if i < x_grid_size - 1:
            edges.append([node_id, j * x_grid_size + (i + 1)])
        if j < y_grid_size - 1:
            edges.append([node_id, (j + 1) * x_grid_size + i])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
graph = Data(x=node_features, edge_index=edge_index)


# visualize the initial graph structure
G = to_networkx(graph, to_undirected=False)
pos = {i: (graph.x[i, 0].item(), graph.x[i, 1].item()) for i in range(graph.x.shape[0])}
plt.figure(figsize=(4, 3))
plt.title("2D Grid with Directed Edges (Out: Up, Right)")
nx.draw(
    G, pos, with_labels=True, node_color="lightblue", node_size=500, arrows=True, arrowstyle="-|>"
)
plt.savefig("./images/2d_grid.png", dpi=300)
# plt.show()


# select points to be labeled
# num_labels = 7
# selected_points = torch.randperm(num_nodes)[:num_labels]
selected_points = [1, 2, 4, 6, 7, 8, 10]


def get_CDF_2D(selected_points):
    labels = torch.full((num_nodes,), float("nan"))
    for node in selected_points:
        x, y = node % x_grid_size + 1, node // x_grid_size + 1
        labels[node] = x * y
    return labels / num_nodes


# set train set
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[selected_points] = True
graph.y = get_CDF_2D(selected_points)  # label: CDF F(x)
graph.train_mask = train_mask


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


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = MaxAggConv(in_channels, hidden_channels)
        self.conv2 = MaxAggConv(hidden_channels, out_channels)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set model
model = GCN(2, 16, 1).to(device)
graph = graph.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.MSELoss()


# print model output before training
# out = model(graph).squeeze(dim=-1).detach().cpu()
# out[graph.train_mask], graph.y[graph.train_mask]


# train
def train():
    model.train()
    optimizer.zero_grad()
    out = model(graph).squeeze(dim=-1)
    loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(3000):
    loss = train()
    if epoch % 300 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


# model output
model.eval()
out = model(graph).squeeze(dim=-1).detach().cpu()
# print(f"\n{out=}")
print(f"\n{out[graph.train_mask]=}")
print(f"\n{graph.y[graph.train_mask]=}")
err = (out[graph.train_mask] - graph.y[graph.train_mask]).pow(2).mean()
print(f"Final MSE: {err}")

# visualization
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
# Ground truth
G = to_networkx(graph, to_undirected=False)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=graph.y.cpu(),
    cmap=plt.get_cmap("coolwarm"),
    ax=axs[0],
    vmin=0,
    vmax=1,
)
axs[0].set_title("Ground Truth")
# model output
masked_out = torch.full(out.shape, float("nan"))
masked_out[graph.train_mask] = out[graph.train_mask]
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=masked_out.detach().cpu(),
    cmap=plt.get_cmap("coolwarm"),
    ax=axs[1],
    vmin=0,
    vmax=1,
)
axs[1].set_title("Model output")

plt.colorbar(axs[1].collections[0], ax=axs[1])
plt.tight_layout()
plt.savefig("./images/2D_demo.png", dpi=300)
# plt.show()


#### visualize only the model output
# visualization
fig, axs = plt.subplots(1, 1, figsize=(6, 4))
# model output
masked_out = torch.full(out.shape, float("nan"))
masked_out[graph.train_mask] = out[graph.train_mask]
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=masked_out.detach().cpu(),
    cmap=plt.get_cmap("coolwarm"),
    ax=axs,
    # vmin=0,
    # vmax=1,
)
# axs.set_title("Model output")

plt.colorbar(axs.collections[0], ax=axs)
plt.tight_layout()
plt.savefig("./images/2D_demo_only_model.png", dpi=300)
# plt.show()

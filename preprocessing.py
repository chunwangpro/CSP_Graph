import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from utils import *


def get_num_nodes(column_interval_number):
    """
    Calculate total number of nodes in a grid graph.
    """
    return torch.prod(torch.tensor(column_interval_number), dtype=torch.int64)


def get_strides(column_interval_number):
    """
    Strides are used to convert multi-dim coordinates to 1D indices, vice versa.
    """
    return torch.tensor(
        [
            torch.prod(torch.tensor(column_interval_number[i + 1 :]))
            for i in range(len(column_interval_number))
        ],
        device=device,
    )


def torch_unravel_index(indices, column_interval_number, strides):
    """
    Samilar as np.unravel_index, convert 1D indices to multi-dim coordinates, but we need to provide column_interval_number and strides (can be calculated by get_strides).
    Pytorch can leverage GPU for parallel computing, more efficient than numpy, and support vecterized operations.

    example:
    indices = torch.tensor([0, 6])
    column_interval_number = [5, 5]
    strides = get_strides(column_interval_number)

    torch_unravel_index(indices, column_interval_number, strides) --> torch.tensor([[0, 0], [1,1]])
    """
    return torch.stack(
        [
            (indices // strides[i]) % column_interval_number[i]
            for i in range(len(column_interval_number))
        ],
        dim=1,
    )


def torch_ravel_multi_index(coords, strides):
    """
    Similar as np.ravel_multi_index, convert multi-dim coordinates to 1D indices, but we need to provide strides (can be calculated by get_strides).
    Pytorch can leverage GPU for parallel computing, more efficient than numpy, and support vecterized operations.

    example:
    coords = torch.tensor([[0, 0], [1, 1]])
    column_interval_number = [5, 5]
    strides = get_strides(column_interval_number)

    torch_ravel_multi_index(coords, strides) --> tensor([0, 6])
    """
    return torch.sum(coords * strides, dim=1, dtype=torch.int64)


def define_node_edge_multi_dims(
    column_interval_number, num_nodes, strides, batch_size=100000, device="cuda"
):
    """
    Define the node and edge for multi-dim grid graph, using PyTorch.
    """

    def compute_edges_batch(start, end, column_interval_number, strides, device):
        indices = torch.arange(start, end, device=device, dtype=torch.int64)

        # convert 1D indices to multi-dim coordinates
        coords = torch_unravel_index(indices, column_interval_number, strides)

        all_edges = []
        # Find neighbors in each dimension
        for dim in range(len(column_interval_number)):
            neighbor_mask = coords[:, dim] < (column_interval_number[dim] - 1)
            neighbor_coords = coords[neighbor_mask].clone()
            neighbor_coords[:, dim] += 1
            # convert multi-dim coordinates to 1D indices
            neighbors = torch_ravel_multi_index(neighbor_coords, strides)
            edges = torch.stack([indices[neighbor_mask], neighbors], dim=1)
            all_edges.append(edges)
        return torch.cat(all_edges, dim=0)

    # Connect edges in batches
    edges = []
    for batch_start in tqdm(range(0, num_nodes, batch_size)):
        batch_end = min(batch_start + batch_size, num_nodes)
        batch_edges = compute_edges_batch(
            batch_start, batch_end, column_interval_number, strides, device
        )
        edges.append(batch_edges)

    # Build the graph
    edges = torch.cat(edges, dim=0)
    edge_index = edges.t().contiguous()

    # initialize the nodes
    node_features = torch.arange(num_nodes, device=device, dtype=torch.float32).reshape(-1, 1)
    graph = Data(x=node_features, edge_index=edge_index)
    # graph.pos store the multi-dim coordinates of each node (starts from [0, 0]), whereas graph.x store the 1D indices of each node (stars from 0).
    node_features = torch.arange(num_nodes, device=device, dtype=torch.float32)
    graph.pos = torch_unravel_index(node_features, column_interval_number, strides)

    print("Nodes:", graph.num_nodes)
    print("Edges:", graph.num_edges)
    # check the correctness of the graph edge number
    assert graph.num_edges == theoretical_edge_count(column_interval_number)
    return graph


def theoretical_edge_count(column_interval_number):
    """
    Calculate the number of edges (in theory) in a grid graph for any number of dimensions given by column_interval_number, use this function to check the correctness of the edge count when building the graph.
    """
    total_edges = 0
    for i in range(len(column_interval_number)):
        other_dims = np.prod(column_interval_number[:i] + column_interval_number[i + 1 :])
        edges_in_dim = other_dims * (column_interval_number[i] - 1)
        total_edges += edges_in_dim
    return total_edges


def build_train_set_1_input(query_set, column_intervals, args, table_size):
    """
    Build the training set for 1-input model from the query set and unique intervals.
    """
    X = []
    for query in query_set:
        x = [v[-1] for v in column_intervals.values()]
        idxs, _, vals, _ = query
        for i, v in zip(idxs, vals):
            x[i] = v
        X.append(x)
    X = np.array(X, dtype=np.float32)
    y = np.array([query[-1] for query in query_set], dtype=np.float32).reshape(-1, 1)
    y /= table_size[0]
    train = np.hstack((X, y))

    # make train set unique
    if args.unique_train:
        train = np.unique(train, axis=0)

    # add boundary
    if args.boundary:
        train = add_boundary_1_input(train, column_intervals)

    # shuffle and split
    # np.random.shuffle(train)
    X, y = np.hsplit(train, [-1])
    return X, y


def add_boundary_1_input(train, column_intervals, alpha=0.1):
    return train


def build_train_set_2_input(query_set, column_intervals, args, table_size):
    pass


def add_boundary_2_input(train, column_intervals, alpha=0.1):
    return train


def replace_with_index(X, column_intervals):
    """
    Replace the values in X with the index in its corresponding column unique intervals.
    """
    for i in range(X.shape[1]):
        mapping_list = torch.tensor(column_intervals[i], device=device)
        X[:, i] = torch.searchsorted(mapping_list, X[:, i])
    return X


def define_train_mask_for_graph(X, y, graph, num_nodes, strides, column_intervals):
    """
    Since our method is a semi-supervised learning method, we need to define the training mask for the graph.
    """
    X = torch.tensor(X, device=device)
    X = replace_with_index(X, column_intervals)
    selected_points = torch_ravel_multi_index(X, strides)
    graph.y = torch.full((num_nodes,), float("nan"))
    graph.y[selected_points] = torch.tensor(y).squeeze()
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[selected_points] = True
    graph.train_mask = train_mask
    return graph

def define_train_test_masks_for_graph(X, y, graph, num_nodes, strides, column_intervals, test_ratio=0.2):
    """
    Define train and test masks for the graph, creating a semi-supervised split.
    """
    X = torch.tensor(X, device=device)
    X = replace_with_index(X, column_intervals)
    selected_points = torch_ravel_multi_index(X, strides)
    graph.y = torch.full((num_nodes,), float("nan"))
    graph.y[selected_points] = torch.tensor(y).squeeze()
    
    # Create train mask
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[selected_points] = True
    
    # Now create test mask by sampling a portion of the selected points
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    num_test = int(len(selected_points) * test_ratio)
    test_indices = selected_points[torch.randperm(len(selected_points))[:num_test]]
    test_mask[test_indices] = True

    # Ensure train_mask excludes test_mask points
    train_mask[test_indices] = False

    # Assign the masks to the graph
    graph.train_mask = train_mask
    graph.test_mask = test_mask
    return graph


def setup_graph(args, query_set, column_intervals, column_interval_number, table_size, test=True):
    """
    Setup the training set and model based on the model type.
    X: Train X, query intervals. e.g. [a,b) for each column in 2-input model; (-inf, a] for each column in 1-input model.
    y: Train y, cardinality.
    m: Model.
    values: Unique intervals of each column, it will be used to generate grid intervals in table generation phase after model is well-trained. e.g. [a,b) for each column in 2-input model; (-inf, a] for each column in 1-input model.
    """
    num_nodes = get_num_nodes(column_interval_number)
    strides = get_strides(column_interval_number)
    graph = define_node_edge_multi_dims(
        column_interval_number, num_nodes, strides, batch_size=100000, device=device
    )
    X, y = build_train_set_1_input(query_set, column_intervals, args, table_size)
    if test:
        graph = define_train_test_masks_for_graph(X, y, graph, num_nodes, strides, column_intervals)
    else:
        graph = define_train_mask_for_graph(X, y, graph, num_nodes, strides, column_intervals)
    return graph


def Visualize_initial_Graph_2D(graph, column_interval_number, save_path):
    # visualize the initial graph structure
    G = to_networkx(graph, to_undirected=False)
    pos = {
        i: np.array(np.unravel_index(i, column_interval_number)) + 1
        for i in range(graph.x.shape[0])
    }
    plt.figure(figsize=(10, 8))
    plt.title("2D Grid with Directed Edges (Out: Up, Right)")
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        arrows=True,
        arrowstyle="-|>",
    )
    plt.savefig(f"{save_path}/initial_graph.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    column_interval_number = [100, 100, 100, 10]
    num_nodes = get_num_nodes(column_interval_number)
    strides = get_strides(column_interval_number)
    graph = define_node_edge_multi_dims(
        column_interval_number, num_nodes, strides, batch_size=100000, device="cpu"
    )

import argparse

from dataset import *
from models import *
from preprocessing import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="1-input", help="model type")
parser.add_argument("--dataset", type=str, default="test-2", help="Dataset.")
parser.add_argument("--query-size", type=int, default=1000, help="query size")
parser.add_argument("--min-conditions", type=int, default=1, help="min num of query conditions")
parser.add_argument("--max-conditions", type=int, default=2, help="max num of query conditions")
parser.add_argument(
    "--unique-train", type=bool, default=True, help="whether make train set unique."
)
parser.add_argument(
    "--boundary", type=bool, default=False, help="whether add boundary point to train set."
)
parser.add_argument(
    "--channels", type=str, default="2,16,1", help="Comma-separated list of channels."
)
parser.add_argument("--num_layers", type=int, default=3, help="Number of hidden layers.")
parser.add_argument("--epochs", type=int, default=3000, help="Number of train epochs.")
parser.add_argument("--bs", type=int, default=1000, help="Batch size.")
parser.add_argument("--loss", type=str, default="MSE", help="Loss.")
parser.add_argument("--opt", type=str, default="adam", help="Optimizer.")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")


try:
    args = parser.parse_args()
except:
    # args = parser.parse_args([])
    args, unknown = parser.parse_known_args()

args.channels = [int(x) for x in args.channels.split(",")]


FilePath = (
    f"{args.dataset}_{args.query_size}_{args.min_conditions}_{args.max_conditions}_{args.model}"
)


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


resultsPath = f"results/{FilePath}"
make_directory(resultsPath)


print("\nBegin Loading Data ...")
table, original_table_columns, sorted_table_columns, max_decimal_places = load_and_process_dataset(
    args.dataset, resultsPath
)
table_size = table.shape
print(f"{args.dataset}.csv")
print(f"Table shape: {table_size}")
print("Done.\n")


print("Begin Generating Queries Set ...")
rng = np.random.RandomState(42)
query_set = [generate_random_query(table, args, rng) for _ in tqdm(range(args.query_size))]
print("Done.\n")


print("Begin Intervalization ...")
unique_intervals = column_intervalization(query_set, table_size, args)
column_interval_number = count_unique_vals_num(unique_intervals)
print(f"{column_interval_number=}")
print("Done.\n")


print("Begin Building Graph and Model ...")
graph = setup_graph(args, query_set, unique_intervals, column_interval_number, table_size)
# pos = [
#     np.array(np.unravel_index(i, column_interval_number)).reshape(1, -1) + 1
#     for i in range(graph.x.shape[0])
# ]
# pos = np.concatenate(pos, axis=0)
# graph.pos = torch.from_numpy(pos).float()
# Visualize_initial_Graph_2D(graph, column_interval_number)
model = BaseModel(args, resultsPath, graph, device)
graph = graph.to(device)
print("Done.\n")


print("Begin Model Training ...\n")
model.fit()
print("Done.\n")

print("Begin Model Prediction ...")
# model.load()
out = model.predict(graph).squeeze(dim=-1).detach().cpu()
print(f"\nGround Truth:\n{graph.y[graph.train_mask]}")
print(f"\nModel Output:\n{out[graph.train_mask]}")
err = (out[graph.train_mask] - graph.y[graph.train_mask]).pow(2).mean()
print(f"\nFinal MSE: {err}")
print("\nDone.\n")


Visualize_compare_Graph_2D(
    graph,
    out,
    resultsPath,
    figsize=(15, 6),
    to_undirected=True,
    with_labels=False,
)
# Table_Generated = m.generate_table_by_row(values, batch_size=10000)
# Q_error = calculate_Q_error(Table_Generated, query_set)
# print_Q_error(Q_error, args, resultsPath)
# print(f"\n Original table shape : {table_size}")
# print(f"Generated table shape : {Table_Generated.shape}")

# recovered_Table_Generated = recover_table_as_original(
#     Table_Generated, original_table_columns, sorted_table_columns, max_decimal_places
# )
# recovered_Table_Generated.to_csv(f"{resultsPath}/generated_table.csv", index=False, header=False)

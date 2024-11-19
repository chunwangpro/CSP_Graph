import argparse
import os

from dataset import *
from models import *
from preprocessing import *
from utils import *

import time

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
parser.add_argument("--num_layers", type=int, default=1, help="Number of hidden layers.")
parser.add_argument("--epochs", type=int, default=3000, help="Number of train epochs.")
parser.add_argument("--bs", type=int, default=1000, help="Batch size.")
parser.add_argument("--loss", type=str, default="MSE", help="Loss.")
parser.add_argument("--opt", type=str, default="adam", help="Optimizer.")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--test", type=bool, default=True, help="wheter train-test split.")
parser.add_argument("--plot", type=bool, default=False, help="whether to plot.")
parser.add_argument(
    "--plot_labels",
    type=bool,
    default=False,
    help="whether add labels (selectivity) in compare plot.",
)


try:
    args = parser.parse_args()
except:
    # args = parser.parse_args([])
    args, unknown = parser.parse_known_args()

args.channels = [int(x) for x in args.channels.split(",")]


FilePath = (
    f"{args.model}_{args.dataset}_{args.query_size}_({args.min_conditions}_{args.max_conditions})"
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


print("Begin Generating Queries ...")
rng = np.random.RandomState(42)
query_set = [generate_random_query(table, args, rng) for _ in tqdm(range(args.query_size))]
split = int(np.round(len(query_set) * 0.8))
query_set_train = query_set[:split]
query_set_test = query_set[split:]
print("Done.\n")


print("Begin Intervalization ...")
column_intervals_train = column_intervalization(query_set_train, table_size, args)
column_interval_number_train = count_unique_vals_num(column_intervals_train)
print(f"{column_interval_number_train=}")
print("Done.\n")


print("Begin Building Graph and Model ...")
graph_train = setup_graph(args, query_set_train, column_intervals_train, column_interval_number_train, table_size, test=False)
print(f"shape of graph.x: {graph_train.x.shape}")
print(f"shape of graph.pos: {graph_train.pos.shape}")
print(f"shape of train mask: {graph_train.train_mask.shape}")
# pos = [
#     np.array(np.unravel_index(i, column_interval_number)).reshape(1, -1) + 1
#     for i in range(graph.x.shape[0])
# ]
# pos = np.concatenate(pos, axis=0)
# graph.pos = torch.from_numpy(pos).float()
# Visualize_initial_Graph_2D(graph, column_interval_number)
model = BaseModel(args, resultsPath, graph_train, device)
graph_train = graph_train.to(device)
print("Done.\n")


print("Begin Model Training ...\n")
start_time = time.time()
model.fit()
end_time = time.time()
print(f"train time: {end_time - start_time}")
print("Done.\n")

print("Begin Model Prediction ...")
model.load()
start_time = time.time()
out_train = model.predict(graph_train).squeeze(dim=-1).detach().cpu()
end_time = time.time()
print(f"inference time: {end_time - start_time}")
# print(f"\nGround Truth:\n{graph.y[graph.train_mask]}")
# print(f"\nModel Output:\n{out[graph.train_mask]}")
# err = (out[graph.train_mask] - graph.y[graph.train_mask]).pow(2).mean()
# print(f"\nFinal MSE: {err}")
# print("\nDone.\n")

if args.plot:
    Visualize_compare_Graph_2D(
        graph_train,
        out_train,
        args,
        resultsPath,
        # figsize=(30, 15),
        to_undirected=True,
        with_labels=False,
    )

def generate_table_by_row(graph, out):
    ls_1 = []
    ls_2 = []
    size = table.shape[0]
    card = np.round(out.numpy() * size).astype(int)

    rows = graph.pos[:, 0].detach().cpu().numpy().astype(int)
    cols = graph.pos[:, 1].detach().cpu().numpy().astype(int)
    max_row = rows.max()
    max_col = cols.max()
    cdf = np.zeros((max_row + 1, max_col + 1))
    cdf[rows, cols] = card
    print(f"cdf: {cdf}")
    
    arr_1 = np.zeros((1, cdf.shape[1]))
    arr_2 = np.zeros((cdf.shape[0]+1, 1))
    cdf_pad = np.concatenate([arr_2, np.concatenate([arr_1, cdf], axis=0)], axis=1)
    pdf = cdf_pad[1:, 1:] - cdf_pad[1:, :-1] - cdf_pad[:-1, 1:] + cdf_pad[:-1, :-1]
    pdf[pdf < 0] = 0
    pdf = pdf.astype(int)
    print(f"pdf: {pdf}")
    print(f"sum of pdf: {np.sum(pdf)}")
    
    for i in range(graph.pos.shape[0]):
        idx_1, idx_2 = graph.pos.detach().cpu().numpy()[i, 0].astype(int), graph.pos.detach().cpu().numpy()[i, 1].astype(int)
        ls_1.extend([column_intervals_train[0][int(graph.pos[i, 0])]] * pdf[idx_1, idx_2])
        ls_2.extend([column_intervals_train[1][int(graph.pos[i, 1])]] * pdf[idx_1, idx_2])
    
    return np.concatenate([np.array(ls_1).reshape(-1,1), np.array(ls_2).reshape(-1,1)], axis=1)

Table_Generated = generate_table_by_row(graph_train, out_train)

print("Summary of Q-error:")
print(args)
df_train = print_Q_error(Table_Generated, query_set_train)
df_train.to_csv(f"{resultsPath}/Q_error_train.csv", index=True, header=False)
df_test = print_Q_error(Table_Generated, query_set_test)
df_test.to_csv(f"{resultsPath}/Q_error_test.csv", index=True, header=False)
print(df_train)
print(df_test)
print(f"\n Original table shape : {table_size}")
print(f"Generated table shape : {Table_Generated.shape}")

print("Begin Recovering Data ...")
recovered_Table_Generated = recover_table_as_original(
    Table_Generated, original_table_columns, sorted_table_columns, max_decimal_places
)
recovered_Table_Generated.to_csv(f"{resultsPath}/generated_table.csv", index=False, header=False)
print("Done.\n")
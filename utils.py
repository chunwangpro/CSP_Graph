import random as rn
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# set seed
seed_value = 42
rn.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
tf.random.set_seed(seed_value)

# set device for PyTorch
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
#     device = torch.device("mps")
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
print("\nPyTorch is using device:", device)

# Operators dictionary
OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "=": np.equal,
}


def generate_random_query(table, args, rng):
    """Generate a random query."""
    conditions = rng.randint(args.min_conditions, args.max_conditions + 1)
    if args.model == "1-input":
        ops = rng.choice(["<="], replace=True, size=conditions)
    elif args.model == "2-input":
        ops = rng.choice(["<", "<=", ">", ">=", "="], replace=True, size=conditions)
    idxs = rng.choice(table.shape[1], replace=False, size=conditions)
    idxs = np.sort(idxs)
    cols = table[:, idxs]
    vals = table[rng.randint(0, table.shape[0]), idxs]
    card = calculate_query_cardinality(cols, ops, vals)
    return idxs, ops, vals, card


def column_unique_interval(table):
    # get unique values for each column
    column_interval = {i: np.unique(table[:, i]) for i in range(table.shape[1])}
    return column_interval


def column_intervalization(query_set, table_size, args):
    """
    Get unique intervals (included in queries) for each column, the number of intervals are less than the number of unique values.

    Returns:
    column_interval: dict, where the key is the column index and the value is a list of unique intervals.

    column_interval = {
        0: [99360],
        1: [110, 280, 660, 775],
        2: [960, 1390, 1450, 1700],
        3: [43],
        ...,
    }
    """
    column_interval = {i: set() for i in range(table_size[1])}
    for query in query_set:
        idxs, _, vals, _ = query
        for i in range(len(idxs)):
            column_interval[idxs[i]].add(vals[i])
    # Apply the column_interval to <, <=, >, >=, =
    for k, v in column_interval.items():
        interval_list = sorted(list(v))
        if args.model == "1-input":
            column_interval[k] = interval_list
            continue
        add_small = 2 * interval_list[0] - interval_list[1]
        add_big_1 = 2 * interval_list[-1] - interval_list[-2]
        add_big_2 = 3 * interval_list[-1] - 2 * interval_list[-2]
        column_interval[k] = [add_small] + interval_list + [add_big_1, add_big_2]
    return column_interval


def count_unique_vals_num(column_interval):
    """count unique interval number for each column."""
    return [len(v) for v in column_interval.values()]


def calculate_query_cardinality(data, ops, vals):
    """
    Use ops and vals as queries to find the number of rows in data that meet the conditions.

    Parameters:
    data (2D-array): The subset of table columns involved in the query. Table columns not involved in the query are not included in data.
    ops (1D-array): A list of operators, support operators: '>', '>=', '<', '<=', '='.
    vals (1D-array): A list of values.

    Returns:
    int: The cardinality (number of rows) that satisfy the query.

    Example:
    for empty table, use np.empty((n_row, n_col), dtype=np.float32), return 0.

    for non-empty table:
    table = np.array([[1, 2, 3, 4, 5],
                      [10, 20, 30, 40, 50],
                      [10, 20, 30, 40, 50]]).T
    data = table[:, [1, 2]]
    ops = [">=", ">="]
    vals = [20, 20]
    result = calculate_query_cardinality(data, ops, vals)
    print(result)
    """

    # assert data.shape[1] == len(ops) == len(vals)
    bools = np.ones(data.shape[0], dtype=bool)
    for i, (o, v) in enumerate(zip(ops, vals)):
        bools &= OPS[o](data[:, i], v)
    return bools.sum()


def calculate_Q_error(dataNew, query_set):
    Q_error = []
    for query in tqdm(query_set):
        idxs, ops, vals, card_true = query
        card_pred = calculate_query_cardinality(dataNew[:, idxs], ops, vals)
        if card_pred == 0 and card_true == 0:
            Q_error.append(1)
        elif card_pred == 0:
            Q_error.append(card_true)
        elif card_true == 0:
            Q_error.append(card_pred)
        else:
            Q_error.append(max(card_pred / card_true, card_true / card_pred))
    return Q_error


def print_Q_error(Table_Generated, query_set):
    Q_error = calculate_Q_error(Table_Generated, query_set)
    statistics = {
        "min": np.min(Q_error),
        "10": np.percentile(Q_error, 10),
        "20": np.percentile(Q_error, 20),
        "30": np.percentile(Q_error, 30),
        "40": np.percentile(Q_error, 40),
        "median": np.median(Q_error),
        "60": np.percentile(Q_error, 60),
        "70": np.percentile(Q_error, 70),
        "80": np.percentile(Q_error, 80),
        "90": np.percentile(Q_error, 90),
        "95": np.percentile(Q_error, 95),
        "99": np.percentile(Q_error, 99),
        "max": np.max(Q_error),
        "mean": np.mean(Q_error),
    }
    df = pd.DataFrame.from_dict(statistics, orient="index", columns=["Q-error"])
    df.index.name = None
    return df


def time_count(tic, toc):
    total_time = toc - tic
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    print(f"Time passed:  {h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}")

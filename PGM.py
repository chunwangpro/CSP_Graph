# This is the implementation of LPALG(PGM) algorithm for query optimization.
# "Data generation using declarative constraints"

import argparse
import copy
import itertools
import time

from scipy import optimize

from dataset import *
from plot_util import *
from utils import *


def Find_column_interval_idxs_by_op(column, index, op, column_interval_number):
    """Find all matching indexs in the column based on query interval and the operator. Apply to both "1-input" and "2-input" model types"""
    end = column_interval_number[column]
    if op == "<=":
        return list(range(0, index + 1))
    elif op == "<":
        return list(range(0, index))
    elif op == ">":
        return list(range(index + 1, end))
    elif op == ">=":
        return list(range(index, end))
    elif op == "=":
        return [index]
    else:
        raise ValueError("Invalid operator")


def Assign_query_to_interval_idx(query_set, n_column, column_interval, column_interval_number):
    """
    Convert query to corresponding column interval indexs, interval index are independent for each column (i.e. each column range from 0).

    Returns:
    query_to_interval_idx: dict, key is query index, value is a list of column interval index.

    query_to_interval_idx = {
        0: [ [0, 1], [], [], [] ],
        1: [ [], [], [0], [] ],
        2: [ [0], [0, 1, 2, 3, 4], [], [] ],
        ...,
    }

    Here,
    - query 0 only include column 0 and include interval index 0 and 1.
    - query 1 only include column 2 and include interval index 0.
    - query 2 include column 0 and 1, and include interval index 0 for column 0 and interval index 0 to 4 for column 1.

    Refer to column_interval for the interval index mapping. You can use _reveal_query_to_interval_idx(query_to_interval_idx, column_interval) to convert the interval index to the original query format, to better understanding.
    """
    query_to_interval_idx = {i: [[] for _ in range(n_column)] for i in range(len(query_set))}
    for i in range(len(query_set)):
        idxs, ops, vals, _ = query_set[i]
        for j in range(len(idxs)):
            col = idxs[j]
            index = column_interval[col].index(vals[j])
            query_to_interval_idx[i][col] = Find_column_interval_idxs_by_op(
                col, index, ops[j], column_interval_number
            )
    return query_to_interval_idx


def _reveal_query_to_interval_idx(query_to_interval_idx, column_interval):
    """This is a helper function to print the query_to_interval_idx in the query format, to check the implement of Assign_query_to_interval_idx."""
    for query_idx, x_idxs in query_to_interval_idx.items():
        print(f"\nquery {query_idx}")
        for col, x_idx in enumerate(x_idxs):
            if x_idx:
                interval = [column_interval[col][i] for i in x_idx]
                print(f" column {col} interval {interval}")


def Fill_query_to_interval_idx(query_to_interval_idx, column_interval_number):
    """Fill the empty list in query_to_interval_idx with all index range for that column."""
    query_to_full_interval_idx = copy.deepcopy(query_to_interval_idx)
    for k, v in query_to_full_interval_idx.items():
        for i in range(len(v)):
            if not v[i]:
                query_to_full_interval_idx[k][i] = list(range(column_interval_number[i]))
    return query_to_full_interval_idx


def Define_query_error_constraints(query_set, query_to_full_interval_idx, column_interval_number):
    """Each query has a constraint that the sum of the selected X should be equal to the cardinality of the query. This function first get the index of the selected X (x_index) for each query, then put it into the constraints list."""
    query_error_list = []
    for k, v in query_to_full_interval_idx.items():
        card_true = query_set[k][-1]
        x_ind = np.array([x for x in itertools.product(*v)])  # , dtype=np.uint16)
        x_index = np.ravel_multi_index(x_ind.T, column_interval_number)

        def query_cardinality_error(x, card_true=card_true, value=x_index):
            return x[value].sum() - card_true

        query_error_list.append(query_cardinality_error)
    return query_error_list


def x0(total_x, n_row):
    # There are (total_x) x variables, each x is initialized by average: n_row / total_x
    return np.ones(total_x) * n_row / total_x


def bounds(total_x, n_row):
    # each x should be in range [0, n_row]
    return np.array([[0, n_row]] * total_x)


def constraints(n_row):
    # constraints is always limited to zero when using SLSQP
    # (sum of x - n_row) should always equal to zero
    return [{"type": "eq", "fun": lambda x: n_row - x.sum()}]


def fun(query_error_list, n_row):
    def error(x):
        # divided by n_row or (n_row)**2 to limit the error to a small range near zero, to avoid the overflow of the optimization
        return sum([query_err(x) ** 2 for query_err in query_error_list]) / (n_row)  # **2

    return error


def randomized_rouding(x):
    # with probability to shift to interval left or right points
    int_x = copy.deepcopy(x)
    for i in range(len(x)):
        xi = x[i]
        floor = np.floor(xi)
        ceil = np.ceil(xi)
        if not floor == ceil:
            int_x[i] = np.random.choice([floor, ceil], p=[xi - floor, ceil - xi])
    return int_x


def random_sample(left, right, m, step=1):
    """
    Args:
        left: column_interval[j][start]
        right: column_interval[j][start+1]
        m: int_x[i]
    """
    interval = np.arange(left, right, step)
    samples = np.random.choice(interval, size=m, replace=True)
    return samples


def generate_table_data(column_interval, int_x, n_column, column_interval_number):
    """Generate table data based on z3 model solution."""
    Table_Generated = np.empty((0, n_column), dtype=np.float32)
    column_to_x = [list(range(i)) for i in column_interval_number]
    all_x = np.array([x for x in itertools.product(*column_to_x)], dtype=np.uint16)
    # all_x.shape = (total_x, n_column), total_x == len(int_x)
    for i in range(len(int_x)):
        if int_x[i] < 1:
            continue
        try:
            # Use random_sample to generate values for each column
            subtable = np.array(
                [
                    [
                        random_sample(
                            left=column_interval[j][all_x[i][j]],
                            right=column_interval[j][all_x[i][j] + 1],
                            m=1,  # Generate one value per cell
                        )[
                            0
                        ]  # Extract the single sample value
                        for j in range(n_column)
                    ]
                    for _ in range(int_x[i])  # Repeat for the number of rows
                ],
                dtype=np.float32,
            )
        except:
            vals = [column_interval[j][all_x[i][j]] for j in range(n_column)]
            subtable = np.tile(vals, (int_x[i], 1))
        Table_Generated = np.concatenate((Table_Generated, subtable), axis=0)
    return Table_Generated


# wine, query 11, (1,3) is good
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="1-input", help="model type")
parser.add_argument("--dataset", type=str, default="census-3", help="Dataset.")
parser.add_argument("--query-size", type=int, default=50, help="query size")
parser.add_argument("--min-conditions", type=int, default=1, help="min num of query conditions")
parser.add_argument("--max-conditions", type=int, default=2, help="max num of query conditions")


try:
    args = parser.parse_args()
except:
    # args = parser.parse_args([])
    args, unknown = parser.parse_known_args()


ModelName = "PGM"
FilePath = (
    f"{args.model}_{args.dataset}_{args.query_size}_({args.min_conditions}_{args.max_conditions})"
)
resultsPath = f"results/{ModelName}/{FilePath}"
make_directory(resultsPath)


print("\nBegin Loading Data ...")
table, original_table_columns, sorted_table_columns, max_decimal_places = load_and_process_dataset(
    args.dataset, resultsPath
)
table_size = table.shape
n_row, n_column = table_size
print(f"{args.dataset}.csv")
print(f"Table shape: {table_size}")
print("Done.\n")


print("Begin Generating Queries ...")
rng = np.random.RandomState(42)
query_set = [generate_random_query(table, args, rng) for _ in tqdm(range(args.query_size))]
print("Done.\n")


print("Begin Intervalization ...")
column_interval = column_intervalization(query_set, table_size, args)

for k, v in column_interval.items():
    if not v:
        column_interval[k] = [0]

column_interval_number = count_unique_vals_num(column_interval)
total_x = np.product(column_interval_number)
print(f"{column_interval_number=}")
print("Done.\n")


print("\nBegin Building LPALG (PGM) Model ...")
query_to_interval_idx = Assign_query_to_interval_idx(
    query_set, n_column, column_interval, column_interval_number
)
# _reveal_query_to_interval_idx(query_to_interval_idx, column_interval)
query_to_full_interval_idx = Fill_query_to_interval_idx(
    query_to_interval_idx, column_interval_number
)
query_error_list = Define_query_error_constraints(
    query_set, query_to_full_interval_idx, column_interval_number
)
print("Done.\n")


print(f"Begin Solving LP problem with total param = {total_x} ...")
tic = time.time()
res = optimize.minimize(
    fun(query_error_list, n_row),
    x0(total_x, n_row),
    method="SLSQP",
    constraints=constraints(n_row),
    bounds=bounds(total_x, n_row),
    # tol=1e-15,
    # options={'maxiter': 1e10},
    # options={'maxiter': 1}, # check the algorithm works
)
toc = time.time()
print("Find a solution successfully: ", res.success)
time_count(tic, toc)
print("\n Optimize.minimize Solver Status: \n", res)
# int_x = randomized_rouding(res.x).astype(int)  # original paper's rouding menthod has worse performance
int_x = np.round(res.x).astype(int)  # simple rounding has better performance
print(f"\n Integer X: ( length = {len(int_x)} )\n", int_x)


print("\nBegin Generating Data ...")
Table_Generated = generate_table_data(column_interval, int_x, n_column, column_interval_number)
# Table_Generated = np.array(Table_Generated)
# print(Table_Generated)
print("Done.\n")


print("Summary of Q-error:")
print(args)
df = calculate_Q_error(Table_Generated, query_set)
df.to_csv(f"{resultsPath}/Q_error.csv", index=True, header=False)
print(df)
print(f"\nOriginal  table shape : {table_size}")
print(f"Generated table shape : {Table_Generated.shape}\n")


print("Begin storing Data in file...")
recovered_Table_Generated = recover_table_as_original(
    Table_Generated, original_table_columns, sorted_table_columns, max_decimal_places
)
recovered_Table_Generated.to_csv(f"{resultsPath}/generated_table.csv", index=False, header=False)
print("Done.\n")

plot_3d_subplots(table, Table_Generated, f"plot3d_{args.query_size}.png")

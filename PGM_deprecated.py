# This is the implementation of LPALG(PGM) algorithm for query optimization.
# "Data generation using declarative constraints"

import argparse
import itertools
import os
import time
from copy import deepcopy

from scipy import optimize

from dataset import *
from utils import *

# def column_intervalization(table_size, query_set):
#     # Traverse all queries to apply the intervalization skill for each column
#     n_column = table_size[1]
#     column_interval = {}
#     for i in range(n_column):
#         column_interval[i] = set([sys.maxsize])  # use set([0, sys.maxsize]) to adapt '>' and '<'.
#     for query in query_set:
#         col_idxs = query[1]
#         vals = query[3]
#         for i in range(len(col_idxs)):
#             column_interval[col_idxs[i]].add(vals[i][0])
#     for k, v in column_interval.items():
#         if not v:
#             column_interval[k] = [0]
#         else:
#             column_interval[k] = sorted(list(v))
#     return column_interval


def Assign_column_variable(column_interval):
    """Generate sequential variable index for each column interval.

    Returns:
    column_to_variable: dict, key is column index, value is a list of variable index, from 0 to total_intervals - 1, where total_intervals = sum(column_interval_number)

    column_to_variable = {
        0: [0],
        1: [1, 2, 3, 4],
        2: [5, 6, 7, 8],
        3: [9],
        ...,
    }
    """
    column_to_variable = {}
    total_intervals = 0
    for k, v in column_interval.items():
        count = len(v)
        column_to_variable[k] = [total_intervals + i for i in range(count)]
        total_intervals += count
    return column_to_variable


def Assign_variable_interval(column_interval, column_to_variable):
    """Generate the mapping from variable index to interval value (left point).
    Returns:
    variable_to_interval: dict, key is variable index, value is the left point of the interval. variable index range from 0 to total_intervals - 1, where total_intervals = sum(column_interval_number)

    variable_to_interval = {
        0: 99360,
        1: 110,
        2: 280,
        3: 660,
        4: 775,
        5: 960,
        ...,
    }
    """
    variable_to_interval = {}
    for col, variable in column_to_variable.items():
        for i in range(len(variable)):
            variable_to_interval[variable[i]] = column_interval[col][i]
    return variable_to_interval


def Find_variables_by_op(column_to_variable, column, index, op):
    """Find all matching variables in the column based on query interval and the operator. Apply to both "1-input" and "2-input" model types"""
    column_interval_idx = np.array(column_to_variable[column])
    if op == "<=":
        return list(column_interval_idx[: index + 1])
    elif op == "<":
        return list(column_interval_idx[:index])
    elif op == ">":
        return list(column_interval_idx[index + 1 :])
    elif op == ">=":
        return list(column_interval_idx[index:])
    elif op == "=":
        return [column_interval_idx[index]]
    else:
        raise ValueError("Invalid operator")


def Assign_query_to_variable(query_set, column_interval, column_to_variable):
    """
    Generate the mapping from query index to variable index.

    Returns:
    query_to_variable: dict, key is query index, value is a list of variable index. Each variable index represents a column interval.

    query_to_variable = {
        0: [[16, 17]],
        1: [[21], [25]],
        2: [[1], [25]],
        3: [[5]],
        4: [[15], [16, 17, 18, 19, 20]],
        5: [[5, 6, 7, 8], [10]],
        ...,
    }
    """
    query_to_variable = {}
    for i in range(len(query_set)):
        query_to_variable[i] = []
        idxs, ops, vals, _ = query_set[i]
        for j in range(len(idxs)):
            col = idxs[j]
            index = column_interval[col].index(vals[j])
            query_to_variable[i].append(
                Find_variables_by_op(column_to_variable, col, index, ops[j])
            )
    return query_to_variable


def Assign_query_to_x_index(query_set, query_to_variable, column_to_variable):
    # Traverse all queries to find their corresponding x index
    query_to_x_index = {i: [[] for i in range(n_column)] for i in range(len(query_set))}
    for i in range(len(query_set)):
        idxs = query_set[i][0]
        for j in range(len(idxs)):
            col = idxs[j]
            variable = query_to_variable[i][j]
            for k in variable:
                x_index = column_to_variable[col].index(k)
                query_to_x_index[i][col].append(x_index)
    return query_to_x_index


def transfer_x_index(query_to_x_index, column_interval_number):
    # Transfer all empty x-index-list to all-indexes-list corresponding to the column
    x_index = {}
    for k, v in query_to_x_index.items():
        x_index[k] = []
        for i in range(len(v)):
            if v[i] == []:
                x_index[k].append([j for j in range(column_interval_number[i])])
            else:
                x_index[k].append(v[i])
    return x_index


def x0():
    # use average value to initialize the x: n_row / total_x
    return np.ones(total_x) * n_row / total_x


def bounds():
    # each x should be in [0, n_row]
    return np.array([[0, n_row]] * total_x)


def constraints():
    # constraints is always limited to zero when using SLSQP
    # sum of all x should always equal to n_row
    return [{"type": "eq", "fun": lambda x: n_row - x.sum()}]


def query_constraints(query_set, x_index, column_interval_number):
    query_constraints_list = []
    # Find the corresponding x (then sum) for each query
    # because x may be a array with multiple dimensions(such as 10), build and store such big matrix is not feasible
    # so we use a 1D array to represent the x, and use the following method to find the corresponding x index
    # To be simple: 5D array [2, 3, 4, 5, 6] -> 1D array [x_ind @ find]
    find = np.array(
        [np.product(column_interval_number[i:]) for i in range(1, len(column_interval_number))]
        + [1]
    )
    for key, values in x_index.items():
        sel = query_set[key][-1] / n_row
        x_ind = np.array([x for x in itertools.product(*values)])  # , dtype=np.uint16)
        result = x_ind @ find

        def value_constraints(x, sel=sel, value=result):
            # the cardinality-error of a query
            return x[value].sum() - sel * n_row

        query_constraints_list.append(value_constraints)
    return query_constraints_list


def fun():
    def error(x):
        # divided by n_row or (n_row)**2 to limit the error to a small range
        return sum([constraint(x) ** 2 for constraint in query_constraints_list]) / (n_row)  # **2

    return error


def randomized_rouding(x):
    # with probability to shift to interval left or right points
    int_x = deepcopy(x)
    for i in range(len(x)):
        xi = x[i]
        floor = np.floor(xi)
        ceil = np.ceil(xi)
        if not floor == ceil:
            int_x[i] = np.random.choice([floor, ceil], p=[xi - floor, ceil - xi])
    return int_x


def generate_table_data(column_interval, int_x, column_interval_number):
    df = pd.DataFrame(
        columns=[f"col_{i}" for i in range(n_column)], index=[i for i in range(int_x.sum())]
    )

    column_to_x = [list(range(i)) for i in column_interval_number]
    all_x = np.array([x for x in itertools.product(*column_to_x)], dtype=np.uint16)

    count = 0
    for i in range(len(int_x)):
        # total_x == len(int_x), n_column == all_x.shape[1]
        if int_x[i] < 1:
            continue
        df.iloc[count : count + int_x[i], :] = [
            column_interval[j][all_x[i][j]] for j in range(n_column)
        ]
        count += int_x[i]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="1-input", help="model type")
    parser.add_argument("--dataset", type=str, default="test-3", help="Dataset.")
    parser.add_argument("--query-size", type=int, default=10, help="query size")
    parser.add_argument("--min-conditions", type=int, default=1, help="min num of query conditions")
    parser.add_argument("--max-conditions", type=int, default=2, help="max num of query conditions")

    try:
        args = parser.parse_args()
    except:
        # args = parser.parse_args([])
        args, unknown = parser.parse_known_args()

    FilePath = f"{args.model}_{args.dataset}_{args.query_size}_({args.min_conditions}_{args.max_conditions})"

    def make_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    resultsPath = f"results/PGM/{FilePath}"
    make_directory(resultsPath)

    print("\nBegin Loading Data ...")
    table, original_table_columns, sorted_table_columns, max_decimal_places = (
        load_and_process_dataset(args.dataset, resultsPath)
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
    column_interval_number = count_unique_vals_num(column_interval)
    total_x = np.product(column_interval_number)
    print(f"{column_interval_number=}")
    print("Done.\n")

    print("\nBegin Building LPALG (PGM) Model ...")
    column_to_variable = Assign_column_variable(column_interval)

    variable_to_interval = Assign_variable_interval(column_interval, column_to_variable)

    query_to_variable = Assign_query_to_variable(query_set, column_interval, column_to_variable)

    query_to_x_index = Assign_query_to_x_index(query_set, query_to_variable, column_to_variable)

    x_index = transfer_x_index(query_to_x_index, column_interval_number)

    query_constraints_list = query_constraints(query_set, x_index, column_interval_number)
    print("Done.\n")

    print(f"Begin Solving LP problem with total param = {total_x} ...")
    tic = time.time()
    res = optimize.minimize(
        fun(),
        x0(),
        method="SLSQP",
        constraints=constraints(),
        bounds=bounds(),
        # tol=1e-15,
        # options={'maxiter': 1e10},
        # options={'maxiter': 1}, # only for checking the algorithm works
    )
    toc = time.time()
    print("\n Optimize.minimize Solver Status: \n", res)

    print("\nBegin Generating Data ...")
    # int_x = randomized_rouding(res.x).astype(int)
    int_x = np.round(res.x).astype(int)
    print(f"\n Integer X: ( length = {len(int_x)} )\n", int_x)

    Table_Generated = generate_table_data(column_interval, int_x, column_interval_number)
    Table_Generated = np.array(Table_Generated)
    # print(Table_Generated)
    print("Done.\n")

    print("Summary of Q-error:")
    print(args)
    time_count(tic, toc)
    df = print_Q_error(Table_Generated, query_set)
    df.to_csv(f"{resultsPath}/Q_error.csv", index=True, header=False)
    print(df)
    print(f"\n Original table shape : {table_size}")
    print(f"Generated table shape : {Table_Generated.shape}")

    print("Begin Recovering Data ...")
    recovered_Table_Generated = recover_table_as_original(
        Table_Generated, original_table_columns, sorted_table_columns, max_decimal_places
    )
    recovered_Table_Generated.to_csv(
        f"{resultsPath}/generated_table.csv", index=False, header=False
    )
    print("Done.\n")

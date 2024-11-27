import argparse
import itertools
import time

from z3 import *

from dataset import *
from models import *
from preprocessing import *
from utils import *


def make_unique(query_set):
    """Make the query set unique, remove duplicated queries."""
    seen = set()
    unique_query_set = []
    for row in query_set:
        hashable_row = tuple(tuple(item) if isinstance(item, np.ndarray) else item for item in row)
        if hashable_row not in seen:
            seen.add(hashable_row)
            unique_query_set.append(row)
    return unique_query_set


def Fill_query_to_interval_idx(query_to_interval_idx, column_interval_number):
    """Fill the empty list in query_to_interval_idx with all index range for that column."""
    query_to_full_interval_idx = copy.deepcopy(query_to_interval_idx)
    for k, v in query_to_full_interval_idx.items():
        for i in range(len(v)):
            if not v[i]:
                query_to_full_interval_idx[k][i] = list(range(column_interval_number[i]))
    return query_to_full_interval_idx


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


def define_solver(query_set, query_to_full_interval_idx, column_interval_number, penalty_weight=10):
    solver = Optimize()
    # Initialize an array of Z3 variables for each possible interval index
    total_x = np.product(column_interval_number)
    X = [Int(f"x_{i}") for i in range(total_x)]
    bounds_constraints = [And(xi >= 0, xi <= n_row) for xi in X]
    solver.add(bounds_constraints)

    query_constraints = []
    for k, v in query_to_full_interval_idx.items():
        card_true = query_set[k][-1]  # Get the true cardinality for this query
        # Flatten the multi-column intervals into a list of indices for the Z3 array
        x_ind = np.array([x for x in itertools.product(*v)])  # , dtype=np.uint16)
        x_index = np.ravel_multi_index(x_ind.T, column_interval_number)
        # Define the constraint that the sum should equal the cardinality
        query_constraint = Sum([X[i] for i in x_index]) == card_true
        query_constraints.append(query_constraint)
    solver.add(query_constraints)

    # Add the total constraint as a soft constraint with a penalty weight
    total_constraint_error = Abs(Sum(X) - n_row)
    # Soften only the total constraint
    solver.add_soft(total_constraint_error == 0, weight=penalty_weight)
    return X, solver


def generate_table_data(column_interval, int_x, n_column, column_interval_number):
    """Generate table data based on z3 model solution."""
    Table_Generated = np.empty((0, n_column), dtype=np.float32)
    column_to_x = [list(range(i)) for i in column_interval_number]
    all_x = np.array([x for x in itertools.product(*column_to_x)], dtype=np.uint16)
    # all_x.shape = (total_x, n_column), total_x == len(int_x)
    for i in range(len(int_x)):
        if int_x[i] < 1:
            continue
        vals = [column_interval[j][all_x[i][j]] for j in range(n_column)]
        subtable = np.tile(vals, (int_x[i], 1))
        Table_Generated = np.concatenate((Table_Generated, subtable), axis=0)
    return Table_Generated


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="1-input", help="model type")
parser.add_argument("--dataset", type=str, default="census-2", help="Dataset.")
parser.add_argument("--query-size", type=int, default=10, help="query size")
parser.add_argument("--min-conditions", type=int, default=1, help="min num of query conditions")
parser.add_argument("--max-conditions", type=int, default=2, help="max num of query conditions")


try:
    args = parser.parse_args()
except:
    # args = parser.parse_args([])
    args, unknown = parser.parse_known_args()


ModelName = "SMT_PGM"
FilePath = (
    f"{args.dataset}_{args.query_size}_{args.min_conditions}_{args.max_conditions}_{args.model}"
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
print(f"query_to_interval_idx={query_to_interval_idx}")
# _reveal_query_to_interval_idx(query_to_interval_idx, column_interval)
query_to_full_interval_idx = Fill_query_to_interval_idx(
    query_to_interval_idx, column_interval_number
)
print(f"query_to_full_interval_idx={query_to_full_interval_idx}")


X, solver = define_solver(query_set, query_to_full_interval_idx, column_interval_number)


tic = time.time()
if solver.check() == sat:
    print("Satisfiable solution founded")
    toc = time.time()
    print("\nBegin Generating Data ...")
    model = solver.model()
    print(model)
    int_x = [
        model[Int(f"x_{i}")].as_long() if model[Int(f"x_{i}")] is not None else 0
        for i in range(len(model))
    ]
    print(f"\n Integer X: ( length = {len(int_x)} )\n", int_x)
    Table_Generated = generate_table_data(column_interval, int_x, n_column, column_interval_number)
    print("Done.\n")
else:
    print("No solution founded.")
    toc = time.time()
time_count(tic, toc)

print("\nSummary of Q-error:")
print(args)
df = calculate_Q_error(Table_Generated, query_set)
df.to_csv(f"{resultsPath}/Q_error.csv", index=True, header=False)
print(df)
print(f"\n Original table shape : {table_size}")
print(f"Generated table shape : {Table_Generated.shape}\n")


output_dir = "results/"
plt.figure()
plt.scatter(table[:, 0], table[:, 1], label="Table")
plt.scatter(Table_Generated[:, 0], Table_Generated[:, 1], label="Table_Generated")
plt.legend()
plt.title("Scatter Plot of Table and Table_Generated")
plt.savefig(f"{output_dir}/scatter_plot.png")
plt.close()
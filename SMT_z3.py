import argparse
import random
import time
import itertools

from z3 import *

from dataset import *
from models import *
from preprocessing import *
from utils import *
from plot_util import *
from ce_util import *

def make_unique(query_set):
    seen = set()
    unique_query_set = []
    for row in query_set:
        hashable_row = tuple(tuple(item) if isinstance(item, np.ndarray) else item for item in row)
        if hashable_row not in seen:
            seen.add(hashable_row)
            unique_query_set.append(row)
    return unique_query_set

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
            subtable = np.array([
                [
                    random_sample(
                        left=column_interval[j][all_x[i][j]],
                        right=column_interval[j][all_x[i][j]+1],
                        m=1,  # Generate one value per cell
                    )[0]  # Extract the single sample value
                    for j in range(n_column)
                ]
                for _ in range(int_x[i])  # Repeat for the number of rows
            ], dtype=np.float32)
        except:
            vals = [column_interval[j][all_x[i][j]] for j in range(n_column)]
            subtable = np.tile(vals, (int_x[i], 1))
        Table_Generated = np.concatenate((Table_Generated, subtable), axis=0)
    return Table_Generated


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

def define_solver(query_set, query_to_full_interval_idx, column_interval_number, penalty_weight=1, query_penalty_weight=1):
    solver = Optimize()
    
    # Initialize an array of Z3 variables for each possible interval index
    total_x = np.product(column_interval_number)
    X = [Int(f"x_{i}") for i in range(total_x)]
    
    # Add bounds constraints
    bounds_constraints = [And(xi >= 0, xi <= n_row) for xi in X]
    solver.add(bounds_constraints)
        
    # Add soft constraints for each query
    for k, v in query_to_full_interval_idx.items():
        card_true = query_set[k][-1]  # Get the true cardinality for this query
        
        # Flatten the multi-column intervals into a list of indices for the Z3 array
        x_ind = np.array([x for x in itertools.product(*v)])
        x_index = np.ravel_multi_index(x_ind.T, column_interval_number)
        
        # Define the constraint that the sum should approximately equal the cardinality
        query_cardinality_error = Abs(Sum([X[i] for i in x_index]) - card_true)
        solver.add_soft(query_cardinality_error == 0, weight=query_penalty_weight)  # Soften the query constraint
    
    # Add the total constraint as a soft constraint with a penalty weight
    total_constraint_error = Abs(Sum(X) - n_row)
    solver.add_soft(total_constraint_error == 0, weight=penalty_weight)  # Soften the total constraint
    
    return X, solver


def calculate_Q_error_smt(Table_Generated, table):
    gen_rows = Table_Generated.shape[0]
    true_rows = table.shape[0]
    return max(gen_rows, true_rows) / min(gen_rows, true_rows)


def export_to_csv(table, filename):
    try:
        np.savetxt(filename, table, delimiter=",", fmt="%s")
        print(f"Table successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving array to CSV: {e}")

def count_matching_rows(arr1, arr2):
    # Convert both arrays to the same type (integer) for comparison
    arr1_int = arr1.astype(int)
    arr2_int = arr2.astype(int)
    
    # Use a set for efficient row matching
    arr2_set = {tuple(row) for row in arr2_int}
    match_count = sum(1 for row in arr1_int if tuple(row) in arr2_set)
    
    return match_count


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="1-input", help="model type")
parser.add_argument("--dataset", type=str, default="census-3", help="Dataset.")
parser.add_argument("--query-size", type=int, default=100, help="query size")
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
# print(f"query_to_interval_idx={query_to_interval_idx}")
# _reveal_query_to_interval_idx(query_to_interval_idx, column_interval)
query_to_full_interval_idx = Fill_query_to_interval_idx(
    query_to_interval_idx, column_interval_number
)
# print(f"query_to_full_interval_idx={query_to_full_interval_idx}")

tic = time.time()
X, solver = define_solver(
    query_set, query_to_full_interval_idx, column_interval_number
)

if solver.check() == sat:
    print("Satisfiable solution founded")
    toc = time.time()
    print("\nBegin Generating Data ...")
    model = solver.model()
    # print(model)
    int_x = [model[Int(f"x_{i}")].as_long() if model[Int(f"x_{i}")] is not None else 0 for i in range(len(model))]
    print(f"\n Integer X: ( length = {len(int_x)} )\n")
    Table_Generated = generate_table_data(column_interval, int_x, n_column, column_interval_number)
    print("Done.\n")
else:
    print("No solution founded.")
    toc = time.time()
time_count(tic, toc)

print("\nSummary of Q-error:")
print(args)
Table_Generated = Table_Generated.astype(int)
q_error = calculate_Q_error_smt(Table_Generated, table)
print(f"table size-based {q_error=}")
print(f"\n Original table shape : {table_size}")
print(f"Generated table shape : {Table_Generated.shape}\n")
print(args)
df = calculate_Q_error(Table_Generated, query_set)
df.to_csv(f"{resultsPath}/Q_error.csv", index=True, header=False)
print(df)
print(f"\nOriginal  table shape : {table_size}")
print(f"Generated table shape : {Table_Generated.shape}\n")
print(f"# of matched rows : {count_matching_rows(table, Table_Generated)}\n")
print(f"CE: {AR_ComputeCE(table, Table_Generated)}")

plot_3d_subplots(table, Table_Generated, f'plot3d_{args.query_size}.png')
# export_to_csv(table, "results/ground-truth.csv")
# export_to_csv(Table_Generated, "results/generated.csv")
# plot_3d(table, Table_Generated)
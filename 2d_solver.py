import argparse
import time

from z3 import *

from dataset import *
from models import *
from preprocessing import *
from utils import *


def make_unique(query_set):
    seen = set()
    unique_query_set = []
    for row in query_set:
        hashable_row = tuple(tuple(item) if isinstance(item, np.ndarray) else item for item in row)
        if hashable_row not in seen:
            seen.add(hashable_row)
            unique_query_set.append(row)
    return unique_query_set


def convert_queries_to_constraints(num_rows, num_columns, queries, max_col_val):
    solver = Solver()

    # Create a 2D list of z3 integer variables representing the table
    db = [[Int(f"cell_{r}_{c}") for c in range(num_columns)] for r in range(num_rows)]

    # Add domain constraints (values should be in a reasonable range)
    for row in db:
        for i in range(len(row)):
            cell = row[i]
            solver.add(cell >= 0, cell <= max_col_val[i])

    # Convert each query into a z3 constraint and add it to the solver
    for idxs, ops, vals, card in queries:
        # for r in range(num_rows):
        #     constraints = []
        #     for i, idx in enumerate(idxs):
        #         if ops[i] == "<=":
        #             constraints.append(db[r][idx] <= vals[i])
        #         elif ops[i] == "<":
        #             constraints.append(db[r][idx] < vals[i])
        #         elif ops[i] == ">":
        #             constraints.append(db[r][idx] > vals[i])
        #         elif ops[i] == ">=":
        #             constraints.append(db[r][idx] >= vals[i])
        #         elif ops[i] == "=":
        #             constraints.append(db[r][idx] == vals[i])

        #     # Add a constraint ensuring that a row satisfies all conditions in the query
        #     solver.add(Implies(And(*constraints), True))

        # Add cardinality constraint (number of rows satisfying the condition)
        count = Sum(
            [
                If(
                    And(
                        *[
                            (
                                db[r][idx] <= vals[i]
                                if ops[i] == "<="
                                else (
                                    db[r][idx] < vals[i]
                                    if ops[i] == "<"
                                    else (
                                        db[r][idx] > vals[i]
                                        if ops[i] == ">"
                                        else (
                                            db[r][idx] >= vals[i]
                                            if ops[i] == ">="
                                            else db[r][idx] == vals[i]
                                        )
                                    )
                                )
                            )
                            for i, idx in enumerate(idxs)
                        ]
                    ),
                    1,
                    0,
                )
                for r in range(num_rows)
            ]
        )
        solver.add(count == card)

    return solver, db


def generate_table(solver, db, num_rows, num_columns):
    model = solver.model()
    Table_Generated = np.array(
        [
            [
                (model[db[r][c]].as_long() if model[db[r][c]] is not None else 0)
                for c in range(num_columns)
            ]
            for r in range(num_rows)
        ]
    )
    return Table_Generated


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="1-input", help="model type")
parser.add_argument("--dataset", type=str, default="test-2", help="Dataset.")
parser.add_argument("--query-size", type=int, default=10, help="query size")
parser.add_argument("--min-conditions", type=int, default=1, help="min num of query conditions")
parser.add_argument("--max-conditions", type=int, default=2, help="max num of query conditions")

try:
    args = parser.parse_args()
except:
    # args = parser.parse_args([])
    args, unknown = parser.parse_known_args()


ModelName = "SMT"
FilePath = (
    f"{args.dataset}_{args.query_size}_{args.min_conditions}_{args.max_conditions}_{args.model}"
)
resultsPath = f"results/{ModelName}/{FilePath}"
make_directory(resultsPath)


print("\nBegin Loading Data ...")
table, original_table_columns, sorted_table_columns, max_decimal_places = load_and_process_dataset(
    args.dataset, resultsPath
)
max_col_val = np.max(table, axis=0)
table_size = table.shape
num_rows = table_size[0]
num_columns = table_size[1]
print(f"{args.dataset}.csv,    shape: {table_size}")
print("Done.\n")


print("Begin Generating Queries Set ...")
rng = np.random.RandomState(42)
query_set = [generate_random_query(table, args, rng) for _ in tqdm(range(args.query_size))]
query_set = make_unique(query_set)
print("Done.\n")


print("Begin Solving SMT ...")
tic = time.time()
solver, db = convert_queries_to_constraints(num_rows, num_columns, query_set, max_col_val)
if solver.check() == sat:
    print("Satisfiable solution founded")
    toc = time.time()
    Table_Generated = generate_table(solver, db, num_rows, num_columns)
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


plt.scatter(table[:, 0], table[:, 1])
plt.scatter(Table_Generated[:, 0], Table_Generated[:, 1])

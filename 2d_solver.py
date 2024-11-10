import argparse
import time

from dataset import *
from models import *
from preprocessing import *
from utils import *

from z3 import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="1-input", help="model type")
parser.add_argument("--dataset", type=str, default="test-2", help="Dataset.")
parser.add_argument("--query-size", type=int, default=200, help="query size")
parser.add_argument("--min-conditions", type=int, default=1, help="min num of query conditions")
parser.add_argument("--max-conditions", type=int, default=2, help="max num of query conditions")

try: 
    args = parser.parse_args()
except:
    # args = parser.parse_args([])
    args, unknown = parser.parse_known_args()
    
FilePath = (
    f"{args.dataset}_{args.query_size}_{args.min_conditions}_{args.max_conditions}_{args.model}"
)

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_queries_to_constraints(num_rows, num_columns, queries):
    # Create a 2D list of z3 integer variables representing the synthetic database
    db = [[Int(f'cell_{r}_{c}') for c in range(num_columns)] for r in range(num_rows)]
    
    # Create a solver instance
    solver = Solver()

    # Add domain constraints (e.g., values should be in a reasonable range)
    for row in db:
        for cell in row:
            solver.add(cell >= 0, cell <= 1000)  # Example: cell values between 0 and 1000

    # Convert each query into a z3 constraint and add it to the solver
    for idxs, ops, vals, card in queries:
        for r in range(num_rows):
            constraints = []
            for i, idx in enumerate(idxs):
                if ops[i] == '<=':
                    constraints.append(db[r][idx] <= vals[i])
                elif ops[i] == '<':
                    constraints.append(db[r][idx] < vals[i])
                elif ops[i] == '>':
                    constraints.append(db[r][idx] > vals[i])
                elif ops[i] == '>=':
                    constraints.append(db[r][idx] >= vals[i])
                elif ops[i] == '=':
                    constraints.append(db[r][idx] == vals[i])
            
            # Add a constraint ensuring that a row satisfies all conditions in the query
            solver.add(Implies(And(*constraints), True))

        # Add cardinality constraint (number of rows satisfying the condition)
        count = Sum([If(And(*[db[r][idx] <= vals[i] if ops[i] == '<=' else
                             db[r][idx] < vals[i] if ops[i] == '<' else
                             db[r][idx] > vals[i] if ops[i] == '>' else
                             db[r][idx] >= vals[i] if ops[i] == '>=' else
                             db[r][idx] == vals[i]
                             for i, idx in enumerate(idxs)]), 1, 0)
                     for r in range(num_rows)])
        solver.add(count == card)
    
    return solver, db

resultsPath = f"results/{FilePath}"
make_directory(resultsPath)

print("\nBegin Loading Data ...")
table, original_table_columns, sorted_table_columns, max_decimal_places = load_and_process_dataset(
    args.dataset, resultsPath
)
table_size = table.shape
print(f"{args.dataset}.csv,    shape: {table_size}")
print("Done.\n")


print("Begin Generating Queries Set ...")
rng = np.random.RandomState(42)
query_set = [generate_random_query(table, args, rng) for _ in tqdm(range(args.query_size))]
# print("print query set: ")
# for item in query_set:
#     print(f"query: {item}")
print("Done.\n")

num_rows = table_size[0]
num_columns = table_size[1]

# Convert queries to constraints and create the solver
solver, satisfied_table = convert_queries_to_constraints(num_rows, num_columns, query_set)

start_time = time.time()

# Check if the solution is satisfiable
if solver.check() == sat:
    end_time = time.time()
    model = solver.model()
    print("Satisfiable solution found:")
    for r in range(num_rows):
        row_values = [model[satisfied_table[r][c]] if model[satisfied_table[r][c]] is not None else '?' for c in range(num_columns)]
        print(f"{r}: {row_values}")
else:
    end_time = time.time()
    print("No solution found.")

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
import subprocess
import time

file_name = "PGM"
param_list = [{"model": "1-input", "dataset": "census", "query_size": 10 * i} for i in range(1, 11)]
output_file = f"results/Multiple_Run_{file_name}.txt"

with open(output_file, "a") as f:
    f.write(f"\nModel: {file_name}\n")
    for i, params in enumerate(param_list):
        args = [
            "python",
            file_name + ".py",
            "--model",
            params["model"],
            "--dataset",
            params["dataset"],
            "--query-size",
            str(params["query_size"]),
        ]
        start_time = time.time()
        f.write(f"\nRound {i}: {params}\n")
        subprocess.run(args, stdout=f, stderr=f)
        end_time = time.time()
        run_time = end_time - start_time
        f.write(f"Round {i} Time: {run_time:.2f} second\n\n")

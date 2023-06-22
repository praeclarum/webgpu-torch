import torch
from itertools import product
import time
import json
import sys


machine_id = sys.argv[1]

has_cuda = torch.cuda.is_available()

# Get machine CPU info
import cpuinfo
cpu_info = cpuinfo.get_cpu_info()
device_name = torch.cuda.get_device_name(0) if has_cuda else cpu_info["brand_raw"]

def get_input_permutations(benchmark_inputs):
    return list(product(*[inp["values"] for inp in benchmark_inputs]))


def run_unary_benchmark(benchmark, inputs):
    shape = inputs[0]
    operation = getattr(torch, inputs[1])

    x = torch.ones(shape, dtype=torch.float32)
    # y = torch.zeros(shape, dtype=torch.float32)
    if has_cuda:
        x = x.cuda()
        # y = y.cuda()
    def run_iteration():
        start = time.perf_counter()
        y = x
        for i in range(benchmark["depth"]):
            # operation(x, out=y)
            y = operation(y)
        yar = y.cpu().tolist()
        end = time.perf_counter()
        return (end - start)*1000 / benchmark["depth"]

    for _ in range(benchmark["warmupIterations"]):
        run_iteration()

    ms = []
    for _ in range(benchmark["iterations"]):
        ms.append(run_iteration())

    return ms


def run_benchmark(benchmark, inputs):
    print(f"Running {benchmark}...")
    if benchmark["type"] == "unary":
        return run_unary_benchmark(benchmark, inputs)
    else:
        raise ValueError(f"Unknown benchmark type '{benchmark['type']}'")


def run_benchmarks():
    # Load the test JSON
    json_file = "../web/benchmarks/benchmarks.json"
    with open(json_file) as f:
        benchmarks = json.load(f)

    # Run the benchmarks
    benchmark_results = {}
    for b in benchmarks["benchmarks"]:
        input_perms = get_input_permutations(b["inputs"])

        for ip in input_perms:
            input_str = ", ".join([repr(i) for i in ip])
            benchmark_key = f"{b['name']}({input_str})"
            result = {"mean_ms": 0.0}
            try:
                times = run_benchmark(b, ip)
                mean_time = sum(times) / len(times)
                result["mean_ms"] = mean_time
            except Exception as e:
                result["error"] = e
                print(e)
            benchmark_results[benchmark_key] = result

    return benchmark_results

results = run_benchmarks()

with open("../web/benchmarks/results.json", "r") as f:
    results_obj = json.load(f)

results_obj[machine_id] = {
    "device_name": device_name,
    "results": results,
}

with open("../web/benchmarks/results.json", "w") as f:
    json.dump(results_obj, f, indent=2)

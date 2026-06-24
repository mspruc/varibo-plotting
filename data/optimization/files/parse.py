import os
import sys
from collections import defaultdict

files_dir = os.path.join(os.path.dirname(__file__), "optimizations")

time_keys = {"Encoding", "Inference", "Unpacking", "Decoding", "Conversion"}


def parse_directory(dir_path):
    accumulated_times = defaultdict(list)
    for filename in sorted(os.listdir(dir_path)):
        filepath = os.path.join(dir_path, filename)
        if not os.path.isfile(filepath):
            continue
        parsed = {}
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, sep, value = line.partition(":")
                if not sep:
                    continue
                key = key.strip()
                value = value.strip()
                if key in time_keys:
                    parsed[key] = int(value)
                elif key == "# of operators":
                    parsed["operators"] = int(value)

        if "operators" not in parsed:
            print(f"Skipping incomplete file: {filename}", file=sys.stderr)
            continue

        total = sum(parsed[k] for k in time_keys if k in parsed)
        accumulated_times[parsed["operators"]].append(total)

    return {
        ops: sum(times) / len(times)
        for ops, times in sorted(accumulated_times.items())
    }


variants = [
    ("bvae", files_dir),
    ("native", os.path.join(files_dir, "native")),
    ("cost", os.path.join(files_dir, "cost")),
]

for name, path in variants:
    result = parse_directory(path)
    print(f"# {name}")
    print("operators,avg_accumulated_time_ms")
    for ops, avg in result.items():
        print(f"{ops},{avg:.2f}")
    print()

import os
import sys
from collections import defaultdict
from statistics import median

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
        ops: median(times)
        for ops, times in sorted(accumulated_times.items())
    }


BENCHMARKS = ["tpch", "stats", "job"]

VARIANTS = [
    ("bvae", files_dir),
    ("native", os.path.join(files_dir, "native")),
    ("cost", os.path.join(files_dir, "cost")),
]


def parse_avg_total_time():
    for optimizer, base_path in VARIANTS:
        for benchmark in BENCHMARKS:
            path = os.path.join(base_path, benchmark)
            result = parse_directory(path)
            print(f"# {optimizer}_{benchmark}")
            print("Operators\tTime")
            for ops, avg in result.items():
                print(f"{ops}\t{avg:.2f}")
            print()


def parse_avg_segments_time():
    for optimizer, base_path in VARIANTS:
        for benchmark in BENCHMARKS:
            path = os.path.join(base_path, benchmark)
            segment_times = defaultdict(list)
            for filename in sorted(os.listdir(path)):
                filepath = os.path.join(path, filename)
                if not os.path.isfile(filepath):
                    continue
                with open(filepath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        key, sep, value = line.partition(":")
                        if not sep:
                            continue
                        key = key.strip()
                        if key == "# of operators":
                            continue
                        try:
                            segment_times[key].append(int(value.strip()))
                        except ValueError:
                            continue

            print(f"# {optimizer}_{benchmark}")
            print("Segment\tTime")
            for segment, times in sorted(segment_times.items()):
                print(f"{segment}\t{median(times):.2f}")
            print()



def main():
    #parse_avg_total_time()
    parse_avg_segments_time()


if __name__ == "__main__":
    main()


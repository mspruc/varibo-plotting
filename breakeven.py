"""Offline exploration cost and break-even point of VariBO relative to Native.

Workload runtimes come from each benchmark's summary.csv (Total (s) column).
Break-even (# executions) = offline time / time saved per workload execution.
Break-even (# queries)    = break-even executions * total queries of the benchmark.

Writes data/breakeven.tex.
Run with: venv/bin/python breakeven.py
"""
import pandas as pd

from explored_speedups import EXPLORED

# (name, summary.csv, total queries in the benchmark, offline exploration time in hours)
BENCHMARKS = [
    ('JOB-C', './data/job/summary.csv',   30,  38.92),
    ('STATS', './data/stats/summary.csv', 146, 19.98),
    ('TPC-H', './data/tpch/summary.csv',  30,  42.43),
]


def compute() -> pd.DataFrame:
    rows = []
    for name, path, total_queries, offline_h in BENCHMARKS:
        summary = pd.read_csv(path).set_index('Optimizer')['Total (s)']
        native_s, varibo_s = summary['Native'], summary['VariBO']
        saved_s = native_s - varibo_s
        executions = offline_h * 3600 / saved_s
        explored = EXPLORED[name]
        rows.append({
            'Benchmark': name,
            'Queries (total)': total_queries,
            'Queries (explored)': total_queries if explored is None else len(explored),
            'Offline time (h)': offline_h,
            'Native (min)': native_s / 60,
            'VariBO (min)': varibo_s / 60,
            'Time saved (min)': saved_s / 60,
            'Break-even (executions)': executions,
            'Break-even (queries)': executions * total_queries,
        })
    return pd.DataFrame(rows).set_index('Benchmark')


def to_latex(table: pd.DataFrame) -> str:
    def row(label, column, fmt, bold=False):
        cells = [fmt.format(v) for v in table[column]]
        if bold:
            cells = [rf'\textbf{{{c}}}' for c in cells]
        return f'        {label:<34} & ' + ' & '.join(cells) + r' \\'

    return '\n'.join([
        r'\begin{table}[h]',
        r'    \centering',
        r'    \small',
        r'    \caption{Offline exploration cost and break-even point relative to \textsc{Native}.',
        r'    End-to-end time is the total runtime of one workload execution. Time saved is the',
        r'    reduction in that runtime per workload execution. The break-even point is the number',
        r'    of workload executions, and the corresponding number of query executions, after',
        r'    which the saved time exceeds the offline time.}',
        r'    \label{tab:general-breakeven}',
        r'    \begin{tabular}{lrrr}',
        r'        \toprule',
        r'         & ' + ' & '.join(table.index) + r' \\',
        r'        \midrule',
        row('Queries (total)', 'Queries (total)', '{:d}'),
        row('Queries (explored)', 'Queries (explored)', '{:d}'),
        row('Offline time (h)', 'Offline time (h)', '{:.2f}'),
        r'        \midrule',
        row(r'End-to-end, \textsc{Native} (min)', 'Native (min)', '{:.2f}'),
        row(r'End-to-end, \system (min)', 'VariBO (min)', '{:.2f}'),
        row('Time saved per execution (min)', 'Time saved (min)', '{:.2f}'),
        r'        \midrule',
        row(r'Break-even (\# executions)', 'Break-even (executions)', '{:.2f}', bold=True),
        row(r'Break-even (\# queries)', 'Break-even (queries)', '{:.0f}'),
        r'        \bottomrule',
        r'    \end{tabular}',
        r'    \vspace{-0.2cm}',
        r'\end{table}',
    ]) + '\n'


if __name__ == '__main__':
    table = compute()
    with open('./data/breakeven.tex', 'w') as f:
        f.write(to_latex(table))
    print(table.round(2).T.to_string())

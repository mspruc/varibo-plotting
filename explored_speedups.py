"""Speedup over Native on explored and unexplored queries per benchmark.

Speedup = sum(Native runtime) / sum(optimizer runtime) over the queries of a subset.
Queries where Native or the optimizer has no measurement are skipped.

Writes data/explored_speedups.csv and data/explored_speedups.tex.
Run with: venv/bin/python explored_speedups.py
"""
import pandas as pd

BENCHMARKS = [
    ('JOB-C', './data/job/table.csv'),
    ('STATS', './data/stats/table.csv'),
    ('TPC-H', './data/tpch/table.csv'),
]

# None means every query of the benchmark was explored
EXPLORED = {
    'JOB-C': ['10', '11', '14', '15', '17', '2', '29', '7', '9'],
    'STATS': [
        'q109961', 'q113925678', 'q11637523617', 'q155696936', 'q16049150', 'q1633681',
        'q1771584', 'q183537163', 'q18597973', 'q2016238', 'q2034610', 'q2101105',
        'q232039659', 'q251068', 'q312148706', 'q315516', 'q354388', 'q38686327',
        'q42036', 'q4254157', 'q4582', 'q469', 'q4801', 'q490921', 'q554302',
        'q569476', 'q692609', 'q70752', 'q7491903',
    ],
    'TPC-H': None,
}

SUBSETS = ['Explored', 'Unexplored']
OPTIMIZERS = ['NativeML', 'NativeML retrained', 'Classifier', 'VariBO']
TEX_NAMES = {'VariBO': r'\system'}


def speedup(df: pd.DataFrame, optimizer: str):
    measured = df.dropna(subset=['Native', optimizer])
    if measured.empty:
        return None
    return measured['Native'].sum() / measured[optimizer].sum()


def compute() -> pd.DataFrame:
    rows = []
    tables = {name: pd.read_csv(path, dtype={'Query': str}) for name, path in BENCHMARKS}
    for subset in SUBSETS:
        for optimizer in OPTIMIZERS:
            row = {'Subset': subset, 'Optimizer': optimizer}
            for name, df in tables.items():
                explored = EXPLORED[name]
                is_explored = df['Query'].isin(explored) if explored is not None else pd.Series(True, index=df.index)
                part = df[is_explored] if subset == 'Explored' else df[~is_explored]
                row[name] = speedup(part, optimizer)
            rows.append(row)
    return pd.DataFrame(rows)


def to_latex(table: pd.DataFrame) -> str:
    names = [name for name, _ in BENCHMARKS]
    best = table.groupby('Subset')[names].transform('max')

    def cell(value, is_best):
        if pd.isna(value):
            return '--'
        return rf'$\mathbf{{{value:.2f}\times}}$' if is_best else rf'${value:.2f}\times$'

    lines = [
        r'\begin{table}[t]',
        r'	\centering',
        r'	\caption{Speedup over \textsc{Native} in total runtime of the queries seen during',
        r'		exploration (Explored) and of the remaining queries (Unexplored). Values above',
        r'		$1.0\times$ indicate a speedup over \textsc{Native}. TPC-H has no unexplored',
        r'		queries, since \system explores all of them.}',
        r'	\label{tab:speedups}',
        r'	\begin{tabular}{llccc}',
        r'		\toprule',
        r'		\textbf{Query subset} & \textbf{Optimizer} & '
        + ' & '.join(rf'\textbf{{{n}}}' for n in names) + r' \\',
        r'		\midrule',
    ]
    for i, subset in enumerate(SUBSETS):
        if i > 0:
            lines.append(r'		\cmidrule(lr){1-5}')
        part = table[table['Subset'] == subset]
        for j, (idx, row) in enumerate(part.iterrows()):
            first = rf'\multirow{{{len(part)}}}{{*}}{{{subset}}}' if j == 0 else ''
            optimizer = TEX_NAMES.get(row['Optimizer'], row['Optimizer'])
            cells = [cell(row[n], row[n] == best.loc[idx, n]) for n in names]
            lines.append(f'		{first:<27} & {optimizer:<18} & ' + ' & '.join(cells) + r' \\')
    lines += [
        r'		\bottomrule',
        r'	\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    table = compute()
    table.to_csv('./data/explored_speedups.csv', index=False, float_format='%.2f')
    with open('./data/explored_speedups.tex', 'w') as f:
        f.write(to_latex(table))
    print(table.round(3).to_string(index=False))

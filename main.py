import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np

plt.close("all")

EXPLORED_COLOR   = '#0072B2'
CLASSIFIER_COLOR = '#E69F00'
VARIBO_COLOR     = "#009E73"
NATIVE_COLOR     = "#D55E00"
NATIVEML_COLOR   = "#CC79A7"
SPARK_COLOR      = "#56B4E9"

EXPLORED_HATCHING   = '---'
CLASSIFIER_HATCHING = '\\\\\\'
VARIBO_HATCHING     = '///'
NATIVE_HATCHING     = 'xx'
NATIVEML_HATCHING   = '||'
SPARK_HATCHING      = '..'

SYSTEM_NAME = "VariBO"

# every optimizer in the comparison, in plotting order: (table.csv column, label, color, hatch)
ALL_OPTIMIZERS = [
    ('Java',               'Java',               EXPLORED_COLOR,   EXPLORED_HATCHING),
    ('Spark',              'Spark',              SPARK_COLOR,      SPARK_HATCHING),
    ('Native',             'Native',             NATIVE_COLOR,     NATIVE_HATCHING),
    ('NativeML',           'NativeML',           NATIVEML_COLOR,   NATIVEML_HATCHING),
    ('NativeML retrained', 'NativeML retrained', NATIVEML_COLOR,   '++'),
    ('Classifier',         'Classifier',         CLASSIFIER_COLOR, CLASSIFIER_HATCHING),
    (SYSTEM_NAME,          SYSTEM_NAME,          VARIBO_COLOR,     VARIBO_HATCHING),
]

def set_paper_style():
    plt.rcParams.update({
        "font.size": 9 * 1.5,            # base font
        "axes.titlesize": 10 * 2,      # title
        "axes.labelsize": 9 * 2,       # axis labels
        "xtick.labelsize": 8 * 2,
        "ytick.labelsize": 8 * 2,
        "legend.fontsize": 8 * 2,
        "figure.titlesize": 10 * 2,

        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

def transparent(color, amount=0.5):
    r, g, b = mcolors.to_rgb(color)

    r = 1 - (1 - r) * (1 - amount)
    g = 1 - (1 - g) * (1 - amount)
    b = 1 - (1 - b) * (1 - amount)

    return (r, g, b)

# shows speed up ratio over each benchmark compared to native
# we ablate the queries THAT WE DONT EXPLORE ON from the benchmark and see if the explored queries correctly reconstruct
# JOB STATS TPCH ordered lexicographically
def finetuning_capability_ablation_study():
    job_df   = pd.read_csv("./data/job/table.csv", delimiter=",")
    stats_df = pd.read_csv("./data/stats/table.csv", delimiter=",")
    tpch_df  = pd.read_csv("./data/tpch/table.csv", delimiter=",")

    def restrict_to_explored(full_df, explored_df):
        explored_queries = set(explored_df["Query"])
        benchmarked_queries = full_df["Query"]
        return full_df[benchmarked_queries.isin(explored_queries)]

    job_explored_df   = pd.read_csv("./data/job/explored.csv", delimiter='\t')
    stats_explored_df = pd.read_csv("./data/stats/explored.csv", delimiter='\t')
    tpch_explored_df  = pd.read_csv("./data/tpch/explored.csv", delimiter='\t')

    job_df   = restrict_to_explored(job_df, job_explored_df)
    stats_df = restrict_to_explored(stats_df, stats_explored_df)
    tpch_df  = restrict_to_explored(tpch_df, tpch_explored_df)

    # classifier % of native
    job_percentage_classifier   = (job_explored_df['Native'].sum() / job_df['Classifier'].sum())
    stats_percentage_classifier = (stats_explored_df['Native'].sum() / stats_df['Classifier'].sum())
    tpch_percentage_classifier  = (tpch_explored_df['Native'].sum() / tpch_df['Classifier'].sum())

    # varibo % of native
    job_percentage_varibo   = (job_explored_df['Native'].sum() / job_df[SYSTEM_NAME].sum())
    stats_percentage_varibo = (stats_explored_df['Native'].sum() / stats_df[SYSTEM_NAME].sum())
    tpch_percentage_varibo  = (tpch_explored_df['Native'].sum() / tpch_df[SYSTEM_NAME].sum())

    # nativeml % of native
    job_percentage_nativeml   = (job_explored_df['Native'].sum() / job_df['NativeML'].sum())
    stats_percentage_nativeml = (stats_explored_df['Native'].sum() / stats_df['NativeML'].sum())
    tpch_percentage_nativeml  = (tpch_explored_df['Native'].sum() / tpch_df['NativeML'].sum())

    labels = ['JOB-C', 'STATS', 'TPC-H']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))

    bars1 = ax.bar(
        x - width,
        [job_percentage_nativeml, stats_percentage_nativeml, tpch_percentage_nativeml],
        width,
        edgecolor=NATIVEML_COLOR,
        facecolor=transparent(NATIVEML_COLOR, 0.7),
        hatch=NATIVEML_HATCHING,
        linewidth=1.5,
        label='NativeML'
    )

    bars2 = ax.bar(
        x,
        [job_percentage_classifier, stats_percentage_classifier, tpch_percentage_classifier],
        width,
        edgecolor=CLASSIFIER_COLOR,
        facecolor=transparent(CLASSIFIER_COLOR, 0.7),
        hatch=CLASSIFIER_HATCHING,
        linewidth=1.5,
        label='Classifier'
    )

    bars3 = ax.bar(
        x + width,
        [job_percentage_varibo, stats_percentage_varibo, tpch_percentage_varibo],
        width,
        edgecolor=VARIBO_COLOR,
        facecolor=transparent(VARIBO_COLOR, 0.7),
        hatch=VARIBO_HATCHING,
        linewidth=1.5,
        label='VariBO'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, size=9*4)
    #ax.set_ylabel('Speedup ratio over Native')

    ax.grid(False)

    ax.set_ylim(0, 7)
    plt.yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    def label_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}x",
                ha='center',
                va='bottom'
            )

    label_bars(bars1)
    label_bars(bars2)
    label_bars(bars3)

    #ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('./data/ablation_explored.pdf')


# shows speed up ratio over each benchmark compared to native
# we ablate the explored queries from the benchmark and see if the explored queries provide value to the rest of the benchmark
# JOB STATS TPCH ordered lexicographically
def generalization_ablation_study():
    job_df   = pd.read_csv("./data/job/table.csv", delimiter=",")
    stats_df = pd.read_csv("./data/stats/table.csv", delimiter=",")

    def restrict_to_not_explored(full_df, explored_df):
        explored_queries = set(explored_df["Query"])
        return full_df[~full_df["Query"].isin(explored_queries)]

    job_explored_df   = pd.read_csv("./data/job/explored.csv", delimiter='\t')
    stats_explored_df = pd.read_csv("./data/stats/explored.csv", delimiter='\t')

    job_df   = restrict_to_not_explored(job_df, job_explored_df)
    stats_df = restrict_to_not_explored(stats_df, stats_explored_df)

    # classifier % of native
    job_percentage_nativeml   = (job_df['Native'].sum() / job_df['NativeML'].sum())
    stats_percentage_nativeml = (stats_df['Native'].sum() / stats_df['NativeML'].sum())

    # classifier % of native
    job_percentage_classifier   = (job_df['Native'].sum() / job_df['Classifier'].sum())
    stats_percentage_classifier = (stats_df['Native'].sum() / stats_df['Classifier'].sum())

    # varibo % of native
    job_percentage_varibo   = (job_df['Native'].sum() / job_df[SYSTEM_NAME].sum())
    stats_percentage_varibo = (stats_df['Native'].sum() / stats_df[SYSTEM_NAME].sum())

    labels = ['JOB-C', 'STATS']
    x = np.array([0,0.5])
    width = 0.25 * (1/2)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.set_xlim(-0.4, 1.0)
    ax.set_ylim(0, 7)

    bars1 = ax.bar(
        x - width,
        [job_percentage_nativeml, stats_percentage_nativeml],
        width,
        edgecolor=NATIVEML_COLOR,
        facecolor=transparent(NATIVEML_COLOR, 0.7),
        label='NativeML',
        hatch=NATIVEML_HATCHING,
        linewidth=1.5
    )

    bars2 = ax.bar(
        x,
        [job_percentage_classifier, stats_percentage_classifier],
        width,
        edgecolor=CLASSIFIER_COLOR,
        facecolor=transparent(CLASSIFIER_COLOR, 0.7),
        label='Classifier',
        hatch=CLASSIFIER_HATCHING,
        linewidth=1.5
    )

    bars3 = ax.bar(
        x + width,
        [job_percentage_varibo, stats_percentage_varibo],
        width,
        edgecolor=VARIBO_COLOR,
        facecolor=transparent(VARIBO_COLOR, 0.7),
        label='VariBO',
        hatch=VARIBO_HATCHING,
        linewidth=1.5
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, size=9*4)
    #ax.set_ylabel('Speedup ratio over Native')

    ax.grid(False)

    ax.set_ylim(0, 7)
    plt.yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    def label_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}x",
                ha='center',
                va='bottom'
            )

    label_bars(bars1)
    label_bars(bars2)
    label_bars(bars3)

    #ax.legend(frameon=False)
    plt.tight_layout()

    plt.savefig('./data/ablation_unexplored.pdf')

# shows speedup ratio relative to native of each benchmark side by side
# JOB STATS TPCH ordered lexicographically
def speedup_side_by_side():
    job_df   = pd.read_csv("./data/job/table.csv", delimiter=",")
    stats_df = pd.read_csv("./data/stats/table.csv", delimiter=",")
    tpch_df  = pd.read_csv("./data/tpch/table.csv", delimiter=",")

    # classifier % of native
    job_percentage_classifier   = (job_df['Native'].sum() / job_df['Classifier'].sum())#*100
    stats_percentage_classifier = (stats_df['Native'].sum() / stats_df['Classifier'].sum())#*100
    tpch_percentage_classifier  = (tpch_df['Native'].sum() / tpch_df['Classifier'].sum())#*100

    # varibo % of native
    job_percentage_varibo   = (job_df['Native'].sum() / job_df[SYSTEM_NAME].sum())#*100
    stats_percentage_varibo = (stats_df['Native'].sum() / stats_df[SYSTEM_NAME].sum())#*100
    tpch_percentage_varibo  = (tpch_df['Native'].sum() / tpch_df[SYSTEM_NAME].sum())#*100

    # NativeML % of native
    job_percentage_nativeml   = (job_df['Native'].sum() / job_df['NativeML'].sum())#*100
    stats_percentage_nativeml = (stats_df['Native'].sum() / stats_df['NativeML'].sum())#*100
    tpch_percentage_nativeml  = (tpch_df['Native'].sum() / tpch_df['NativeML'].sum())#*100

    labels = ['JOB-C', 'STATS', 'TPC-H']

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))

    bars2 = ax.bar(
        x,
        [job_percentage_classifier, stats_percentage_classifier, tpch_percentage_classifier],
        width,
        edgecolor=CLASSIFIER_COLOR,
        facecolor=transparent(CLASSIFIER_COLOR, 0.7),
        label='Classifier',
        hatch=CLASSIFIER_HATCHING,
        linewidth=1.5,
    )

    bars3 = ax.bar(
        x + width,
        [job_percentage_varibo, stats_percentage_varibo, tpch_percentage_varibo],
        width,
        edgecolor=VARIBO_COLOR,
        facecolor=transparent(VARIBO_COLOR, 0.7),
        label='VariBO',
        hatch=VARIBO_HATCHING,
        linewidth=1.5
    )

    bars4 = ax.bar(
        x - width,
        [job_percentage_nativeml, stats_percentage_nativeml, tpch_percentage_nativeml],
        width,
        edgecolor=NATIVEML_COLOR,
        facecolor=transparent(NATIVEML_COLOR, 0.7),
        label='NativeML',
        hatch=NATIVEML_HATCHING,
        linewidth=1.5
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, size=9 * 4)
    ax.set_ylabel('Speedup ratio over Native', size=8 * 2.5, loc='top')
    ax.grid(False)

    ax.legend(frameon=False, loc="upper left", fontsize=8 * 3,ncol=2)
    ax.set_ylim(0, 7)
    #plt.yticks([])
    #ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    def label_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}x",
                ha='center',
                va='bottom'
            )

    label_bars(bars2)
    label_bars(bars3)
    label_bars(bars4)

    #ax.legend(frameon=False)
    #plt.set_title("Speedup ratio over native with each model on JOB-C, STATS and TPC-H")
    plt.tight_layout()
    plt.savefig("./data/speedups.pdf")

# shows total execution time of the JOB-C benchmark with each optimizer
# runtimes in table.csv are in milliseconds, plotted in minutes
def job_total_runtime():
    job_df = pd.read_csv("./data/job/table.csv", delimiter=",")

    ms_to_min = 1000 * 60

    optimizers = [
        ('Native',       NATIVE_COLOR,     NATIVE_HATCHING),
        ('NativeML',     NATIVEML_COLOR,   NATIVEML_HATCHING),
        ('Classifier',   CLASSIFIER_COLOR, CLASSIFIER_HATCHING),
        (SYSTEM_NAME,    VARIBO_COLOR,     VARIBO_HATCHING),
    ]

    labels = [name for name, _, _ in optimizers]
    runtimes = [job_df[name].sum() / ms_to_min for name in labels]

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(7, 4))

    all_bars = []
    for i, (name, color, hatch) in enumerate(optimizers):
        bars = ax.bar(
            x[i],
            runtimes[i],
            width,
            edgecolor=color,
            facecolor=transparent(color, 0.7),
            label=name,
            hatch=hatch,
            linewidth=1.5
        )
        all_bars.append(bars)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, size=8 * 1.6)
    ax.set_ylabel('Total execution time (min)', size=8 * 2.5, loc='top')
    ax.grid(False)

    ax.set_ylim(0, max(runtimes) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    def label_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.1f}",
                ha='center',
                va='bottom'
            )

    for bars in all_bars:
        label_bars(bars)

    plt.tight_layout()
    plt.savefig("./data/job/total_runtime.pdf")

# shows total execution time (min) of every optimizer, one panel per benchmark
# totals are summed from table.csv over the queries that every optimizer finished, so all
# bars of a panel cover the same queries (ms in table.csv, plotted in minutes)
# optimizers keep a fixed order and color across panels so they are comparable
def total_runtime_summary(output_file: str):
    benchmarks = [
        ('JOB-C', './data/job/table.csv'),
        ('STATS', './data/stats/table.csv'),
        ('TPC-H', './data/tpch/table.csv'),
    ]

    optimizers = ALL_OPTIMIZERS
    columns = [col for col, _, _, _ in optimizers]

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(16, 4.5))
    x = np.arange(len(optimizers))
    width = 0.8

    for ax, (title, path) in zip(axes, benchmarks):
        df = pd.read_csv(path, dtype={'Query': str}).dropna(subset=columns)
        runtimes = [df[col].sum() / (1000 * 60) for col in columns]

        for i, (_, label, color, hatch) in enumerate(optimizers):
            bar = ax.bar(
                x[i],
                runtimes[i],
                width,
                edgecolor=color,
                facecolor=transparent(color, 0.7),
                hatch=hatch,
                linewidth=1.5,
                label=label,
            )[0]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.1f}",
                ha='center',
                va='bottom',
                size=8 * 1.4,
            )

        ax.set_title(f'{title}')
        ax.set_xticks([])
        ax.set_ylim(0, max(runtimes) * 1.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    axes[0].set_ylabel('Total execution time (min)')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(optimizers),
               frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=8 * 1.6,
               handlelength=1.5, columnspacing=1.0)

    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.savefig(output_file, bbox_inches='tight')

# shows the per-query speedup of VariBO over one baseline (baseline runtime / VariBO runtime)
# queries are sorted by speedup; bars start at 1.0 and grow up where VariBO is faster and
# down where the baseline is faster. queries missing a runtime for either optimizer are skipped.
# the y axis is logarithmic so a 2x win and a 2x loss have the same bar length
def varibo_per_query_speedup(table_file: str, baseline: str, baseline_color: str,
                             baseline_hatch: str, output_file: str):
    df = pd.read_csv(table_file, dtype={'Query': str}).dropna(subset=[baseline, SYSTEM_NAME])
    speedups = (df[baseline] / df[SYSTEM_NAME]).sort_values(ascending=False).to_numpy()

    faster = speedups >= 1
    x = np.arange(len(speedups))

    fig, ax = plt.subplots(figsize=(7, 4))
    for mask, color, hatch, label in [
        (faster,  VARIBO_COLOR,   VARIBO_HATCHING, f'{SYSTEM_NAME} faster'),
        (~faster, baseline_color, baseline_hatch,  f'{baseline} faster'),
    ]:
        ax.bar(
            x[mask],
            speedups[mask] - 1,
            width=0.8,
            bottom=1,
            edgecolor=color,
            facecolor=transparent(color, 0.7),
            hatch=hatch,
            linewidth=1.0,
            label=label,
        )

    ax.set_yscale('log')
    low, high = speedups.min(), speedups.max()
    ymin, ymax = min(low, 1) / 1.15, max(high, 1) * 1.15
    ax.set_ylim(ymin, ymax)
    candidates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    if ymax / ymin < 20:
        candidates = sorted(candidates + [0.25, 0.33, 0.67, 1.5, 3, 4])
    ax.yaxis.set_major_locator(mticker.FixedLocator([t for t in candidates if ymin <= t <= ymax]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}x'))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.axhline(1, color='black', linewidth=0.8)

    ax.set_xlim(-0.6, len(speedups) - 0.4)
    ax.set_xticks([])
    ax.set_xlabel('Queries (sorted by improvement)')
    ax.set_ylabel(f'Speedup over {baseline}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


# per-query speedup of VariBO over Native, NativeML and Classifier for every benchmark
def varibo_per_query_speedups():
    benchmarks = ['job', 'stats', 'tpch']
    baselines = [
        ('Native',     NATIVE_COLOR,     NATIVE_HATCHING,     'native'),
        ('NativeML',   NATIVEML_COLOR,   NATIVEML_HATCHING,   'nativeML'),
        ('Classifier', CLASSIFIER_COLOR, CLASSIFIER_HATCHING, 'classifier'),
    ]
    for benchmark in benchmarks:
        for baseline, color, hatch, suffix in baselines:
            varibo_per_query_speedup(
                f'./data/{benchmark}/table.csv', baseline, color, hatch,
                f'./data/{benchmark}/varibo_vs_{suffix}.pdf',
            )

# runtime percentiles of every optimizer for one benchmark
# each optimizer's queries are sorted by its own runtime, then the runtime at each percentile
# is taken (linear interpolation, 100% = slowest query). only queries that every optimizer
# finished are used, so all optimizers are measured on the same query set.
# runtimes in table.csv are in milliseconds, plotted in seconds
PERCENTILES = [25, 50, 75, 90, 95, 100]

def runtime_percentiles(table_file: str, output_file: str, log_scale: bool = False):
    columns = [col for col, _, _, _ in ALL_OPTIMIZERS]
    df = pd.read_csv(table_file, dtype={'Query': str}).dropna(subset=columns)

    group = np.arange(len(PERCENTILES))
    width = 0.8 / len(ALL_OPTIMIZERS)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (col, label, color, hatch) in enumerate(ALL_OPTIMIZERS):
        values = np.percentile(df[col] / 1000, PERCENTILES)
        ax.bar(
            group + (i - (len(ALL_OPTIMIZERS) - 1) / 2) * width,
            values,
            width,
            edgecolor=color,
            facecolor=transparent(color, 0.7),
            hatch=hatch,
            linewidth=1.2,
            label=label,
        )

    ax.set_xticks(group)
    ax.set_xticklabels([f'{p}%' for p in PERCENTILES])
    ax.set_xlabel(f'Percentile of all queries')
    ax.set_ylabel('Runtime (s)')
    if log_scale:
        ax.set_yscale('log')
        low, high = ax.get_ylim()
        subs = (1, 2, 5) if high / low < 100 else (1,)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=subs))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
              fontsize=8 * 1.6, handlelength=1.5, columnspacing=1.0)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close(fig)


def runtime_percentiles_all():
    runtime_percentiles('./data/job/table.csv',   './data/job/runtime_percentiles.pdf', log_scale=True)
    runtime_percentiles('./data/stats/table.csv', './data/stats/runtime_percentiles.pdf', log_scale=True)
    runtime_percentiles('./data/tpch/table.csv',  './data/tpch/runtime_percentiles.pdf', log_scale=True)

# shows per-query runtimes of the queries where VariBO beats every baseline (Java, Spark, Native)
# benchmarks is a list of (section label, table.csv path); all are drawn on one shared x axis,
# with each benchmark's queries grouped into a labelled section
# runtimes in table.csv are in milliseconds, plotted in seconds
# a runtime of 0 marks a missing measurement, so those queries are skipped
# log_scale helps when runtimes span several orders of magnitude (e.g. STATS)
def varibo_best_queries(benchmarks: list, output_file: str, log_scale: bool = False):
    # (column in table.csv, legend label, color, hatching), in plotting order
    optimizers = [
        ('Java',         'Java',                     NATIVEML_COLOR, NATIVEML_HATCHING),
        ('Spark',        'Spark',                    SPARK_COLOR,    SPARK_HATCHING),
        ('Native',       'Wayang',                   NATIVE_COLOR,   NATIVE_HATCHING),
        (SYSTEM_NAME,    'Cross-platform potential', VARIBO_COLOR,   VARIBO_HATCHING),
    ]
    names = [name for name, _, _, _ in optimizers]

    sections = []
    for benchmark, input_file in benchmarks:
        df = pd.read_csv(input_file, delimiter=",")
        available = [name for name in names if name in df.columns]
        df = df[(df[available] > 0).all(axis=1)]
        df = df[df[available].idxmin(axis=1) == SYSTEM_NAME]
        df = df.sort_values('Query').reindex(columns=['Query'] + names)
        df['Benchmark'] = benchmark
        sections.append(df)
    table_df = pd.concat(sections).reset_index(drop=True)

    ms_to_s = 1000

    x = np.arange(len(table_df))
    width = 0.8 / len(optimizers)

    # center each query's bars on its tick, skipping baselines its benchmark lacks so no gaps remain
    present = table_df[names].notna()
    slot = present.cumsum(axis=1) - 1
    offset = (present.sum(axis=1) - 1) / 2

    fig, ax = plt.subplots(figsize=(14, 4.5))

    for name, label, color, hatch in optimizers:
        rows = present[name]
        ax.bar(
            x[rows] + (slot[name][rows] - offset[rows]) * width,
            table_df[name][rows] / ms_to_s,
            width,
            edgecolor=color,
            facecolor=transparent(color, 0.7),
            label=label,
            hatch=hatch,
            linewidth=1.5
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(q) for q in table_df['Query']],
        rotation=45,
        ha='right',
        rotation_mode='anchor',
        size=8 * 1.5
    )
    ax.set_xlim(-0.5, len(table_df) - 0.5)
    ax.set_ylabel('Query runtime (s)')
    ax.grid(False)

    if log_scale:
        ax.set_yscale('log')

    # separate the benchmark sections and name each one below its tick labels
    section_bounds = []
    first = 0
    for df in sections:
        section_bounds.append((first, first + len(df) - 1, df['Benchmark'].iloc[0]))
        first += len(df)

    for first, last, benchmark in section_bounds:
        if first > 0:
            ax.axvline(first - 0.5, color='grey', linestyle='--', linewidth=1)
        ax.annotate(
            '', xy=(first - 0.4, -0.3), xytext=(last + 0.4, -0.3),
            xycoords=('data', 'axes fraction'),
            arrowprops=dict(arrowstyle='-', color='black', linewidth=1),
            annotation_clip=False,
        )
        ax.text(
            (first + last) / 2, -0.33, benchmark,
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontweight='bold'
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(optimizers),
        fontsize=8 * 1.5,
        handlelength=1.5,
        columnspacing=1.0,
    )

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')

# TPC-H can run on a single engine, so its java/spark baselines are measured separately
# each <engine>/query<N>-executions.txt holds one runtime (ms) per line, one line per repetition
# the median per query is written to the Java and Spark columns of the tpch table.csv
def add_tpch_single_engine_runtimes():
    table_file = "./data/tpch/table.csv"
    table_df = pd.read_csv(table_file, delimiter=",")

    for engine in ['Java', 'Spark']:
        runtimes = []
        for query in table_df['Query']:
            with open(f"./data/tpch/{engine.lower()}/query{query}-executions.txt") as f:
                runs = [int(line) for line in f if line.strip()]
            runtimes.append(int(np.median(runs)))

        if engine in table_df.columns:
            table_df[engine] = runtimes
        else:
            table_df.insert(table_df.columns.get_loc('Native') + 1 + ['Java', 'Spark'].index(engine), engine, runtimes)

    table_df.to_csv(table_file, sep=",", index=False)

# shows per-query runtimes of the TPC-H queries where Wayang or VariBO beats both Java and Spark
# runtimes in table.csv are in milliseconds, plotted in seconds
def tpch_varibo_best_queries(output_file: str, log_scale: bool = False):
    # (column in table.csv, legend label, color, hatching), in plotting order
    optimizers = [
        ('Java',      'Java',                     EXPLORED_COLOR, EXPLORED_HATCHING),
        ('Spark',     'Spark',                    SPARK_COLOR,    SPARK_HATCHING),
        ('Native',    'Wayang',                   NATIVE_COLOR,   NATIVE_HATCHING),
        (SYSTEM_NAME, 'Cross-platform potential', VARIBO_COLOR,   VARIBO_HATCHING),
    ]
    names = [name for name, _, _, _ in optimizers]

    table_df = pd.read_csv("./data/tpch/table.csv", delimiter=",", usecols=['Query'] + names)
    table_df = table_df[(table_df[names] > 0).all(axis=1)]
    # the cross-platform potential can at best match the fastest single engine, faster times are measurement noise
    best_single_engine = table_df[['Java', 'Spark']].min(axis=1)
    table_df[SYSTEM_NAME] = table_df[SYSTEM_NAME].clip(lower=best_single_engine)
    table_df = table_df[(table_df['Native'] < best_single_engine) | (table_df[SYSTEM_NAME] <= best_single_engine)]
    table_df = table_df.sort_values('Query').reset_index(drop=True)

    ms_to_s = 1000

    x = np.arange(len(table_df))
    width = 0.2
    offset = (len(optimizers) - 1) / 2

    fig, ax = plt.subplots(figsize=(14, 4))

    for i, (name, label, color, hatch) in enumerate(optimizers):
        ax.bar(
            x + (i - offset) * width,
            table_df[name] / ms_to_s,
            width,
            edgecolor=color,
            facecolor=transparent(color, 0.7),
            label=label,
            hatch=hatch,
            linewidth=1.5
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in table_df['Query']])
    ax.set_xlabel('Query')
    ax.set_ylabel('Query runtime (s)')
    ax.grid(False)

    if log_scale:
        ax.set_yscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=8 * 1.3,
        handlelength=1.5,
        columnspacing=1.0,
    )

    plt.tight_layout()
    plt.savefig(output_file)

def exploration_graph(input_file: str, ouput_file: str, queries: list, ylim):
    table_df                = pd.read_csv(input_file, delimiter='\t', usecols=['Native', 'Explored'])
    table_df['improvement'] = table_df['Native'] / table_df['Explored']
    table_df['query']       = table_df.index

    table_df = table_df.sort_values('improvement', ascending=False).reset_index(drop=True)

    baseline = 1.0

    colors = [EXPLORED_COLOR if v >= baseline else NATIVE_COLOR
              for v in table_df['improvement']]

    bottoms = []
    heights = []

    for v in table_df['improvement']:
        if v >= baseline:
            bottoms.append(baseline)
            heights.append(v - baseline)
        else:
            bottoms.append(v)
            heights.append(baseline - v)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        range(len(table_df)),
        heights,
        bottom=bottoms,
        color=[transparent(color, 0.7) for color in colors],
        edgecolor=colors,
        width=0.7,
        hatch=[EXPLORED_HATCHING if v >= baseline else NATIVE_HATCHING
              for v in table_df['improvement']]
    )

    ax.set_xlim(-0.5, len(table_df) - 0.5)
    ax.set_xticks(range(len(table_df)))
    ax.set_xticklabels(
        [str(q) for q in table_df['query']],
        rotation=45
    )

    ax.set_xlabel('Query (sorted by improvement)')
    ax.set_ylabel('Speedup over Native')

    ax.axhline(y=baseline, color='black', linestyle='--', linewidth=1)

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:.2f}x')
    )

    ax.spines[['top', 'right']].set_visible(False)

    ax.legend(
        handles=[
            Patch(
                facecolor=transparent(EXPLORED_COLOR, 0.7),
                edgecolor=EXPLORED_COLOR,
                hatch=EXPLORED_HATCHING,
                label='Explored faster'
            ),
            Patch(
                facecolor=transparent(NATIVE_COLOR, 0.7),
                edgecolor=NATIVE_COLOR,
                hatch=NATIVE_HATCHING,
                label='Native faster'
            )
        ],
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(ouput_file, dpi=300)

def make_tpch_cumsum():
    tpch_df  = pd.read_csv("./data/tpch/table.csv", delimiter=",", usecols=['Native','NativeML','Classifier',SYSTEM_NAME])
    job_df   = pd.read_csv("./data/job/table.csv", delimiter=",", usecols=['Native','NativeML','Classifier',SYSTEM_NAME])
    stats_df = pd.read_csv("./data/stats/table.csv", delimiter=",", usecols=['Native','NativeML','Classifier',SYSTEM_NAME])

    def speedups(df):
        totals = df.sum()
        return totals['Native'] / totals['NativeML'], totals['Native'] / totals['Classifier'], totals['Native'] / totals[SYSTEM_NAME]

    job_native_ml, job_classifier,   job_varibo   = speedups(job_df)
    stats_native_ml, stats_classifier, stats_varibo = speedups(stats_df)
    tpch_native_ml, tpch_classifier,  tpch_varibo  = speedups(tpch_df)

    labels = ['JOB-C', 'STATS', 'TPC-H']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width, [job_native_ml, stats_native_ml, tpch_native_ml], width, color=EXPLORED_COLOR, label='NativeML')
    bars2 = ax.bar(x,         [job_classifier, stats_classifier, tpch_classifier], width, color=CLASSIFIER_COLOR, label='Classifier')
    bars3 = ax.bar(x + width, [job_varibo,     stats_varibo,     tpch_varibo],     width, color=VARIBO_COLOR,     label='VariBO')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup over Native")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)
    ax.set_ylim(0, 4.5)
    plt.yticks([])

    def label_bars(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}x",
                    ha='center', va='bottom')

    label_bars(bars1)
    label_bars(bars2)
    label_bars(bars3)

    ax.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    plt.savefig('./data/speedups.pdf')

def make_optimization_barchart():
    segments = ["Encoding", "Inference", "Unpacking", "Decoding", "Enumeration"]
    segment_colors = ['#56B4E9', '#F0E442', '#CC79A7', '#D55E00', '#E69F00']
    hatches = ['---', '\\\\\\', '///', 'xx', '||']

    datasets = [
        ("job",   "JOB-C"),
        ("stats", "STATS"),
        ("tpch",  "TPC-H"),
    ]

    opt_dfs = []
    for dataset, label in datasets:
        opt_df = pd.read_csv(f"./data/optimization/files/table_barchart_{dataset}.csv",
                             delimiter='\t', usecols=["Optimizer"] + segments)
        opt_df = opt_df.set_index("Optimizer").fillna(0)

        fig, ax = plt.subplots(figsize=(7, 4))

        bottoms = np.zeros(len(opt_df))
        for seg, color, hatch in zip(segments, segment_colors, hatches):
            values = opt_df[seg].values
            ax.bar(opt_df.index, values, bottom=bottoms, color=transparent(color, 0.7), edgecolor=color, hatch=hatch, label=seg)
            bottoms += values

        for i, total in enumerate(bottoms):
            ax.text(i, total, f'{total:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_ylabel("Optimization time (ms)")
        #ax.set_title(label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, loc='upper left')
        plt.tight_layout()
        plt.savefig(f'./data/optimization/barchart_{dataset}.pdf')
        plt.close(fig)
        opt_dfs.append(opt_df)

    avg_df = sum(opt_dfs) / len(opt_dfs)

    fig, ax = plt.subplots(figsize=(7, 4))

    bottoms = np.zeros(len(avg_df))
    for seg, color, hatch in zip(segments, segment_colors, hatches):
        values = avg_df[seg].values
        ax.bar(avg_df.index, values, bottom=bottoms, color=transparent(color, 0.7), edgecolor=color, hatch=hatch, label=seg)
        bottoms += values

    for i, total in enumerate(bottoms):
        ax.text(i, total, f'{total:.1f}', ha='center', va='bottom', fontsize=9 * 1.5)

    ax.set_ylabel("Optimization time (ms)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.legend(frameon=False, loc='upper right')

    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
    )

    plt.tight_layout()
    plt.savefig('./data/optimization/optimization.pdf')
    plt.close(fig)

def make_optimization_linechart():
    base = "./data/optimization/files/optimizations"
    datasets = [
        ("job",   "JOB-C"),
        ("stats", "STATS"),
        ("tpch",  "TPC-H"),
    ]

    for dataset, label in datasets:
        native_df = pd.read_csv(f"{base}/native/{dataset}/table.csv", sep=r'\s+')
        cost_df   = pd.read_csv(f"{base}/cost/{dataset}/table.csv",   sep=r'\s+')
        varibo_df = pd.read_csv(f"{base}/{dataset}/table.csv",        sep=r'\s+')

        operators     = native_df['Operators']
        cost_factor   = native_df['Time'].values / cost_df['Time'].values
        varibo_factor = native_df['Time'].values / varibo_df['Time'].values

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.axhline(y=1.0, color=NATIVE_COLOR, linestyle='--', linewidth=0.8, label='Native')
        ax.plot(operators, cost_factor,   color=NATIVEML_COLOR, label='NativeML', linewidth=1.5)
        ax.plot(operators, varibo_factor, color=VARIBO_COLOR,   label='VariBO',   linewidth=1.5)

        ax.set_xlabel('Number of operators')
        ax.set_ylabel('Speedup over Native')
        #ax.set_title(label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
        )
        plt.tight_layout()
        plt.savefig(f'./data/optimization/linechart_{dataset}.pdf')
        plt.close(fig)

def smooth_monotone(x, y, num=200):
    """Monotone cubic (PCHIP) interpolation, so the curve never overshoots the measured points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.diff(x)
    delta = np.diff(y) / h

    d = np.zeros_like(y)
    for k in range(1, len(x) - 1):
        if delta[k - 1] * delta[k] > 0:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    if len(x) > 1:
        d[0], d[-1] = delta[0], delta[-1]

    xs = np.linspace(x[0], x[-1], num)
    k = np.clip(np.searchsorted(x, xs, side='right') - 1, 0, len(x) - 2)
    t = (xs - x[k]) / h[k]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    ys = h00 * y[k] + h10 * h[k] * d[k] + h01 * y[k + 1] + h11 * h[k] * d[k + 1]
    return xs, ys

def make_exploration_comparison(varibo_df: pd.DataFrame,
                                random_df: pd.DataFrame,
                                ablation_df: pd.DataFrame,
                                xlim: int,
                                output_file: str):

    series = [
        (varibo_df, VARIBO_COLOR, 'VariBO', 'o'),
        (random_df, CLASSIFIER_COLOR, 'Random', 's'),
        (ablation_df, EXPLORED_COLOR, r'VariBO ($\gamma = 0$)', '^'),
    ]

    # plot against wall-clock time if every series has timestamps, otherwise against steps
    use_time = all("Timestamp" in df.columns for df, _, _, _ in series)
    # a Time column (HH:MM since the optimization started) is plotted as optimization time in hours
    use_hours = all("Time" in df.columns for df, _, _, _ in series)
    x_col = "Elapsed" if use_time else "Hours" if use_hours else "Steps"
    pad_col = "Hours" if use_hours else "Steps"

    for df, _, _, _ in series:
        df["Runtime"] = df["Runtime"] / 1000.0 #convert runtime to s

        if use_hours:
            hours_minutes = df["Time"].astype(str).str.split(":", expand=True).reindex(columns=[0, 1]).fillna(0).astype(int)
            df["Hours"] = hours_minutes[0] + hours_minutes[1] / 60.0

        if use_time:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H:%M:%S")
            df.sort_values("Timestamp", inplace=True)

            start_time = df["Timestamp"].iloc[0]
            df["Elapsed"] = (df["Timestamp"] - start_time).dt.total_seconds()

        if df[pad_col].iloc[-1] < xlim:
            df.loc[len(df)] = {**df.iloc[-1].to_dict(), pad_col: xlim}

    fig, ax = plt.subplots(figsize=(7, 4))

    # hollow, distinct markers so overlapping points stay visible
    legend_handles = []
    for df, color, label, marker in series:
        points = df.drop_duplicates(subset=x_col, keep='last')
        xs, ys = smooth_monotone(points[x_col], points['Runtime'])
        ax.plot(xs, ys, color=color)
        ax.plot(points[x_col], points['Runtime'], linestyle='none', marker=marker, clip_on=False,
                markerfacecolor='none', markeredgewidth=1.5, color=color)
        legend_handles.append(Line2D([], [], color=color, marker=marker, markerfacecolor='none',
                                     markeredgewidth=1.5, label=label))

    texts = []

    if use_time:
        for df, color, _, _ in series:
            for _, row in df.iterrows():
                texts.append(
                    ax.text(
                        row["Elapsed"],
                        row["Runtime"],
                        str(int(row["Steps"])),
                        fontsize=12,
                        color=color
                    )
                )

        adjust_text(
            texts,
            ax=ax
        )

    ax.set_xlabel("Time since start (s)" if use_time else "Optimization time (h)" if use_hours else "Steps")
    ax.set_ylabel("Best-found plan runtime (s)")

    # put the origin exactly in the bottom left corner, no padding
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=legend_handles, frameon=False)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)

def main():
    set_paper_style()
    #finetuning_capability_ablation_study()
    #generalization_ablation_study()
    speedup_side_by_side()
    job_total_runtime()
    total_runtime_summary('./data/total_runtime_summary.pdf')
    varibo_per_query_speedups()
    runtime_percentiles_all()
    varibo_best_queries(
        [('JOB', './data/job/table.csv'), ('STATS', './data/stats/table.csv'), ('TPC-H', './data/tpch/table.csv')],
        './data/varibo_best_queries.pdf',
        log_scale=True
    )
    tpch_varibo_best_queries('./data/tpch/varibo_best_queries.pdf', log_scale=True)

    job_varibo_df = pd.read_csv("./data/job/14.explored.csv", delimiter='\t', usecols=['Time','Runtime'], dtype={'Time': str})
    job_random_df = pd.read_csv("./data/job/random.14.explored.csv", delimiter='\t', usecols=['Time','Runtime'], dtype={'Time': str})
    job_ablation_df = pd.read_csv("./data/job/ablation.14.explored.csv", delimiter='\t', usecols=['Time','Runtime'], dtype={'Time': str})
    make_exploration_comparison(job_varibo_df, job_random_df, job_ablation_df, xlim=3, output_file='./data/job/exploration_comparison.pdf')

    tpch_varibo_df = pd.read_csv("./data/tpch/12.explored.csv", delimiter='\t', usecols=['Time','Runtime'], dtype={'Time': str})
    tpch_random_df = pd.read_csv("./data/tpch/random.12.explored.csv", delimiter='\t', usecols=['Time','Runtime'], dtype={'Time': str})
    tpch_ablation_df = pd.read_csv("./data/tpch/ablation.12.explored.csv", delimiter='\t', usecols=['Time','Runtime'], dtype={'Time': str})
    make_exploration_comparison(tpch_varibo_df, tpch_random_df, tpch_ablation_df, xlim=0.5, output_file='./data/tpch/exploration_comparison.pdf')

    stats_varibo_df = pd.read_csv("./data/stats/query.explored.csv", delimiter='\t', usecols=['Time','Runtime'], dtype={'Time': str})
    stats_random_df = pd.read_csv("./data/stats/random.query.explored.csv", delimiter='\t', usecols=['Time','Runtime'], dtype={'Time': str})
    stats_ablation_df = pd.read_csv("./data/stats/ablation.explored.csv", delimiter='\t', usecols=['Time','Runtime'], dtype={'Time': str})
    make_exploration_comparison(stats_varibo_df, stats_random_df, stats_ablation_df, xlim=2, output_file='./data/stats/exploration_comparison.pdf')


if __name__ == "__main__":
    #stats_exploration_queries = pd.read_csv("./data/stats/explored.csv", delimiter='\t', usecols=["Query"])['Query']
    #exploration_graph('./data/stats/explored.csv', './data/stats/explored.pdf', list(stats_exploration_queries))
    main()
    #finetuning_capability_ablation_study()


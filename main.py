import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import numpy as np

plt.close("all")

CLASSIFIER_COLOR = '#0072B2' #classifier
VARIBO_COLOR = '#E69F00' #varibo
NATIVE_COLOR = "#009E73" #native
EXPLORED_COLOR = "#D55E00" #explored

SYSTEM_NAME = "VariBO"

# shows speed up ratio over each benchmark compared to native
# we ablate the queries THAT WE DONT EXPLORE ON from the benchmark and see if the explored queries correctly reconstruct
# JOB STATS TPCH ordered lexicographically
def finetuning_capability_ablation_study():
    job_df   = pd.read_csv("./data/job/table.csv", delimiter='\t')
    stats_df = pd.read_csv("./data/stats/table.csv", delimiter='\t')
    tpch_df  = pd.read_csv("./data/tpch/table.csv", delimiter='\t')

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

    labels = ['JOB', 'STATS', 'TPCH']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))

    bars1 = ax.bar(x - width, [job_percentage_classifier, stats_percentage_classifier, tpch_percentage_classifier], width, color=CLASSIFIER_COLOR, label='Classifier')
    bars2 = ax.bar(x,         [job_percentage_varibo, stats_percentage_varibo, tpch_percentage_varibo], width, color=VARIBO_COLOR, label='VariBO')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Speedup ratio over Native')

    plt.title('Speedup over Native in explored queries')


    ax.grid(False)

    ax.set_ylim(0, 7.5)
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
                va='bottom',
                fontsize=9
            )

    label_bars(bars1)
    label_bars(bars2)

    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('./data/ablation.pdf')


# shows speed up ratio over each benchmark compared to native
# we ablate the explored queries from the benchmark and see if the explored queries provide value to the rest of the benchmark
# JOB STATS TPCH ordered lexicographically
def generalization_ablation_study():
    job_df   = pd.read_csv("./data/job/table.csv", delimiter='\t')
    stats_df = pd.read_csv("./data/stats/table.csv", delimiter='\t')

    def restrict_to_not_explored(full_df, explored_df):
        explored_queries = set(explored_df["Query"])
        return full_df[~full_df["Query"].isin(explored_queries)]

    job_explored_df   = pd.read_csv("./data/job/explored.csv", delimiter='\t')
    stats_explored_df = pd.read_csv("./data/stats/explored.csv", delimiter='\t')

    job_df   = restrict_to_not_explored(job_df, job_explored_df)
    stats_df = restrict_to_not_explored(stats_df, stats_explored_df)

    # classifier % of native
    job_percentage_classifier   = (job_df['Native'].sum() / job_df['Classifier'].sum())
    stats_percentage_classifier = (stats_df['Native'].sum() / stats_df['Classifier'].sum())

    # varibo % of native
    job_percentage_varibo   = (job_df['Native'].sum() / job_df[SYSTEM_NAME].sum())
    stats_percentage_varibo = (stats_df['Native'].sum() / stats_df[SYSTEM_NAME].sum())

    labels = ['JOB', 'STATS']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))

    bars1 = ax.bar(x - width, [job_percentage_classifier, stats_percentage_classifier], width, color=CLASSIFIER_COLOR, label='Classifier')
    bars2 = ax.bar(x,         [job_percentage_varibo, stats_percentage_varibo], width, color=VARIBO_COLOR, label='VariBO')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Speedup ratio over Native')

    ax.grid(False)

    ax.set_ylim(0, 3.5)
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
                va='bottom',
                fontsize=9
            )

    label_bars(bars1)
    label_bars(bars2)

    ax.legend(frameon=False)
    plt.tight_layout()

    plt.savefig('./data/ablation_unexplored.pdf')

# shows speedup ratio relative to native of each benchmark side by side
# JOB STATS TPCH ordered lexicographically
def speedup_side_by_side():
    job_df   = pd.read_csv("./data/job/table.csv", delimiter='\t')
    stats_df = pd.read_csv("./data/stats/table.csv", delimiter='\t')
    tpch_df  = pd.read_csv("./data/tpch/table.csv", delimiter='\t')

    # classifier % of native
    job_percentage_classifier   = (job_df['Native'].sum() / job_df['Classifier'].sum())#*100
    stats_percentage_classifier = (stats_df['Native'].sum() / stats_df['Classifier'].sum())#*100
    tpch_percentage_classifier  = (tpch_df['Native'].sum() / tpch_df['Classifier'].sum())#*100

    # varibo % of native
    job_percentage_varibo   = (job_df['Native'].sum() / job_df[SYSTEM_NAME].sum())#*100
    stats_percentage_varibo = (stats_df['Native'].sum() / stats_df[SYSTEM_NAME].sum())#*100
    tpch_percentage_varibo  = (tpch_df['Native'].sum() / tpch_df[SYSTEM_NAME].sum())#*100

    labels = ['JOB', 'STATS', 'TPCH']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))

    bars1 = ax.bar(x - width, [job_percentage_classifier, stats_percentage_classifier, tpch_percentage_classifier], width, color=CLASSIFIER_COLOR, label='Classifier')
    bars2 = ax.bar(x,         [job_percentage_varibo, stats_percentage_varibo, tpch_percentage_varibo], width, color=VARIBO_COLOR, label='VariBO')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Speedup ratio over Native')

    ax.grid(False)

    ax.set_ylim(0, 1.6)
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
                va='bottom',
                fontsize=9
            )

    label_bars(bars1)
    label_bars(bars2)

    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

def exploration_graph(input_file: str, ouput_file: str, queries: list):
    table_df = pd.read_csv(input_file, delimiter='\t', usecols=['Native', 'Explored'])

    # Speedup / improvement
    table_df['improvement'] = table_df['Native'] / table_df['Explored']
    table_df['query'] = table_df.index

    # Sort best to worst
    table_df = table_df.sort_values('improvement', ascending=False).reset_index(drop=True)

    baseline = 1.0

    # Colors
    colors = [NATIVE_COLOR if v >= baseline else EXPLORED_COLOR
              for v in table_df['improvement']]

    # ---- NEW: bar geometry ----
    bottoms = []
    heights = []

    for v in table_df['improvement']:
        if v >= baseline:
            bottoms.append(baseline)
            heights.append(v - baseline)
        else:
            bottoms.append(v)
            heights.append(baseline - v)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.bar(
        range(len(table_df)),
        heights,
        bottom=bottoms,
        color=colors,
        width=0.7
    )

    ax.set_xticks(range(len(table_df)))
    ax.set_xticklabels([f"{q}" for q in queries],
                       rotation=45, ha='right', fontsize=9)

    ax.set_xlabel('Query (sorted by improvement)', color='gray')
    ax.set_ylabel('Speedup over Native', color='gray')

    # Baseline at 1×
    ax.axhline(y=baseline, color='black', linestyle='--', linewidth=1)

    # Axis & styling
    ax.set_ylim(0.75, 3)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:.2f}x')
    )

    ax.spines[['top', 'right']].set_visible(False)

    # Legend
    ax.legend(
        handles=[
            Patch(color='#1D9E75', label='Explored faster'),
            Patch(color='#D85A30', label='Native faster')
        ],
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(ouput_file)

def make_job_cumsum():
    table_df = pd.read_csv("./data/job/table.csv", delimiter='\t', usecols=['Native','Classifier',SYSTEM_NAME])
    table_df = table_df.cumsum()
    plt.figure()
    table_df.plot()
    plt.savefig('./data/job/vis.pdf')

def make_stats_cumsum():
    table_df = pd.read_csv("./data/stats/table.csv", delimiter='\t', usecols=['Native','Classifier',SYSTEM_NAME])
    table_df = table_df.cumsum()
    plt.figure()
    table_df.plot()
    plt.savefig('./data/stats/vis.pdf')

def make_tpch_cumsum():
    tpch_df = pd.read_csv("./data/tpch/table.csv", delimiter='\t', usecols=['Native','NativeML','Classifier',SYSTEM_NAME])
    job_df = pd.read_csv("./data/job/table.csv", delimiter='\t', usecols=['Native','NativeML','Classifier',SYSTEM_NAME])
    stats_df = pd.read_csv("./data/stats/table.csv", delimiter='\t', usecols=['Native','NativeML','Classifier',SYSTEM_NAME])

    def speedups(df):
        totals = df.sum()
        return totals['Native'] / totals['NativeML'], totals['Native'] / totals['Classifier'], totals['Native'] / totals[SYSTEM_NAME]

    job_native_ml, job_classifier,   job_varibo   = speedups(job_df)
    stats_native_ml, stats_classifier, stats_varibo = speedups(stats_df)
    tpch_native_ml, tpch_classifier,  tpch_varibo  = speedups(tpch_df)

    labels = ['JOB', 'STATS', 'TPC-H']
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
                    ha='center', va='bottom', fontsize=9)

    label_bars(bars1)
    label_bars(bars2)
    label_bars(bars3)

    ax.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    plt.savefig('./data/speedups.pdf')

def make_optimization_barchart():
    segments = ["Encoding", "Inference", "Unpacking", "Decoding", "Enumeration"]
    opt_df = pd.read_csv("./data/optimization/table.csv", delimiter='\t',
                         usecols=["Optimizer"] + segments)
    opt_df = opt_df.set_index("Optimizer").fillna(0)

    segment_colors = ['#56B4E9', '#F0E442', '#CC79A7', '#D55E00', '#009E73']

    fig, ax = plt.subplots(figsize=(6, 4))

    bottoms = np.zeros(len(opt_df))
    for seg, color in zip(segments, segment_colors):
        values = opt_df[seg].values
        ax.bar(opt_df.index, values, bottom=bottoms, color=color, label=seg)
        bottoms += values

    ax.set_ylabel("Optimization time (ms)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper left')
    plt.tight_layout()
    plt.savefig('./data/optimization/optimization.pdf')

def main():
    #exploration_graph('./data/tpch/explored.csv', './data/tpch/explored.pdf', range(0, 30))

    #stats_exploration_queries = pd.read_csv("./data/stats/explored.csv", delimiter='\t', usecols=["Query"])['Query']
    #exploration_graph('./data/stats/explored.csv', './data/stats/explored.pdf', list(stats_exploration_queries))
    make_tpch_cumsum()
    #make_optimization_barchart()


if __name__ == "__main__":
    #stats_exploration_queries = pd.read_csv("./data/stats/explored.csv", delimiter='\t', usecols=["Query"])['Query']
    #exploration_graph('./data/stats/explored.csv', './data/stats/explored.pdf', list(stats_exploration_queries))
    #main()
    finetuning_capability_ablation_study()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np

plt.close("all")

EXPLORED_COLOR   = '#0072B2'
CLASSIFIER_COLOR = '#E69F00'
VARIBO_COLOR     = "#009E73"
NATIVE_COLOR     = "#D55E00"
NATIVEML_COLOR   = "#CC79A7"

EXPLORED_HATCHING   = '---'
CLASSIFIER_HATCHING = '\\\\\\'
VARIBO_HATCHING     = '///'
NATIVE_HATCHING     = 'xx'
NATIVEML_HATCHING   = '||'

SYSTEM_NAME = "VariBO"

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
    ax.set_xticklabels(labels)
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
    ax.set_xticklabels(labels)
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

    # NativeML % of native
    job_percentage_nativeml   = (job_df['Native'].sum() / job_df['NativeML'].sum())#*100
    stats_percentage_nativeml = (stats_df['Native'].sum() / stats_df['NativeML'].sum())#*100
    tpch_percentage_nativeml  = (tpch_df['Native'].sum() / tpch_df['NativeML'].sum())#*100

    labels = ['JOB-C', 'STATS', 'TPC-H']

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))

    bars1 = ax.bar(
        x,
        [job_percentage_classifier, stats_percentage_classifier, tpch_percentage_classifier],
        width,
        edgecolor=CLASSIFIER_COLOR,
        facecolor=transparent(CLASSIFIER_COLOR, 0.7),
        label='Classifier',
        hatch=CLASSIFIER_HATCHING,
        linewidth=1.5
    )

    bars2 = ax.bar(
        x + width,
        [job_percentage_varibo, stats_percentage_varibo, tpch_percentage_varibo],
        width,
        edgecolor=VARIBO_COLOR,
        facecolor=transparent(VARIBO_COLOR, 0.7),
        label='VariBO',
        hatch=VARIBO_HATCHING,
        linewidth=1.5
    )

    bars3 = ax.bar(
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
    ax.set_xticklabels(labels)
    ax.set_ylabel('Speedup ratio over Native')

    ax.grid(False)

    ax.set_ylim(0, 4)
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

    label_bars(bars3)
    label_bars(bars1)
    label_bars(bars2)

    #ax.legend(frameon=False)
    #plt.set_title("Speedup ratio over native with each model on JOB-C, STATS and TPC-H")
    plt.tight_layout()
    plt.savefig("./data/speedups.pdf")

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
    tpch_df  = pd.read_csv("./data/tpch/table.csv", delimiter='\t', usecols=['Native','NativeML','Classifier',SYSTEM_NAME])
    job_df   = pd.read_csv("./data/job/table.csv", delimiter='\t', usecols=['Native','NativeML','Classifier',SYSTEM_NAME])
    stats_df = pd.read_csv("./data/stats/table.csv", delimiter='\t', usecols=['Native','NativeML','Classifier',SYSTEM_NAME])

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
        ax.set_title(label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, loc='upper left')
        plt.tight_layout()
        plt.savefig(f'./data/optimization/barchart_{dataset}.pdf')
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
        ax.set_title(label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        plt.tight_layout()
        plt.savefig(f'./data/optimization/linechart_{dataset}.pdf')
        plt.close(fig)

def make_exploration_comparison(varibo_df: pd.DataFrame, random_df: pd.DataFrame, xlim: int, output_file: str):
    for df in (varibo_df, random_df):
        if df['Steps'].iloc[-1] < xlim:
            df.loc[len(df)] = [xlim, df['Runtime'].iloc[-1]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(varibo_df['Steps'], varibo_df['Runtime'], color=VARIBO_COLOR, label='VariBO')
    ax.plot(random_df['Steps'], random_df['Runtime'], color=CLASSIFIER_COLOR, label='Random')

    ax.set_xlim(0, xlim)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Execution time (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_file)


def main():
    set_paper_style()
    finetuning_capability_ablation_study()
    generalization_ablation_study()
    speedup_side_by_side()

    """
    exploration_graph('./data/tpch/explored.csv', './data/tpch/explored.pdf', range(0, 30), (0.75, 7))
    exploration_graph('./data/stats/explored.csv', './data/stats/explored.pdf', range(0, 30), (0.75, 4))
    exploration_graph('./data/job/explored.csv', './data/job/explored.pdf', range(0, 30), (0.75, 3))
    """
    make_optimization_barchart()
    make_optimization_linechart()

    #plt.show()

    #stats_exploration_queries = pd.read_csv("./data/stats/explored.csv", delimiter='\t', usecols=["Query"])['Query']
    #exploration_graph('./data/stats/explored.csv', './data/stats/explored.pdf', list(stats_exploration_queries))
    #make_tpch_cumsum()
    #make_optimization_barchart()

    #job_varibo_df = pd.read_csv("./data/job/14.explored.csv", delimiter='\t', usecols=['Steps','Runtime'])
    #job_random_df = pd.read_csv("./data/job/random.14.explored.csv", delimiter='\t', usecols=['Steps','Runtime'])
    #make_exploration_comparison(job_varibo_df, job_random_df, xlim=125, output_file='./data/job/exploration_comparison.pdf')

    #tpch_varibo_df = pd.read_csv("./data/tpch/12.explored.csv", delimiter='\t', usecols=['Steps','Runtime'])
    #tpch_random_df = pd.read_csv("./data/tpch/random.12.explored.csv", delimiter='\t', usecols=['Steps','Runtime'])
    #make_exploration_comparison(tpch_varibo_df, tpch_random_df, xlim=50, output_file='./data/tpch/exploration_comparison.pdf')

    #stats_varibo_df = pd.read_csv("./data/stats/query.explored.csv", delimiter='\t', usecols=['Steps','Runtime'])
    #stats_random_df = pd.read_csv("./data/stats/random.query.explored.csv", delimiter='\t', usecols=['Steps','Runtime'])
    #make_exploration_comparison(stats_varibo_df, stats_random_df, xlim=50, output_file='./data/stats/exploration_comparison.pdf')



if __name__ == "__main__":
    #stats_exploration_queries = pd.read_csv("./data/stats/explored.csv", delimiter='\t', usecols=["Query"])['Query']
    #exploration_graph('./data/stats/explored.csv', './data/stats/explored.pdf', list(stats_exploration_queries))
    main()
    #finetuning_capability_ablation_study()


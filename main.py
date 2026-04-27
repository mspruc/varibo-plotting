import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import numpy as np

plt.close("all")

def exploration_graph(input_file: str, ouput_file: str, queries: list):
    table_df = pd.read_csv(input_file, delimiter='\t', usecols=['Native', 'Explored'])

    # Compute improvement %
    table_df['improvement'] = (table_df['Native'] - table_df['Explored']) / table_df['Native'] * 100
    table_df['query'] = table_df.index

    # Sort best to worst
    table_df = table_df.sort_values('improvement', ascending=False).reset_index(drop=True)

    colors = ['#1D9E75' if v >= 0 else '#D85A30' for v in table_df['improvement']]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.bar(range(len(table_df)), table_df['improvement'], color=colors, width=0.7)

    ax.set_xticks(range(len(table_df)))
    ax.set_xticklabels([f"{q}" for q in queries], rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Query (sorted by improvement)', color='gray')
    ax.set_ylabel('Improvement over Native (%)', color='gray')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)

    # Legend
    ax.legend(
        handles=[Patch(color='#1D9E75', label='Explored faster'), Patch(color='#D85A30', label='Native faster')],
        frameon=False, fontsize=10
    )

    plt.tight_layout()
    plt.savefig(ouput_file)

def make_job_cumsum():
    table_df = pd.read_csv("./data/job/table.csv", delimiter='\t', usecols=['Native','Classifier','Carbon'])
    table_df = table_df.cumsum()
    plt.figure()
    table_df.plot()
    plt.savefig('./data/job/vis.pdf')

def make_stats_cumsum():
    table_df = pd.read_csv("./data/stats/table.csv", delimiter='\t', usecols=['Native','Classifier','Carbon'])
    table_df = table_df.cumsum()
    plt.figure()
    table_df.plot()
    plt.savefig('./data/stats/vis.pdf')

def make_tpch_cumsum():
    table_df = pd.read_csv("./data/tpch/table.csv", delimiter='\t', usecols=['Native','Classifier','Carbon'])
    explored_df = pd.read_csv("./data/tpch/explored.csv", delimiter='\t')
    table_df['Explored'] = explored_df['Explored']
    table_df = table_df.sort_values(by='Native')
    table_df = table_df.cumsum()
    table_df['Query'] = range(0, 30)
    table_df = table_df.set_index('Query')
    plt.figure()
    table_df.plot()
    plt.savefig('./data/tpch/vis.pdf')

def main():
    #exploration_graph('./data/job/explored.csv', './data/job/explored.pdf', range(0, 10))

    stats_exploration_queries = pd.read_csv("./data/stats/explored.csv", delimiter='\t', usecols=["Query"])['Query']
    exploration_graph('./data/stats/explored.csv', './data/stats/explored.pdf', list(stats_exploration_queries))

if __name__ == "__main__":
    main()

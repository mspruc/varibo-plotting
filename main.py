import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.close("all")

def main():
    table_df = pd.read_csv("./data/tpch/table.csv", delimiter='\t', index_col='Query')
    sum_classifier = table_df['Classifier'].sum()
    sum_native = table_df['Classifier'].sum()
    sum_retrained = table_df['Classifier'].sum()

    table_df['Classifier % of native'] = (table_df['Classifier'] / table_df['Native'])*100
    table_df['Carbon % of native'] = (table_df['Carbon'] / table_df['Native'])*100
    table_df['Explored % of native'] = (pd.read_csv("./data/tpch/explored.csv", delimiter='\t')['Explored'] / table_df['Native'])*100


    # Y positions
    y = np.arange(len(table_df))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Native baseline (100%)
    ax.barh(
        y,
        [100] * len(table_df),
        color='lightgray',
        height=0.6,
        label='Native (100%)'
    )

    # Overlay bars (thinner)
    ax.barh(
        y,
        table_df['Classifier % of native'],
        height=0.35,
        color='tab:blue',
        label='Classifier'
    )

    ax.barh(
        y,
        table_df['Carbon % of native'],
        height=0.25,
        color='tab:orange',
        label='Carbon'
    )

    ax.barh(
        y,
        table_df['Explored % of native'],
        height=0.15,
        color='tab:green',
        label='Explored'
    )

    # Axes & labels
    ax.set_yticks(y)
    ax.set_yticklabels(table_df.index)
    ax.set_xlabel('% of Native')
    ax.set_title('Performance Relative to Native (Baseline = 100%)')
    ax.legend()

    ax.set_xlim(0, max(120, table_df[['Classifier % of native', 'Carbon % of native', 'Explored % of native']].max().max() * 1.1))

    plt.tight_layout()
    plt.show()

    print(table_df)


def make_cumsum():
    table_df = pd.read_csv("./data/tpch/table.csv", delimiter='\t', usecols=['Native','Classifier','Carbon'])
    explored_df = pd.read_csv("./data/tpch/explored.csv", delimiter='\t')    
    table_df['Explored'] = explored_df['Explored']
    table_df = table_df.sort_values(by='Native')
    table_df = table_df.cumsum()
    table_df['Query'] = range(0, 30)
    table_df = table_df.set_index('Query')
    plt.figure()
    table_df.plot()
    plt.savefig('./data/tpch/vis.png')

if __name__ == "__main__":
    main()
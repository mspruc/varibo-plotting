import pandas as pd
import matplotlib.pyplot as plt

plt.close("all")

def main():
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
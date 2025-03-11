import greedyPolicy as gp
import pandas as pd
import viz
import numpy as np

COLORS, PAGE_WIDTH, ROW_HEIGHT = viz.set_plots()

prices = pd.read_csv("../data/prices.csv", index_col=0, parse_dates=True)
prices = prices.loc[f"2023-01-01":f"2023-12-31"].lmp

START_DATE = "2023-01-01"
END_DATE = "2023-01-15"

k = 10

TotalProfits = []
for i in range(k):
    print(i)
    policy = gp.GreedyPolicy(prices, capacity=0.0)
    results = policy.run()
    TotalProfits.append(results.reward.sum())
    if i == 0:
        viz.plotOperations(
            results, START_DATE, END_DATE, save_path="GreedyResults/greedy_policy.png"
        )
print(f"Mean profit: {pd.Series(TotalProfits).mean()}")
np.save("GreedyResults/greedy_policy.npy", TotalProfits)

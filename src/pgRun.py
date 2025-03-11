from policy_gradient import PolicyGradient
import numpy as np
import pandas as pd
from priceGeneration import *
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def main(nYears=None, nMonths=None, month=[], test=10, NEpisodes=1000, num_batches=100):

    prices = pd.read_csv("../data/prices.csv", index_col=0, parse_dates=True)
    if not (nYears is None):
        pricesTraining = GenerateNYearlyProfiles(nYears).lmp
    elif not (nMonths is None):
        pricesTraining = pd.DataFrame([0], columns=["lmp"])
        for _ in tqdm(range(nMonths)):
            for m in month:
                pricesTraining = pd.concat(
                    [pricesTraining, generateMonthlyProfile(m).lmp]
                )
        pricesTraining = pricesTraining[1:].lmp
    pricesTest = prices.loc[f"2023-01-01":f"2023-01-31"].lmp
    pricesTest = pd.concat([pricesTest[-4:], pricesTest])

    PolicyPG = PolicyGradient(pricesTraining, NEpisodes, num_batches)
    rewards, losses = PolicyPG.run()

    np.save("PGResults/pg_rewards.npy", rewards)
    np.save("PGResults/pg_losses.npy", losses)

    TotalProfits = []
    for i in range(test):
        print(i)
        history = PolicyPG.test(pricesTest)
        TotalProfits.append(history.reward.sum())
        print(f"Total profit: {history.reward.sum()}")
        history.to_csv(f"PGResults/pg_history.csv")
    print(f"Mean profit: {pd.Series(TotalProfits).mean()}")

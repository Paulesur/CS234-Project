from softActorCritic import SAC
import numpy as np
import pandas as pd
from priceGeneration import *
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


def main(nYears=None, nMonths=None, month=[], test=10, ep=10):
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
    PolicySAC = SAC(pricesTraining)
    rewards = PolicySAC.train(episodes=ep)

    np.save("SACResults/sac_rewards.npy", rewards)

    TotalProfits = []
    for i in range(test):
        print(i)
        history = PolicySAC.test(pricesTest)
        TotalProfits.append(history.reward.sum())
        print(f"Profit: {history.reward.sum()}")
        history.to_csv(f"SACResults/sac_history.csv")
    print(f"Mean profit: {pd.Series(TotalProfits).mean()}")

import DQNPolicy as dp
import numpy as np
from priceGeneration import *


def main(nYears=None, nMonths=None, nDays=None, month=[], test=10):
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
    elif not (nDays is None):
        pricesTraining = pd.DataFrame([0], columns=["lmp"])
        for _ in tqdm(range(nDays)):
            for m in month:
                pricesTraining = pd.concat(
                    [pricesTraining, generateDailyProfile(m, 1).lmp]
                )
        pricesTraining = pricesTraining[1:].lmp
    pricesTest = prices.loc[f"2023-01-01":f"2023-01-31"].lmp
    pricesTest = pd.concat([pricesTest[-4:], pricesTest])
    PolicyDQN = dp.DQNPolicy(pricesTraining, prices, network_dict=None)
    rewards, losses = PolicyDQN.train()

    np.save("DQNResults/dqn_rewards.npy", rewards)
    np.save("DQNResults/dqn_losses.npy", losses)

    TotalProfits = []
    for i in range(test):
        print(i)
        history = PolicyDQN.run(pricesTest)
        TotalProfits.append(history.reward.sum())
        print(f"Profit: {history.reward.sum()}")
        history.to_csv(f"DQNResults/dqn_history.csv")
    print(f"Mean profit: {pd.Series(TotalProfits).mean()}")

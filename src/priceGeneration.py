import pandas as pd
import numpy as np
from tqdm import tqdm

prices = pd.read_csv("../data/prices.csv", index_col=0, parse_dates=True)


def generateDailyProfile(month, D):
    year = np.random.randint(2020, 2023)
    day = np.random.randint(
        prices.loc[prices.index.month == month].index.day.min(),
        prices.loc[prices.index.month == month].index.day.max(),
    )
    results = pd.DataFrame(columns=["lmp"])
    results["lmp"] = prices.loc["{}-{:02d}-{:02d}".format(year, month, day)][
        "lmp_rescaled"
    ]
    index = prices.loc[
        (prices.index.year == 2021)
        & (prices.index.month == month)
        & (prices.index.day == D)
    ].index
    return results.set_index(index)


def generateMonthlyProfile(month):
    ndays = prices.loc[
        (prices.index.year == 2021) & (prices.index.month == month)
    ].index.day.max()
    monthPrices = pd.DataFrame([0], columns=["lmp"])
    for d in range(1, ndays + 1):
        monthPrices = pd.concat([monthPrices, generateDailyProfile(month, d)])
    return monthPrices[1:]


def GenerateYearlyProfile():
    yearlyProfile = pd.DataFrame([0], columns=["lmp"])
    for m in range(1, 13):
        yearlyProfile = pd.concat([yearlyProfile, generateMonthlyProfile(m)])
    return yearlyProfile[1:]


def GenerateNYearlyProfiles(n):
    profiles = pd.DataFrame([0], columns=["lmp"])
    for i in tqdm(range(n)):
        profiles = pd.concat([profiles, GenerateYearlyProfile()])
    return profiles[1:]

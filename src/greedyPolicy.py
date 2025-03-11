from Station import EVStation
from Battery import Battery
import pandas as pd
import numpy as np


class GreedyPolicy:
    """
    Implement the GreedyPolicy class:
    - station: Station
    - prices: dataset
    - nSteps: int (number of time steps to run
    -----
    - run(): run the greedy policy
    """

    def __init__(
        self,
        prices,
        Nchargers=6,
        PChargers=0.15,
        capacity=0.6,
    ):
        storage = Battery(capacity, soc=0, max_power=0.15)

        self.station = EVStation(Nchargers, PChargers, storage, 300)
        self.prices = prices
        self.nSteps = len(prices)

    def run(self):
        """
        Run the greedy policy:
        - At each time step, we charge the battery if there is no vehicle to charge and if the battery is not full
        - If there are vehicles to charge, we charge them with the maximum power possible from the battery
        """
        a = self.station.includeEVs()
        # Update the history of the station
        self.station.history = pd.concat(
            [
                self.station.history,
                pd.DataFrame(
                    [[0, self.station.Battery.soc, 0, 0, 0, 0, 0, 0, a, 0, False]],
                    columns=self.station.history.columns,
                ),
            ],
            ignore_index=True,
        )
        for t in range(4, self.nSteps):
            demandEV = [
                (
                    min(
                        self.station.PCharger,
                        4
                        * self.station.EVs[i].capacity
                        * (1 - self.station.EVs[i].soc),
                    )
                    if e is not None
                    else 0
                )
                for i, e in enumerate(self.station.EVs)
            ]
            indicEV = [1 if e is not None else 0 for e in self.station.EVs]
            self.station.state = (
                list(self.prices.iloc[t - 4 : t + 1])
                + [self.station.Battery.soc]
                + [
                    np.sum(
                        [
                            e.soc * e.capacity if e is not None else 1
                            for e in self.station.EVs
                        ]
                    )
                    / np.sum(
                        [e.capacity if e is not None else 1 for e in self.station.EVs]
                    )
                ]
                + [sum(demandEV)]
                + [self.prices.index[t].hour]
            )
            if sum(indicEV) == 0:
                if True:
                    PCharge = max(
                        0,
                        min(
                            self.station.Battery.max_power,
                            4
                            * self.station.Battery.capacity
                            * (1 - self.station.Battery.soc),
                        ),
                    )
                else:
                    PCharge = 0
                # 15 minutes, we charge the battery with the maximum power possible.
                action = [int(PCharge / self.station.Battery.DeltaCharge)] + [
                    0
                ] * self.station.Nchargers
                act, state, reward, done, info = self.station.step(action)
            else:
                demandEV = [
                    (
                        min(
                            self.station.PCharger,
                            4
                            * self.station.EVs[i].capacity
                            * (1 - self.station.EVs[i].soc),
                        )
                        if e is not None
                        else 0
                    )
                    for i, e in enumerate(self.station.EVs)
                ]
                totalDemand = sum(demandEV)
                batteryDischarge = min(
                    self.station.Battery.max_power,
                    totalDemand,
                    4 * self.station.Battery.capacity * self.station.Battery.soc,
                )
                action = [-int(batteryDischarge / self.station.Battery.DeltaCharge)] + [
                    d for d in demandEV
                ]
                act, state, reward, done, info = self.station.step(action)
        self.station.history = self.station.history[1:]  # To remove the first row
        self.station.history.index = self.prices.index[
            4:
        ]  # To match the index of the prices
        self.station.history.to_csv("GreedyResults/greedy_policy.csv")
        return self.station.history

from station import EVStation
from battery import Battery
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--yearI",
    required=True,
    type=int,
    choices=[2020, 2021, 2022, 2023, 2024],
    default=2024,
)
parser.add_argument(
    "--yearF",
    required=True,
    type=int,
    choices=[2020, 2021, 2022, 2023, 2024],
    default=2024,
)

parser.add_argument(
    "--monthI",
    required=True,
    type=int,
    choices=range(1, 13),
    default=1,
)

parser.add_argument(
    "--monthF",
    required=True,
    type=int,
    choices=range(1, 13),
    default=12,
)

parser.add_argument(
    "--dayI",
    required=True,
    type=int,
    choices=range(1, 32),
    default=1,
)

parser.add_argument(
    "--dayF",
    required=True,
    type=int,
    choices=range(1, 32),
    default=31,
)

parser.set_defaults(monthI=1, monthF=12, dayI=1, dayF=31)


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
        Nchargers=5,
        PChargers=[0.15] * 5,
    ):
        storage = Battery(
            capacity=0.4, soc=0, max_charge_power=0.1, max_discharge_power=0.1
        )

        self.station = EVStation(Nchargers, PChargers, storage)
        self.prices = prices
        self.nSteps = len(prices)

    def run(self):
        """
        Run the greedy policy:
        - At each time step, we charge the battery if there is no vehicle to charge and if the battery is not full
        - If there are vehicles to charge, we charge them with the maximum power possible from the battery
        """
        a = self.station.includeNewEVs(0)
        for t in range(self.nSteps):
            indicEV = [1 if e is not None else 0 for e in self.station.EVs]
            if sum(indicEV) == 0:  # no vehicle to charge
                PCharge = max(
                    0,
                    min(
                        self.station.Battery.max_charge_power,
                        4
                        * self.station.Battery.capacity
                        * (1 - self.station.Battery.soc),
                    ),
                )
                # 15 minutes, we charge the battery with the maximum power possible.
                self.station.simulate(
                    t, self.prices.iloc[t], [0] * self.station.Nchargers, PCharge
                )
            else:
                demandEV = [
                    (
                        min(
                            self.station.PCharger[i],
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
                    self.station.Battery.max_discharge_power,
                    totalDemand,
                    4 * self.station.Battery.capacity * self.station.Battery.soc,
                )
                self.station.simulate(
                    t, self.prices.iloc[t], demandEV, -batteryDischarge
                )
        print("Writing to greedy_policy.csv")
        self.station.history.index = self.prices.index
        self.station.history.to_csv("greedy_policy.csv")

        return self.station.history


if __name__ == "__main__":
    args = parser.parse_args()
    prices = pd.read_csv("../data/prices.csv", index_col=0, parse_dates=True)
    prices = prices.loc[
        f"{args.yearI}-{args.monthI}-{args.dayI}" :f"{args.yearF}-{args.monthF}-{args.dayF}"
    ].lmp
    policy = GreedyPolicy(prices)
    policy.run()

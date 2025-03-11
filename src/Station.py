from Battery import Battery
from EV import EV
from GenerateEVs import includeNewEVs
import pandas as pd
import numpy as np

nSteps = 96  # number of time steps in a day (15 minutes each)


class EVStation:
    def __init__(self, Nchargers: int, PCharger: float, Battery: Battery, price: float):
        """
        Initialize the EV station
        """
        self.time = 0  # time of the day, between 0 and nSteps
        self.Nchargers = Nchargers  # number of chargers in the station
        self.PCharger = PCharger  # maximum power of each charger
        self.DeltaPcharger = 0.075  # power increment of each charger, meaning that the power delivered by each charger will be a multiple of this value
        self.Battery = Battery  # battery in the station
        self.EVs = [
            None
        ] * Nchargers  # list of EVs in the station, None if no EV is present
        self.history = pd.DataFrame(
            columns=[
                "time",
                "SOC",
                "load",
                "load Batt",
                "load EVs",
                "cost",
                "reward",
                "rewardTrain",
                "newEVs",
                "price",
                "invalidAction",
            ],
            index=[],
        )  # history of the station
        self.price = price  # price of electricity per MWh at the station
        self.observation_space_size = 9  # size of the observation space, including the state of charge of the battery,
        # the state of charge of the EVs, the capacity of each EV and the price of electricity
        self.action_space_size = (
            2 * int(round(self.Battery.max_power / self.Battery.DeltaCharge, 0)) + 1
        )  # size of the action space, including the power of the battery and the power of each charger)
        self.state = [None, None] + [
            None,
            None,
            None,
        ]  # state of the station, including the state of charge of the battery,

    def reset(self):
        """
        Reset the state of the station
        """
        self.EVs = [None] * self.Nchargers
        self.history = pd.DataFrame(
            columns=[
                "time",
                "SOC",
                "load",
                "load Batt",
                "load EVs",
                "cost",
                "reward",
                "rewardTrain",
                "newEVs",
                "price",
                "invalidAction",
            ],
            index=[],
        )
        self.Battery.reset()

    def addEV(self, EV: EV, i):
        """
        Add an EV to the station: a new EV has arrived on the station
        """
        self.EVs[i] = EV

    def updateEVs(self, PEVs):
        """
        Update the state of charge of the EVs in the station
        """
        for i in range(self.Nchargers):
            if self.EVs[i] is not None:
                a = self.EVs[i].soc
                self.EVs[i].updateSOC(PEVs[i])

    def updateBattery(self, pBattery):
        """
        Update the state of charge of the battery in the station
        """
        self.Battery.updateBattery(pBattery)

    def computeCost(self, price, pBattery, PEVs):
        """
        Compute the cost of electricity for the station at this time step
        """
        if pBattery < 0:
            a = 0.9
        else:
            a = 1
        load = sum(PEVs) + pBattery * a
        return price * load * 1 / 4  # 15 minutes

    def removeChargedEVS(self):
        """
        Remove EVs that are fully charged from the station and compute the cost of charging them
        """
        r = 0
        for i, e in enumerate(self.EVs):
            if e is not None:
                if e.soc >= e.soc_f:
                    r += (e.soc - e.soc_i) * e.capacity * self.price
                    self.EVs[i] = None
        return r

    def includeEVs(self, time=None):
        """
        Include new EVs in the station
        """
        a = 0
        t = self.time if time is None else time
        for i, e in enumerate(self.EVs):
            if e is None:
                if includeNewEVs(t):
                    e = EV()  # new EV has arrived at the station
                    self.EVs[i] = e
                    a += 1
        return a

    def step(self, action):
        """
        Perform one step in the environment
        """
        state = self.state
        Pbatt = self.Battery.DeltaCharge * action[0]
        PChargers = action[1:]
        invalidAction = False
        if (action[0] < 0) and (
            self.Battery.soc
            + action[0] * self.Battery.DeltaCharge / self.Battery.capacity * 0.25
            < 0
        ):
            invalidAction = True  # battery cannot be discharged
            Pbatt = (
                int(
                    -1
                    / 0.25
                    * self.Battery.soc
                    * self.Battery.capacity
                    / self.Battery.DeltaCharge
                )
                * self.Battery.DeltaCharge
            )
            # Set the power of the discharging to what is possible
        if (action[0] > 0) and (
            self.Battery.soc
            + action[0] * self.Battery.DeltaCharge / self.Battery.capacity * 0.25
            > 1
        ):  # battery cannot be charged
            invalidAction = True
            Pbatt = (
                int(
                    1
                    / 0.25
                    * (1 - self.Battery.soc)
                    * self.Battery.capacity
                    / self.Battery.DeltaCharge
                )
                * self.Battery.DeltaCharge
            )
        self.updateEVs(PChargers)
        if Pbatt < 0:
            self.updateBattery(0.9 * Pbatt)
        else:
            self.updateBattery(Pbatt)
        cost = self.computeCost(state[4], Pbatt, PChargers)
        r = self.removeChargedEVS()
        revenue = 0.25 * sum(PChargers) * self.price
        if Pbatt < 0:
            maxP = sum(PChargers) + 0.9 * Pbatt
            Phist = 0.9 * Pbatt
        else:
            maxP = sum(PChargers) + Pbatt
            Phist = Pbatt
        penalty = 1e3 if invalidAction else 0
        if maxP >= 0.41:
            rewardTrain = 1 * revenue - 1 * cost - 1e4 * (maxP - 0.4) ** 2 - penalty
        else:
            rewardTrain = 1 * revenue - 1 * cost - penalty
        reward = revenue - cost
        self.time += 1
        a = self.includeEVs()

        newRow = pd.DataFrame(
            {
                "time": self.time,
                "SOC": self.Battery.soc,
                "load": sum(PChargers) + Phist,
                "load Batt": Phist,
                "load EVs": sum(PChargers),
                "cost": cost,
                "reward": reward,
                "rewardTrain": rewardTrain,
                "newEVs": a,
                "price": state[4],
                "invalidAction": invalidAction,
            },
            index=[self.time],
        )
        new_Demand = [
            (
                min(
                    self.PCharger,
                    4 * self.EVs[i].capacity * (1 - self.EVs[i].soc),
                )
                if e is not None
                else 0
            )
            for i, e in enumerate(self.EVs)
        ]
        self.history = pd.concat([self.history, newRow])
        self.time = self.time % nSteps
        totalCap = sum([e.capacity if e is not None else 1 for e in self.EVs])
        new_state = (
            state[:5]
            + [self.Battery.soc]
            + [
                np.sum([e.soc * e.capacity if e is not None else 1 for e in self.EVs])
                / totalCap
            ]
            + [sum(new_Demand)]
            + [state[-1]]
        )
        return int(Pbatt / self.Battery.DeltaCharge), new_state, rewardTrain, False, {}

    def copy(self):
        s = EVStation(self.Nchargers, self.PCharger, self.Battery.copy(), self.price)
        s.EVs = [e.copy() if e is not None else None for e in self.EVs]
        s.history = self.history.copy()
        return s

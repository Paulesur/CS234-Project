from battery import Battery
from EV import EV
from scipy.stats import beta
import numpy as np
import random
import pandas as pd


nSteps = 360  # number of time steps in a day (15 minutes each)
maxProba = 0.1  # maximum probability of an EV arriving at the station at peak hours


def includeNewEVs(time, a=5, b=3):
    """
    Include new EVs in the station.
    To do so, we will generate a beta distribution that will give the probability of an EV arriving at the station at a given time.
    ----
    Input:
    - time: integer representing the time of the day (between 0 and nSteps)
    """
    # Implement the logic to include new EVs in the station
    a, b = 5, 3
    # Generate a beta distribution
    mu = (
        beta.pdf(time / nSteps, a, b)
        / np.max(beta.pdf(np.linspace(0, 1, nSteps), a, b))
        * maxProba
    )
    return random.random() < mu


class EVStation:
    def __init__(self, Nchargers: int, PCharger: list, Battery: Battery):
        self.time = None
        self.Nchargers = Nchargers
        self.PCharger = PCharger
        self.Battery = Battery
        self.EVs = [None] * Nchargers
        self.history = pd.DataFrame(
            columns=["time", "SOC", "load", "cost", "newEVs"], index=[]
        )
        self.historyEVs = pd.DataFrame(columns=["time", "SOC_init", "Cap"], index=[])

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
                self.EVs[i].updateSOC(min(self.PCharger[i], PEVs[i]))

    def updateBattery(self, pBattery):
        """
        Update the state of charge of the battery in the station
        """
        self.Battery.updateBattery(pBattery)

    def computeCost(self, price, pBattery, PEVs):
        """
        Compute the cost of electricity for the station at this time step
        """
        load = sum(PEVs) + pBattery
        return price * load * 1 / 4  # 15 minutes

    def removeChargedEVS(self):
        """
        Remove EVs that are fully charged from the station
        """
        for i, e in enumerate(self.EVs):
            if e is not None:
                if e.soc >= e.soc_f:
                    self.EVs[i] = None  # EV leaves the station

    def includeNewEVs(self, time):
        """
        Include new EVs in the station
        """
        a = 0
        for i, e in enumerate(self.EVs):
            if e is None:
                if includeNewEVs(time):
                    e = EV()  # new EV has arrived at the station
                    self.EVs[i] = e
                    newRow = pd.DataFrame(
                        {
                            "time": time,
                            "SOC_init": e.soc,
                            "Cap": e.capacity,
                        },
                        index=[len(self.historyEVs)],
                    )
                    self.historyEVs = pd.concat([self.historyEVs, newRow])
                    a += 1
        return a

    def simulate(self, time, price, PChargers, Pbatt):
        """
        Simulate the operation of the station for one step
        """
        if time == 0:
            b = 0
        else:
            b = self.history["newEVs"].iloc[-1]
        self.updateEVs(PChargers)
        self.updateBattery(Pbatt)
        cost = self.computeCost(price, Pbatt, PChargers)
        self.removeChargedEVS()
        a = self.includeNewEVs(time % 360)
        newRow = pd.DataFrame(
            {
                "time": time,
                "SOC": self.Battery.soc,
                "load": sum(PChargers) + Pbatt,
                "cost": cost,
                "newEVs": a + b,
            },
            index=[time],
        )
        self.history = pd.concat([self.history, newRow])

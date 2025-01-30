import numpy as np
import random


def chooseRandomSOCInit():
    """
    Choose a random initial state of charge for the EV when it arrives at the station
    """
    a, b = 1.2, 7
    return np.random.beta(a, b, 1)


def chooseRandomCapacity():
    """
    Choose a random capacity for the EV
    """
    return random.uniform(0.02, 0.06)


def chooseRandomSOC_f(soc_i):
    """
    Choose a random state of charge for the EV after which it will leave the station
    """
    return random.uniform(max(0.85, soc_i), 1)


class EV:
    """
    Implement the EV class:
    - name: string
    - capacity: float
    - soc: float

    Methods:
    - updateSOC(P): update the state of charge of the EV
    """

    def __init__(self):
        self.capacity = chooseRandomCapacity()
        self.soc = chooseRandomSOCInit()
        self.soc_f = chooseRandomSOC_f(self.soc)

    def updateSOC(self, P):
        self.soc = max(0, min(1, self.soc + P / (4 * self.capacity)))  # 15 minutes

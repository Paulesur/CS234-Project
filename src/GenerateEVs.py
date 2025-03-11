from scipy.stats import beta
import numpy as np
import random

nSteps = 96  # number of time steps in a day (15 minutes each)
maxProba = 0.2  # maximum probability of an EV arriving at the station at peak hours


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


def chooseRandomSOCInit():
    """
    Choose a random initial state of charge for the EV when it arrives at the station
    """
    a, b = 1.2, 7
    return np.random.beta(a, b, 1)[0]


def chooseRandomCapacity():
    """
    Choose a random capacity for the EV (in MWh)
    """
    return random.uniform(0.02, 0.06)  # between 20 and 60 kWh


def chooseRandomSOC_f(soc_i):
    """
    Choose a random state of charge for the EV after which it will leave the station
    """
    return random.uniform(max(0.85, soc_i), 1)  # between 85% and 100%

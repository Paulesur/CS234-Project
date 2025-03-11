from DQN import DQN
import numpy as np
import torch
import pandas as pd
from Station import EVStation
from Battery import Battery


class DQNPolicy:

    def __init__(
        self,
        pricesTrain,
        pricesTest,
        Nchargers=6,
        PChargers=0.15,
        network_dict=None,
    ):
        storage = Battery(capacity=0.8, soc=0, max_power=0.2)

        self.station = EVStation(Nchargers, PChargers, storage, 300)

        self.pricesTrain = pricesTrain
        self.pricesTest = pricesTest
        self.DQN = DQN(pricesTrain, self.station, 1, 7)
        if network_dict is not None:
            network_dict = torch.load(network_dict)
            self.DQN.mainNetwork.load_state_dict(network_dict)
            self.DQN.targetNetwork.load_state_dict(network_dict)

    def train(self):
        rewardsEp, losses = self.DQN.train()
        return rewardsEp, losses

    def run(self, prices):
        self.DQN.mainNetwork.load_state_dict(torch.load("DQNResults/dqn_model.pth"))
        self.DQN.targetNetwork.load_state_dict(torch.load("DQNResults/dqn_model.pth"))
        rewards = self.DQN.test(prices)
        return rewards

    def test(self):
        Rewards = self.DQN.test(self.pricesTest)
        return Rewards

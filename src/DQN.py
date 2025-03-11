import numpy as np
import torch
from collections import deque
from network_utils import build_mlp
import pandas as pd
import random
from tqdm import tqdm
from copy import deepcopy


class DQN:

    def __init__(self, prices, station, epsilon, lenEpisodes, lr=0.001):
        self.station = station
        self.gamma = 0.999
        self.epsilon = epsilon
        self.lr = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon_decay = 0.97  # 0.95
        self.epsilon_min = 0.001
        self.crit = torch.nn.MSELoss()
        self.prices = prices
        self.nSteps = len(prices)

        # Initialize the Q-network and target network
        self.stateDimension = station.observation_space_size
        self.actionDimension = station.action_space_size
        self.replayBuffer_size = 3000
        self.replayBuffer = deque(maxlen=self.replayBuffer_size)
        self.catastrophicBuffer = deque(maxlen=self.replayBuffer_size)
        self.batchReplaySize = 300
        self.updateNetworkParams = int(96 * 7)
        self.updateEpsilon = 96 * 7
        self.mainNetwork = build_mlp(self.stateDimension, self.actionDimension, 2, 64)
        self.targetNetwork = build_mlp(self.stateDimension, self.actionDimension, 2, 64)
        self.optimizer = torch.optim.Adam(self.mainNetwork.parameters(), lr=self.lr)
        self.targetNetwork = deepcopy(self.mainNetwork)
        # try:
        #     self.targetNetwork.load_state_dict(
        #         torch.load("DQNResults/dqn_model.pth", map_location=self.device)
        #     )
        # except:
        #     pass

    def selectAction(self, state):
        if np.random.rand() <= self.epsilon:
            a = np.random.choice(self.actionDimension)
            return a
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.mainNetwork(state)
            a = torch.argmax(q_values).item()
            return a

    def trainNetwork(self):
        if len(self.replayBuffer) < 96 * 10:  # Get one month of random actions
            return 0
        else:
            batch = random.sample(self.replayBuffer, self.batchReplaySize - 1)
            batch.append(self.replayBuffer[-1])  # Always add the last element
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device).unsqueeze(1)
        rewards = (
            torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        )
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.mainNetwork(states).gather(1, actions)
        next_q_values = self.targetNetwork(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values.unsqueeze(1)
        self.optimizer.zero_grad()
        loss = self.crit(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
        return self.crit(q_values, target_q_values).item()

    def train(self):
        self.station.reset()
        a = self.station.includeEVs()
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
        rewardsEp = []
        losses = []
        r = 0
        for t in tqdm(range(4, self.nSteps)):
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
            previous_prices = self.prices.iloc[t - 4 : t + 1]
            self.station.state = (
                list(previous_prices)
                + [self.station.Battery.soc]
                + [
                    np.sum(
                        [
                            (e.soc * e.capacity if e is not None else 1)
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
            actionbatt = self.selectAction(self.station.state)
            actions = [actionbatt - self.actionDimension // 2] + [0] * (
                self.station.Nchargers
            )

            actions[1:] = demandEV
            a, state, reward, done, info = self.station.step(actions)
            r += reward
            if t % 96 == 0:
                rewardsEp.append(r)
                r = 0
            if t < self.nSteps - 2:
                state[:5] = list(self.prices.iloc[t + 1 - 4 : t + 2])
                state[-1] = self.prices.index[t + 1].hour
            else:
                state[:5] = list(self.prices.iloc[-5:])
                state[-1] = self.prices.index[-1].hour
                done = True
            if self.replayBuffer_size > len(self.replayBuffer):
                self.replayBuffer.append(
                    (
                        self.station.state,
                        actionbatt,
                        reward,
                        state,
                        done,
                    )
                )
            else:
                self.replayBuffer.popleft()
                self.replayBuffer.append(
                    (
                        self.station.state,
                        actionbatt,
                        reward,
                        state,
                        done,
                    )
                )
            self.station.state = state
            if t % 1 == 0:
                l = self.trainNetwork()
                losses.append(l)
            if t % self.updateNetworkParams == 0:
                self.targetNetwork = deepcopy(self.mainNetwork)
            if t % self.updateEpsilon == 0 and self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_decay
        # Save the model
        torch.save(self.mainNetwork.state_dict(), "DQNResults/dqn_model.pth")
        return rewardsEp, losses

    def test(self, testPrices):
        self.mainNetwork.load_state_dict(torch.load("DQNResults/dqn_model.pth"))
        self.epsilon = 0
        RewardReturn = 0
        self.station.reset()
        a = self.station.includeEVs()
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
        for t in range(4, len(testPrices)):
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
            totalCap = np.sum(
                [e.capacity if e is not None else 1 for e in self.station.EVs]
            )
            self.station.state = (
                list(testPrices.iloc[t - 4 : t + 1])
                + [self.station.Battery.soc]
                + [
                    np.sum(
                        [
                            (e.soc * e.capacity if e is not None else 1)
                            for e in self.station.EVs
                        ]
                    )
                    / totalCap
                ]
                + [sum(demandEV)]
                + [testPrices.index[t].hour]
            )
            actionbatt = self.selectAction(self.station.state)
            actions = [actionbatt - self.actionDimension // 2] + [0] * (
                self.station.Nchargers
            )

            actions[1:] = demandEV
            a, state, reward, done, info = self.station.step(actions)
            RewardReturn += reward
            if t < len(testPrices) - 1:
                state[:5] = list(testPrices.iloc[t + 1 - 4 : t + 2])
                state[-1] = testPrices.index[t + 1].hour
            else:
                state[:5] = list(testPrices.iloc[-4:])
                state[-1] = testPrices.index[-1].hour
                done = True
            self.station.state = state

        self.station.history = self.station.history[1:]  # To remove the first row
        self.station.history.index = testPrices.index[
            4:
        ]  # To match the index of the prices
        return self.station.history

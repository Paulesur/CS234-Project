import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from policy import CategoricalPolicy
from network_utils import build_mlp, np2torch
from copy import deepcopy
from Battery import Battery
from EV import EV
from Station import EVStation
from tqdm import tqdm
from collections import deque


class SAC:
    def __init__(
        self,
        prices,
        lr=0.001,
        gamma=0.999,
        tau=0.005,
        batch_size=100,
        buffer_size=100000,
    ):
        storage = Battery(capacity=0.8, soc=0, max_power=0.2)

        self.station = EVStation(6, 0.15, storage, 300)
        self.Ndays = 3  # 3
        self.prices = prices
        self.stateDimension = self.station.observation_space_size
        self.actionDimension = self.station.action_space_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.buffer_counter = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_entropy = 0.6 * (-np.log(1 / self.actionDimension))

        # temperature parameter
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.actor = CategoricalPolicy(
            build_mlp(self.stateDimension, self.actionDimension, 2, 64)
        )
        self.critic1 = build_mlp(self.stateDimension, self.actionDimension, 2, 50)
        self.critic2 = build_mlp(self.stateDimension, self.actionDimension, 2, 50)
        self.target_critic1 = build_mlp(
            self.stateDimension, self.actionDimension, 2, 50
        )
        self.target_critic2 = build_mlp(
            self.stateDimension, self.actionDimension, 2, 50
        )

        self.actor_optimizer = optim.Adam(self.actor.network.parameters(), lr=0.0001)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.001)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.001)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.0001)

        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)

    def select_action(self, state):
        state = np2torch(np.array(state))
        with torch.no_grad():
            distribution = self.actor.action_distribution(state)
            sampled_actions = distribution.sample()
            log_probs = distribution.log_prob(sampled_actions)
            return sampled_actions, log_probs

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if self.buffer_counter < self.buffer_size:
            self.buffer_counter += 1
        else:
            self.buffer.popleft()

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def update(self):

        if len(self.buffer) < 96 * 31:
            return

        batch = np.random.choice(len(self.buffer), self.batch_size, replace=True)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in batch]
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            dist = self.actor.action_distribution(next_states)
            probs = dist.probs
            logpis = torch.log(probs + 1e-8)
            probs = torch.exp(logpis)
            Q_target1 = self.target_critic1(next_states)
            Q_target2 = self.target_critic2(next_states)
            Q_target = torch.min(Q_target1, Q_target2)
            nextV = (probs * (Q_target - self.alpha * logpis)).sum(-1).unsqueeze(-1)
            Q_target_next = (
                rewards
                + self.gamma
                * (probs * (Q_target - self.alpha * logpis)).sum(-1).unsqueeze(-1)
            ).float()

        Q1 = self.critic1(states).gather(1, actions.unsqueeze(1)).float().squeeze(1)
        Q2 = self.critic2(states).gather(1, actions.unsqueeze(1)).float().squeeze(1)

        critic1_loss = nn.MSELoss()(Q1, Q_target_next.detach())

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        critic2_loss = nn.MSELoss()(Q2, Q_target_next.detach())
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        dist = self.actor.action_distribution(states)
        probs = dist.probs
        logpis = torch.log(probs + 1e-8)
        with torch.no_grad():
            actorQ1, actorQ2 = self.critic1(states), self.critic2(states)
            minQ = torch.min(actorQ1, actorQ2)

        self.actor_optimizer.zero_grad()
        actor_loss = (probs * (self.alpha.detach() * logpis - minQ)).sum(-1).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        log_probs = (probs * logpis).sum(-1)
        self.alpha_optimizer.zero_grad()
        alpha_loss = -(
            self.log_alpha * (log_probs.detach() + self.target_entropy)
        ).mean()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def train(self, episodes):

        scores = []
        for episode in tqdm(range(episodes)):
            day = np.random.randint(
                1, len(self.prices) // 96 - self.Ndays
            )  # Choose a random day to simulate the episode
            first_hour = np.random.randint(0, 96)
            score = 0
            station = EVStation(6, 0.15, Battery(0.6, 0, 0.15), 300)
            station.time = first_hour
            a = station.includeEVs(time=first_hour)
            station.history = pd.concat(
                [
                    station.history,
                    pd.DataFrame(
                        [[0, station.Battery.soc, 0, 0, 0, 0, 0, 0, a, 0, False]],
                        columns=station.history.columns,
                    ),
                ],
                ignore_index=True,
            )
            soc_i = np.random.choice(
                [
                    k
                    * 0.25
                    * self.station.Battery.DeltaCharge
                    / self.station.Battery.capacity
                    for k in range(
                        int(
                            round(
                                self.station.Battery.capacity
                                / (0.25 * self.station.Battery.DeltaCharge),
                                0,
                            )
                        )
                    )
                ]
            )
            soc_i = np.clip(soc_i, 0, 1)
            station.Battery.soc = soc_i  # Randomize the initial SOC
            pricesTrain = self.prices.iloc[
                day * 96
                + first_hour
                - 4 : min((day + self.Ndays) * 96 + first_hour, len(self.prices))
            ]
            for t in range(4, len(pricesTrain)):
                demandEV = [
                    (
                        min(
                            station.PCharger,
                            4 * station.EVs[i].capacity * (1 - station.EVs[i].soc),
                        )
                        if e is not None
                        else 0
                    )
                    for i, e in enumerate(station.EVs)
                ]
                station.state = (
                    list(pricesTrain.iloc[t - 4 : t + 1])
                    + [station.Battery.soc]
                    + [
                        np.sum(
                            [
                                (e.soc * e.capacity if e is not None else 1)
                                for e in station.EVs
                            ]
                        )
                        / np.sum(
                            [e.capacity if e is not None else 1 for e in station.EVs]
                        )
                    ]
                    + [sum(demandEV)]
                    + [pricesTrain.index[t].hour]
                )
                state = station.state
                actionbatt, _ = self.select_action(state)
                actionbatt = actionbatt.item()
                actions = [actionbatt - self.actionDimension // 2] + [0] * (
                    self.station.Nchargers
                )

                actions[1:] = demandEV
                a, next_state, reward, done, _ = station.step(actions)
                if t < len(pricesTrain) - 2:
                    next_state[:5] = list(pricesTrain.iloc[t + 1 - 4 : t + 2])
                    next_state[-1] = pricesTrain.index[t + 1].hour
                else:
                    next_state[:5] = list(pricesTrain.iloc[-5:])
                    next_state[-1] = pricesTrain.index[-1].hour
                    done = True
                score += reward
                self.store_transition(state, actionbatt, reward, next_state, done)
                state = next_state
                self.update()
                if t % 5 == 0:
                    for target_param, param in zip(
                        self.target_critic1.parameters(), self.critic1.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )
                    for target_param, param in zip(
                        self.target_critic2.parameters(), self.critic2.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )
            if episode % 20 == 0:  # hard update after 10 episodes
                for target_param, param in zip(
                    self.target_critic1.parameters(), self.critic1.parameters()
                ):
                    target_param.data.copy_(
                        0.95 * param.data + (1 - 0.95) * target_param.data
                    )
                for target_param, param in zip(
                    self.target_critic2.parameters(), self.critic2.parameters()
                ):
                    target_param.data.copy_(
                        0.95 * param.data + (1 - 0.95) * target_param.data
                    )
            scores.append(score / self.Ndays)

        torch.save(self.actor.network.state_dict(), "SACResults/actor_model.pth")
        torch.save(self.critic1.state_dict(), "SACResults/critic1_model.pth")
        torch.save(self.critic2.state_dict(), "SACResults/critic2_model.pth")
        torch.save(
            self.target_critic1.state_dict(), "SACResults/target_critic1_model.pth"
        )
        torch.save(
            self.target_critic2.state_dict(), "SACResults/target_critic2_model.pth"
        )
        np.save("SACResults/scores.npy", scores)
        return scores

    def test(self, testPrices):
        self.actor.network.load_state_dict(torch.load("SACResults/actor_model.pth"))
        self.critic1.load_state_dict(torch.load("SACResults/critic1_model.pth"))
        self.critic2.load_state_dict(torch.load("SACResults/critic2_model.pth"))
        self.target_critic1.load_state_dict(
            torch.load("SACResults/target_critic1_model.pth")
        )
        self.target_critic2.load_state_dict(
            torch.load("SACResults/target_critic2_model.pth")
        )
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
            actionbatt = self.select_action(self.station.state)[0].item()
            actions = [actionbatt - self.actionDimension // 2] + [0] * (
                self.station.Nchargers
            )

            actions[1:] = demandEV
            act, state, reward, done, info = self.station.step(actions)
            if t < len(testPrices) - 2:
                state[:5] = testPrices.iloc[t - 4 + 1 : t + 2]
                state[-1] = testPrices.index[t + 1].hour
            else:
                state[:5] = testPrices.iloc[t - 4 : t + 1]
                state[-1] = testPrices.index[-1].hour
                done = True
            self.station.state = state

        self.station.history = self.station.history[1:]  # To remove the first row
        self.station.history.index = testPrices.index[
            4:
        ]  # To match the index of the prices
        return self.station.history

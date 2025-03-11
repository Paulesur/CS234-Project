from tqdm import tqdm
import numpy as np
import torch
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy
from Station import EVStation
from Battery import Battery
import pandas as pd


class PolicyGradient:
    """
    Class for implementing a policy gradient algorithm
    """

    def __init__(self, pricesTrain, NEpisodes, num_batches, NChargers=6, PCharger=0.15):
        """
        Initialize Policy Gradient Class

        Args:
            pricesTrain: training prices
            NEpisodes: number of episodes per batch
            num_batches: number of batches
            NChargers: number of chargers in the station
            PCharger: power of each charger
        """
        # directory for training outputs
        storage = Battery(capacity=0.8, soc=0, max_power=0.2)

        self.station = EVStation(NChargers, PCharger, storage, 300)

        self.prices = pricesTrain
        self.len_episode = 1

        self.observation_dim = self.station.observation_space_size
        self.action_dim = self.station.action_space_size

        self.lr = 0.0001
        self.gamma = 0.999
        self.NEpisodes = NEpisodes
        self.num_batches = num_batches

        self.init_policy()
        self.baseline_network = BaselineNetwork(self.observation_dim, 0.001, 2, 50)

    def init_policy(self):
        """
        Initialize the policy network
        """
        network = build_mlp(self.observation_dim, self.action_dim, 2, 64)
        self.policy = CategoricalPolicy(network)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.0001)

    def init_averages(self):
        self.avg_reward = 0.0
        self.max_reward = 0.0
        self.std_reward = 0.0
        self.eval_reward = 0.0

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.
        You don't have to change or use anything here.

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def sample_path(self, num_episodes=None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"
        """
        episode = 0
        episode_rewards = []
        paths = []
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
        day = np.random.randint(
            1, len(self.prices) // 96 - self.len_episode
        )  # Choose a random day to simulate the episode
        first_hour = np.random.randint(0, 96)
        prices = self.prices.iloc[
            day * 96
            - 4
            + first_hour : min(
                (day + self.len_episode) * 96 + first_hour, len(self.prices)
            )
        ]
        for i in range(num_episodes):
            self.station.reset()
            self.station.Battery.soc = soc_i
            self.station.time = first_hour
            a = self.station.includeEVs(time=first_hour)
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

            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(4, len(prices)):
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
                self.station.state = (
                    list(prices.iloc[step - 4 : step + 1])
                    + [self.station.Battery.soc]
                    + [
                        np.sum(
                            [
                                e.soc * e.capacity if e is not None else 1
                                for e in self.station.EVs
                            ]
                        )
                        / np.sum(
                            [
                                e.capacity if e is not None else 1
                                for e in self.station.EVs
                            ]
                        )
                    ]
                    + [sum(demandEV)]
                    + [prices.index[step].hour]
                )
                states.append(self.station.state)
                actionBatt = self.policy.act(np.array(self.station.state))[0]
                actionStep = [actionBatt - self.action_dim // 2] + [0] * (
                    self.station.Nchargers
                )

                actionStep[1:] = demandEV
                _, _, reward, _, _ = self.station.step(actionStep)
                actions.append(actionBatt)
                rewards.append(reward)
                episode_reward += reward
            episode_rewards.append(episode_reward)
            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
            }
            paths.append(path)
            episode += 1
        return paths, episode_rewards

    def get_returns(self, paths):
        """
        Calculate the returns G_t for each timestep

        Args:
            paths: recorded sample paths. See sample_path() for details.

        Return:
            returns: return G_t for each timestep

        After acting in the environment, we record the observations, actions, and
        rewards. To get the advantages that we need for the policy update, we have
        to convert the rewards into returns, G_t, which are themselves an estimate
        of Q^π (s_t, a_t):

           G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T
        """

        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = np.zeros_like(rewards)
            for i in range(len(rewards)):
                returns[i] = sum(
                    [self.gamma ** (j - i) * rewards[j] for j in range(i, len(rewards))]
                )
            all_returns.append(returns)
        returns = np.concatenate(all_returns)
        return returns

    def calculate_advantage(self, returns, observations):
        """
        Calculates the advantage for each of the observations
        Args:
            returns: np.array of shape [batch size]
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        # override the behavior of advantage by subtracting baseline
        advantages = self.baseline_network.calculate_advantage(returns, observations)
        return advantages

    def update_policy(self, observations, actions, advantages):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size]
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        self.optimizer.zero_grad()
        distribution = self.policy.action_distribution(observations)
        log_probs = distribution.log_prob(actions)
        loss = -(log_probs * advantages).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        """
        Performs training
        """

        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration
        self.losses = []
        for t in tqdm(range(self.num_batches)):
            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.NEpisodes)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])

            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)

            # advantage will depend on the baseline implementation
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            self.baseline_network.update_baseline(returns, observations)
            l = self.update_policy(observations, actions, advantages)
            self.losses.append(l)

            # logging
            if t % 1 == 0:
                self.update_averages(total_rewards, all_total_rewards)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            averaged_total_rewards.append(avg_reward)
        torch.save(self.policy.network.state_dict(), "PGResults/pg_model.pth")
        torch.save(
            self.baseline_network.network.state_dict(), "PGResults/baseline_model.pth"
        )
        export_plot(averaged_total_rewards, "Score", "Test", "PGResults/score.png")
        return averaged_total_rewards, self.losses

    def evaluate(self, env=None, num_episodes=1):
        """
        Evaluates the return for num_episodes episodes.
        """
        if env == None:
            env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # model
        reward, losses = self.train()
        return reward, losses

    def test(self, pricesTest):
        """
        Test the model
        """
        self.policy.network.load_state_dict(torch.load("PGResults/pg_model.pth"))
        self.baseline_network.network.load_state_dict(
            torch.load("PGResults/baseline_model.pth")
        )
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
        for t in range(4, len(pricesTest)):
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
            totalCap = sum(
                [e.capacity if e is not None else 1 for e in self.station.EVs]
            )
            self.station.state = (
                list(pricesTest.iloc[t - 4 : t + 1])
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
                + [pricesTest.index[t].hour]
            )
            actionBatt = self.policy.act(np.array(self.station.state))[0]
            actionStep = [actionBatt - self.action_dim // 2] + [0] * (
                self.station.Nchargers
            )

            actionStep[1:] = demandEV
            act, state, reward, done, info = self.station.step(actionStep)
            RewardReturn += reward
        self.station.history = self.station.history.iloc[1:]
        self.station.history.index = pricesTest.index[4:]
        self.station.history.to_csv("PGResults/PGHistory.csv")
        return self.station.history

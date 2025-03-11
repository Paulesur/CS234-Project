import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from general import export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy
from policy_gradient import PolicyGradient


class PPO(PolicyGradient):

    def __init__(self, pricesTrain, NEpsiodes, numBatches, NChargers=6, PCharger=0.15):
        super(PPO, self).__init__(pricesTrain, NChargers, PCharger)
        self.eps_clip = 0.5
        self.len_episode = 1
        self.NEpisodes = NEpsiodes
        self.num_batches = numBatches

    def update_policy(self, observations, actions, advantages, old_logprobs):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size, 1]
            old_logprobs: np.array of shape [batch size]

        Perform one update on the policy using the provided data using the PPO clipped
        objective function.
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        old_logprobs = np2torch(old_logprobs)
        log_probs = self.policy.action_distribution(observations).log_prob(actions)
        ratio = torch.exp(log_probs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

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

            states, actions, old_logprobs, rewards = [], [], [], []
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
                actionBatt, old_logprob = self.policy.act(np.array(self.station.state))
                actionStep = [actionBatt - self.action_dim // 2] + [0] * (
                    self.station.Nchargers
                )

                actionStep[1:] = demandEV
                _, _, reward, _, _ = self.station.step(actionStep)
                actions.append(actionBatt)
                old_logprobs.append(old_logprob)
                rewards.append(reward)
                episode_reward += reward
            episode_rewards.append(episode_reward)
            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "old_logprobs": np.array(old_logprobs),
            }
            paths.append(path)
            episode += 1
        return paths, episode_rewards

    def train(self):
        """
        Performs training

        You do not have to change or use anything here, but take a look
        to see how all the code you've written fits together!
        """

        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration
        self.losses = []  # the loss values for each iteration
        for t in tqdm(range(self.num_batches)):
            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.NEpisodes)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            old_logprobs = np.concatenate([path["old_logprobs"] for path in paths])

            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            for _ in range(1):
                self.baseline_network.update_baseline(returns, observations)
                l = self.update_policy(observations, actions, advantages, old_logprobs)
                self.losses.append(l)

            self.update_averages(total_rewards, all_total_rewards)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            averaged_total_rewards.append(avg_reward)
        torch.save(self.policy.network.state_dict(), "PPOResults/ppo_model.pth")
        torch.save(
            self.baseline_network.network.state_dict(), "PPOResults/baseline_model.pth"
        )
        np.save("PPOResults/scores.npy", averaged_total_rewards)
        export_plot(averaged_total_rewards, "Score", "Test", "PPOResults/score.png")
        return averaged_total_rewards, self.losses

    def test(self, pricesTest):
        """
        Test the model
        """
        RewardReturn = 0
        self.station.reset()
        self.policy.network.load_state_dict(torch.load("PPOResults/ppo_model.pth"))
        self.baseline_network.network.load_state_dict(
            torch.load("PPOResults/baseline_model.pth")
        )
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
            self.station.state = (
                list(pricesTest.iloc[t - 4 : t + 1])
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
                + [pricesTest.index[t].hour]
            )
            actionBatt = self.policy.act(np.array(self.station.state))[0]
            actionStep = [actionBatt - self.action_dim // 2] + [0] * (
                self.station.Nchargers
            )

            actionStep[1:] = demandEV
            a, state, reward, done, info = self.station.step(actionStep)
            RewardReturn += reward
        self.station.history = self.station.history.iloc[1:]
        self.station.history.index = pricesTest.index[4:]
        self.station.history.to_csv("PPOResults/PPOhistory.csv")
        return self.station.history

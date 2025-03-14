# CS234-Project

This repository contains the code used for the CS234 project `Fast EV charging station Operations with RL algorithms`.

The data includes four years of electricity prices at a node in the middle of San Francisco, CA. These prices were retrieved using the `GridStatus.io` API [1].

The code for the EV station environment and the four Reinforcement Algorithms used in the project (Deep Q-learning, Policy Gradient, Proximal Policy Optimization and Soft-Actor Critic) can be found in the *src* folder. For each of these algorithm, there is a Jupyter notebook that can be run to train the policies and visualize the optimized policy. The lines to start the policy training have been commented but the optimal policies have been saved in the *Results* folders. The `Greedy` Notebook is used to run the greedy policy, which corresponds to a policy of charging the battery whenever it is possible and discharging it when needed. This policy is used as a baseline to compare the results of the other algorithm.

![Scheme](https://github.com/user-attachments/assets/7ca67166-fc0b-48b5-ae56-bc296bdfe6b6)

[1] GridStatus. URL https://github.com/gridstatus/gridstatus.

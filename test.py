from environment import Environment
from policy import to_action
from algorithm import PPO
from model import PolicyNetwork, ValueNetwork
from value_function import train_value, entropy, GAE
import matplotlib.pyplot as plt
import numpy as np
import copy

def clip(rate, eps):
    if rate < (1-eps):
        return (1-eps)
    elif rate > (1+eps):
        return (1+eps)
    else:
        return rate

plt.ion()

env = Environment()
state = env.reset()
policy_network = PolicyNetwork(len(state), 2)
value_network = ValueNetwork(len(state), 1)
done = False
iter = 10
theta = 0.9
eps = 0.2
epochs = 30
old_network = copy.deepcopy(policy_network)

for epoch in range(epochs):
    data = []
    for _ in range(iter):
        state = env.reset()
        done = False

        while not done:
            action = to_action(state, policy_network)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated

            env.render()
            
            state = next_state


env.close()

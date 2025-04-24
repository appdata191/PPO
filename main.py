from environment import Environment
from policy import to_action
from algorithm import PPO
from model import PolicyNetwork, ValueNetwork
from value_function import train_value, entropy, GAE
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2

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
epochs = 100
old_network = copy.deepcopy(policy_network)

for epoch in range(epochs):
    data = []
    for _ in range(iter):
        state = env.reset()
        done = False

        trajectory = []
        while not done:
            action = env.env.action_space.sample()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated

            #trajectory.append((state, action, reward, next_state, done))

            env.render()
            
            state = next_state

            #temp = GAE(trajectory, value_network)
            #data.extend(temp)


    #old_network = copy.deepcopy(policy_network)
    #loss_iter = PPO(data, policy_network, eps)

    #train_value(value_network, data, 5)
    #print(f"it's the {epoch}")
env.close()

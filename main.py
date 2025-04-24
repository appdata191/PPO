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
epochs = 100
old_network = copy.deepcopy(policy_network)

fig, ax = plt.subplots(3,3, figsize=(12,8))
ax[0,0].set_title("Action")
ax[0,1].set_title("Value function")
ax[0,2].set_title("Reward")
ax[1,0].set_title("Entropy")
ax[1,1].set_title("Advantage")
ax[1,2].set_title("Rate change")
ax[2,0].set_title("Esperance")
ax[2,1].set_title("Record esperance")
ax[2,2].set_title("loss")

plt.subplots_adjust(wspace=0.5, hspace=0.5)

line1, = ax[0,0].plot([],[],color='r', label="stick")
line2, = ax[0,0].plot([],[],color='b', label="hit")
line3, = ax[0,1].plot([],[])
line4, = ax[0,2].plot([],[],color='c')
line5, = ax[1,0].plot([],[])
line6, = ax[1,1].plot([],[],color='m')
line7, = ax[1,2].plot([],[])
line8, = ax[2,0].plot([],[], color='y')
line9, = ax[2,1].plot([],[])
line10, = ax[2,2].plot([],[])

game = 0
esperance_iter = []
record_esperance = 0
record_esperance_iter = []
record_step = 0
record_step_iter = []

for epoch in range(epochs):
    data = []
    for _ in range(iter):
        game += 1
        state = env.reset()
        done = False

        reward_iter=[]
        y1_policy = []
        y2_policy = []
        y_value = []
        entropy_iter = []
        advantage_iter = []
        rate_iter = []
        trajectory = []
        step = 0
        while not done:
            action = to_action(state, policy_network)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated

            trajectory.append((state, action, reward, next_state, done))

            env.render()

            entropy_iter.append(entropy(policy_network(state)))

            y1_policy.append(policy_network(state)[0].item())
            y2_policy.append(policy_network(state)[1].item())
            y_value.append(value_network(state).item())

            reward_iter.append(reward)

            rate = policy_network(state)[action].item()/old_network(state)[action].item()
            rate_iter.append(rate)
            
            state = next_state

            step += 1
        temp = GAE(trajectory, value_network)
        data.extend(temp)

        esperance = 0
        for state, action, advantage, reward, next_state, done in temp:
            policy_new = policy_network(state)[action].item()
            policy_old = old_network(state)[action].item()
            ratio = policy_new/policy_old
            esperance += min(ratio*advantage, clip(ratio, eps))

            advantage_iter.append(advantage)

        esperance = esperance/step
        record_esperance = max(record_esperance, esperance)
        esperance_iter.append(esperance)
        record_esperance_iter.append(record_esperance)

        if (step>record_step):
            record_step = step
        record_step_iter.append(record_step)

        line1.set_xdata(np.arange(step))
        line1.set_ydata(y1_policy)

        line2.set_xdata(np.arange(step))
        line2.set_ydata(y2_policy)

        line3.set_xdata(np.arange(step))
        line3.set_ydata(y_value)

        line5.set_xdata(np.arange(step))
        line5.set_ydata(entropy_iter)

        line6.set_xdata(np.arange(step))
        line6.set_ydata(advantage_iter)

        line7.set_xdata(np.arange(step))
        line7.set_ydata(rate_iter)

        ax[0,0].relim()
        ax[0,0].autoscale_view()

        ax[0,1].relim()
        ax[0,1].autoscale_view()

        line4.set_xdata(np.arange(game))
        line4.set_ydata(record_step_iter)

        line8.set_xdata(np.arange(game))
        line8.set_ydata(esperance_iter)

        line9.set_xdata(np.arange(game))
        line9.set_ydata(record_esperance_iter)

        ax[0,2].relim()
        ax[0,2].autoscale_view()

        ax[1,0].relim()
        ax[1,0].autoscale_view()

        ax[1,1].relim()
        ax[1,1].autoscale_view()

        ax[1,2].relim()
        ax[1,2].autoscale_view()

        ax[2,0].relim()
        ax[2,0].autoscale_view()

        ax[2,1].relim()
        ax[2,1].autoscale_view()

        ax[0,0].legend()
        plt.draw()
        plt.pause(0.1)
    old_network = copy.deepcopy(policy_network)
    loss_iter = PPO(data, policy_network, eps)

    line10.set_xdata(np.arange(len(loss_iter)))
    line10.set_ydata(loss_iter)
    
    ax[2,2].relim()
    ax[2,2].autoscale_view()

    plt.draw()
    plt.pause(0.1)    
    train_value(value_network, data, 5)
    print(f"it's the {epoch}")
env.close()

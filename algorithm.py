import torch
import torch.optim as optim
import torch.utils.tensorboard
from value_function import esperance
from torch.utils.tensorboard import SummaryWriter
import copy

count = 0
def PPO(data, policy_network, eps):
    global count
    loss_iter = []
    
    old_network = copy.deepcopy(policy_network)
    optimizer = optim.Adam(policy_network.parameters(), lr=0.01)
    for _  in range(5):
        optimizer.zero_grad()
        loss = esperance(data, policy_network, eps, old_network)
        loss.backward()  
        optimizer.step()

        loss_iter.append(loss.item())
        print(loss.item())
    return loss_iter



import torch

def to_action(state, network):
    weights = network(state)

    action = [0,1]
    dist = torch.distributions.Categorical(weights)
    index = dist.sample().item()

    return action[index]

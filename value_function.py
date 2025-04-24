import torch
import torch.nn as nn
import torch.optim as optim

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def entropy(dist):
    return -torch.mul(dist, torch.log2(dist)).sum().detach().cpu().numpy()

def esperance(data, network,eps, old_network):
    T = len(data)
    E = torch.tensor([0.0]).to(device)
    for state, action, advantage, _, _,_ in data:
        policy_new = network(state)[action]
        policy_old = old_network(state)[action].item()
        ratio = policy_new/policy_old
        E = E + torch.min(ratio * advantage, torch.clamp(ratio, 1-eps, 1+eps)*advantage)
    E = (1/T)*E
    loss = -E
    return loss

def train_value(network, data, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.01)
    network.train()
    for _ in range(epochs):
        for state, _, advantage, reward, new_state, done in data:
            with torch.no_grad():
                if done:
                    y_true = torch.tensor([-10], dtype=torch.float).to(device)
                else:
                    y_true = reward + 0.9*network(new_state)

            optimizer.zero_grad()
            y_pred = network(state)
            loss = criterion(y_pred,y_true)

            loss.backward()
            optimizer.step()

def GAE(trajectory, network):
    theta = 0.99
    lamda = 0.3

    data = []

    network.eval()
    with torch.no_grad():
        for i, (state, action, reward, next_state, done) in enumerate(trajectory):
            
            advantage = 0
            for l, (state_l, _, reward_l, next_state_l, done_l) in enumerate(trajectory[i:]):
                TD = reward_l + theta * network(next_state_l).item() - network(state_l).item()
                advantage += (lamda)**l*TD
            

            data.append((state, action, advantage, reward, next_state, done))

    return data
        

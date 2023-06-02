import torch as T
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

# Hyper Parameters

class Actor(nn.Module):
    def __init__(self, space_states_dim, space_actions_dim, device, LAYER1_SIZE= 300, LAYER2_SIZE= 200):
        super(Actor, self).__init__()
        self.device = device
        self.fc1= nn.Linear(space_states_dim, LAYER1_SIZE).to(self.device)

        self.fc2= nn.Linear(LAYER1_SIZE, LAYER2_SIZE).to(self.device)

        self.fc3= nn.Linear(LAYER2_SIZE, space_actions_dim).to(self.device)


    def forward(self, state):

        l1 = F.relu(self.fc1(state)).to(self.device)
        l2 = F.relu(self.fc2(l1)).to(self.device)
        actions = T.tanh(self.fc3(l2))

        return actions

class Critic(nn.Module):
    def __init__(self, space_states_dim, space_actions_dim, device, LAYER1_SIZE= 300, LAYER2_SIZE= 200):
        super(Critic, self).__init__()
        self.device = device
        self.fc1= nn.Linear(space_states_dim+space_actions_dim, LAYER1_SIZE).to(self.device)
        self.fc2= nn.Linear(LAYER1_SIZE, LAYER2_SIZE).to(self.device)
        self.fc3= nn.Linear(LAYER2_SIZE, 1).to(self.device)


    def forward(self, state, action):
        x =T.cat([state, action],1).to(self.device) # ----- 4 -----
        l1 = F.relu(self.fc1(x)).to(self.device)
        l2 = F.relu(self.fc2(l1)).to(self.device)
        Q = self.fc3(l2)

        return Q

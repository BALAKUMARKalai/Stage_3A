import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic (nn.Module): 
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,1)
        
    def forward(self, state, action):
        input = torch.cat([state,action], dim = 1)
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        
        
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, Pmax):
        super(Actor,self).__init__() 
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc_mu = nn.Linear(64, action_dim)
        self.fc_log_std = nn.Linear(64, action_dim)
        self.Pmax = Pmax
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x  = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clip(log_std, -20, 2)
        std = torch.exp(log_std)
        
        #Reparametrization
        epsilon = torch.randn_like(mu)
        action = torch.tanh(mu + std * epsilon)
        action_scaled = (action +1)/2
        action_scaled[..., 0] = action_scaled[..., 0] * self.Pmax
        return action_scaled,
        
    def sample(self, state):
        
        
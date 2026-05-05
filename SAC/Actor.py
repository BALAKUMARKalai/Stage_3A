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
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        #Reparametrization
        epsilon = torch.randn_like(mu)
        x_t = mu + std * epsilon
        y_t = torch.tanh(x_t)
        
        #rescale les actions
        scale = torch.ones_like(y_t)
        scale[..., 0] = self.Pmax

        #Log prob avec correction tanh uniquement (standard SAC)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        action = (y_t + 1) / 2 * scale 
        
        #Mean Action pour evaluation
        mean_action = (torch.tanh(mu) +1)/2
        mean_action[...,0] = mean_action[...,0] * self.Pmax
        return action, log_prob, mean_action
      
        
    def sample(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, mean_action = self.forward(state)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy(), mean_action.cpu().numpy().flatten()
        
        
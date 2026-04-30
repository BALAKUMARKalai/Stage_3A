import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic,self).__init__()
        self.fc11 = nn.Linear(state_dim + action_dim, 128)
        self.fc21 = nn.Linear(128,64)
        self.fc31 = nn.Linear(64,1)
        self.fc12 = nn.Linear(state_dim + action_dim, 128)
        self.fc22 = nn.Linear(128,64)
        self.fc32 = nn.Linear(64,1)
    
    def forward(self,state,action):
        x = torch.cat([state,action], dim = 1)
        Q1 = F.relu(self.fc11(x))
        Q1 = F.relu(self.fc21(Q1))
        Q1 = self.fc31(Q1)
        Q2 = F.relu(self.fc12(x))
        Q2 = F.relu(self.fc22(Q2))
        Q2 = self.fc32(Q2)
        return Q1,Q2
    
    def Q1_only(self, state,action):
        Q1,_ = self.forward(state,action)
        return Q1
        
        
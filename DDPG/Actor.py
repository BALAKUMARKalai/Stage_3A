import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, Pmax):
        super(Actor,self).__init__() 
        #architecture standard du papier originel. Les dimensions des couches peuvent être tunés
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64, action_dim)
        self.Pmax = Pmax
            
            
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        #On rescale Ps puisque la sortie d'une sigmoide est dans [0,1] et non [0, Pmax]
        scale = torch.ones_like(x)
        scale[..., 0] = self.Pmax
        x = x * scale
        return x
        
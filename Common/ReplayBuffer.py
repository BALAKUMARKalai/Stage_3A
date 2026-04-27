import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        #On initialise les arrays.
        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dones = np.zeros((max_size, 1))
        self.max_size = max_size
        self.ptr, self.size = 0, 0
    
    def add(self, state, action, reward, next_state, done):
        #On remplit les arrays à partir des séquences
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.next_state[self.ptr] = next_state
        self.ptr = (self.ptr + 1)% self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    
    def sample(self, batch_size):
        index =np.random.randint(0, self.size, batch_size)
        return (
        torch.FloatTensor(self.states[index]),
        torch.FloatTensor(self.actions[index]),
        torch.FloatTensor(self.rewards[index]),
        torch.FloatTensor(self.next_state[index]),
        torch.FloatTensor(self.dones[index]))
        
        
        
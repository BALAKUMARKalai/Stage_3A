import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Actor import Actor
from Critic import Critic
from Common.ReplayBuffer import ReplayBuffer


class DDPG:
    def __init__(self, state_dim, action_dim, Pmax, K, tau = 0.001, gamma = 0.95,
                lr_actor = 5e-5, lr_critic = 3e-4, buffer_size = 100000, batch_size = 64):
        self.N_EPISODES = 2000
        self.T = 500 #500
        self.N = 1000 #1000 #Averaging
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ReplayBuffer(max_size = buffer_size , state_dim = state_dim, action_dim = action_dim)
        self.actor = Actor(state_dim = state_dim , action_dim = action_dim , Pmax = Pmax).to(self.device)
        self.critic = Critic(state_dim = state_dim, action_dim = action_dim).to(self.device)
        self.actor_target = Actor(state_dim=state_dim, action_dim=action_dim, Pmax=Pmax).to(self.device)
        self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = lr_critic)
        

        
        
    def select_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().detach().numpy().flatten()
            
    
    def train(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = [t.to(self.device) for t in replay_buffer.sample(batch_size)]
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            Q_target = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * (1-dones) * Q_target #Bellman Equation
            
        #Update Critic
        Q_current = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(Q_current, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) #catastroph forgetting
        self.critic_optimizer.step()

        #Update Actor
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
            
        #Soft Update Target Networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
    
    
    def soft_update(self, network, target_network):
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data = self.tau * param.data + (1- self.tau)* target_param.data
        
    
        
        
    
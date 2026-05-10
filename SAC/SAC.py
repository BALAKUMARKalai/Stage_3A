import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Actor import Actor
from Critic import Critic
from Common.ReplayBuffer import ReplayBuffer

class SAC:
    def __init__(self, state_dim, action_dim, Pmax,K, tau = 0.001, gamma = 0.95,
                lr_actor = 3e-4, lr_critic = 3e-4, lr_alpha = 3e-4, buffer_size = 100000, batch_size = 64):
        self.N_EPISODES = 2000
        self.T = 500 #500
        self.N = 1000 #1000 #Averaging
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ReplayBuffer(max_size=buffer_size, state_dim=state_dim, action_dim=action_dim)
        self.actor = Actor(state_dim = state_dim, action_dim = action_dim, Pmax = Pmax).to(self.device)
        self.critic = Critic(state_dim = state_dim, action_dim = action_dim).to(self.device)
        self.critic_target = Critic(state_dim = state_dim, action_dim = action_dim).to(self.device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = lr_critic)
        self.log_alpha = torch.zeros(1, requires_grad=True, device = self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = torch.tensor(-float(action_dim), dtype=torch.float32, device=self.device)
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action,_,_ = self.actor(state)
        return action.cpu().detach().numpy().flatten()
        
    def train(self, replay_buffer, batch_size):
        states, actions, rewards,next_states, dones = [t.to(self.device) for t in replay_buffer.sample(batch_size)]
        #Calcul de y
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor(next_states)
            Q1_target, Q2_target = self.critic_target(next_states, next_actions)
            Q_min = torch.min(Q1_target, Q2_target)
            y= rewards + self.gamma * (1-dones) * (Q_min - self.log_alpha.exp().detach() * next_log_prob)
        #Update Critic
        Q1_current, Q2_current = self.critic(states, actions)
        critic_loss = torch.nn.functional.mse_loss(Q1_current, y) + torch.nn.functional.mse_loss(Q2_current,y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        #Update Actor
        actions_pred, log_prob, _ = self.actor(states)
        Q1, Q2 = self.critic(states, actions_pred)
        actor_loss = (self.log_alpha.exp().detach() * log_prob - torch.min(Q1, Q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        #Update alpha
        alpha_loss = - (self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        #Soft Update
        
        self.soft_update(self.critic, self.critic_target)
        # Ajoute dans train()
        """
        print(f"alpha: {self.log_alpha.exp().item():.4f}")
        print(f"actor_loss: {actor_loss.item():.4f}")
        print(f"critic_loss: {critic_loss.item():.4f}")
        """
    
    
    def soft_update(self, network,target_network):
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data = self.tau * param.data + (1- self.tau)* target_param.data
                
        
            
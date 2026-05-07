import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SAC import SAC
from Common.environment import BackComEnv
import numpy as np
import matplotlib.pyplot as plt


K = 2
state_dim = 3 * K
action_dim = 2 + K
MIN_BUFFER_SIZE = 1000
N_EPISODES = 2000

env = BackComEnv()
agent = SAC(state_dim=state_dim, action_dim=action_dim, Pmax=1.0, K=K)

ee_per_episode = []
reward_per_episode = []

for episode in range(N_EPISODES):
    state = env.reset()
    episode_reward = 0
    
    for t in range(env.T):
        action = agent.select_action(state)
        action[0] = np.clip(action[0], 0, env.Pmax)
        action[1] = np.clip(action[1], 1e-6, 1.0)
        action[2:] = np.clip(action[2:], 0, 1)
        
        next_state, reward, done = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        
        #if agent.replay_buffer.size >= MIN_BUFFER_SIZE and t % 4 == 0: 
        if agent.replay_buffer.size >= MIN_BUFFER_SIZE: 
            agent.train(agent.replay_buffer, agent.batch_size)
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    ee = env.sum_Rsum / env.sum_Etotal
    ee_per_episode.append(ee)
    reward_per_episode.append(episode_reward)
    
    if episode % 10 == 0:
        print(f"Episode {episode} | EE: {ee:.4f} | Reward: {episode_reward:.4f}")
window = 50
ee_smooth = np.convolve(ee_per_episode, np.ones(window)/window, mode='valid')

plt.figure()
plt.plot(ee_per_episode, alpha=0.3, label='EE brut')
plt.plot(range(window - 1, N_EPISODES), ee_smooth, label=f'Moyenne glissante ({window} ep)')
plt.xlabel('Episodes')
plt.ylabel('EE')
plt.legend()
plt.show()
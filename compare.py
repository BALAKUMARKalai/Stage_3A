import sys
import os
import numpy as np
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))

np.random.seed(200)
original_path = list(sys.path)

sys.path.insert(0, os.path.join(base, 'DDPG'))
from DDPG import DDPG
for _m in ['Actor', 'Critic', 'DDPG']:
    sys.modules.pop(_m, None)
sys.path = list(original_path)

sys.path.insert(0, os.path.join(base, 'TD3'))
from TD3 import TD3
for _m in ['Actor', 'Critic', 'TD3']:
    sys.modules.pop(_m, None)
sys.path = list(original_path)

sys.path.insert(0, os.path.join(base, 'SAC'))
from SAC import SAC
for _m in ['Actor', 'Critic', 'SAC']:
    sys.modules.pop(_m, None)
sys.path = list(original_path)

sys.path.insert(0, os.path.join(base, 'greedy_dinkelbach'))
from greedy_dinkelbach import GreedyDinkelbach
sys.path = list(original_path)

sys.path.insert(0, base)
from Common.environment import BackComEnv

K           = 2
state_dim   = 3 * K
action_dim  = 2 + K
MIN_BUFFER  = 1000
N_EPISODES  = 2000
WINDOW      = 50
NOISE_INIT  = 0.3
NOISE_DECAY = 0.995
NOISE_MIN   = 0.01


def run(env, agent, name, external_noise=True):
    ee_history = []
    noise_std = NOISE_INIT

    for ep in range(N_EPISODES):
        state = env.reset()

        for t in range(env.T):
            action = agent.select_action(state)

            if external_noise:
                action = action + np.random.normal(0, noise_std, size=action.shape)

            action[0]  = np.clip(action[0],  0,    env.Pmax)
            action[1]  = np.clip(action[1],  1e-6, 1.0)
            action[2:] = np.clip(action[2:], 0,    1.0)

            next_state, reward, done = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)

            if agent.replay_buffer.size >= MIN_BUFFER and t % 4 == 0:
                agent.train(agent.replay_buffer, agent.batch_size)

            state = next_state
            if done:
                break

        ee = env.sum_Rsum / env.sum_Etotal
        ee_history.append(ee)

        if external_noise:
            noise_std = max(NOISE_MIN, noise_std * NOISE_DECAY)

        if ep % 100 == 0:
            print(f"[{name}] ep {ep:4d} | EE: {ee:.4f}")

    return np.array(ee_history)


def run_baseline(env, agent, name):
    ee_history = []

    for ep in range(N_EPISODES):
        state = env.reset()

        for t in range(env.T):
            action = agent.select_action(state)
            action[0]  = np.clip(action[0],  0,    env.Pmax)
            action[1]  = np.clip(action[1],  1e-6, 1.0)
            action[2:] = np.clip(action[2:], 0,    1.0)
            state, _, done = env.step(action)
            if done:
                break

        ee = env.sum_Rsum / env.sum_Etotal
        ee_history.append(ee)

        if ep % 100 == 0:
            print(f"[{name}] ep {ep:4d} | EE: {ee:.4f}")

    return np.array(ee_history)


env = BackComEnv()

dinkelbach = GreedyDinkelbach(
    K=K, Pmax=env.Pmax, eta=env.eta, eps=env.eps,
    Psc=env.Psc, Prc=env.Prc, Ptc_k=env.Ptc_k, sigma2=env.sigma2,
)

configs = [
    ('DDPG',       DDPG(state_dim, action_dim, Pmax=1.0, K=K), True),
    ('TD3',        TD3 (state_dim, action_dim, Pmax=1.0, K=K), True),
    ('SAC',        SAC (state_dim, action_dim, Pmax=1.0, K=K), False),
]

results = {}
for name, agent, use_noise in configs:
    print(f"\n{'='*40} {name} {'='*40}")
    results[name] = run(env, agent, name, external_noise=use_noise)

print(f"\n{'='*40} Greedy Dinkelbach {'='*40}")
results['Greedy Dinkelbach'] = run_baseline(env, dinkelbach, 'Greedy Dinkelbach')

# ── Plot superposé ────────────────────────────────────────────────────────────
colors = {
    'DDPG':             'steelblue',
    'TD3':              'tomato',
    'SAC':              'seagreen',
    'Greedy Dinkelbach':'darkorange',
}

fig, ax = plt.subplots(figsize=(10, 6))

for name, ee in results.items():
    smooth = np.convolve(ee, np.ones(WINDOW) / WINDOW, mode='valid')
    ax.plot(ee, alpha=0.15, color=colors[name])
    ax.plot(range(WINDOW - 1, N_EPISODES), smooth,
            color=colors[name], linewidth=2, label=name)

ax.set_xlabel('Épisodes')
ax.set_ylabel('EE (bits/Joule)')
ax.set_title('Comparaison DDPG / TD3 / SAC / Greedy Dinkelbach')
ax.legend()
plt.tight_layout()
plt.savefig('resultats4.png', dpi=150)
print("\nFigure sauvegardée : resultats_same_plot.png")

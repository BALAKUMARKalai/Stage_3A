import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Common.environment import BackComEnv

np.random.seed(150)

K          = 2
N_EPISODES = 2000
WINDOW     = 50

env = BackComEnv()
print(f"d_0k={env.d_0k.round(2)}, dk={env.dk.round(2)}")


class GreedyDinkelbach:
    def __init__(self, K, Pmax, eta, eps, Psc, Prc, Ptc_k, sigma2,
                 n_iter=100, tol=1e-6):
        self.K      = K
        self.Pmax   = Pmax
        self.eta    = eta
        self.eps    = eps
        self.Psc    = Psc
        self.Prc    = Prc
        self.Ptc_k  = Ptc_k
        self.sigma2 = sigma2
        self.n_iter = n_iter
        self.tol    = tol

    def select_action(self, state):
        Ek      = state[0          : self.K]
        hk_sq   = state[self.K     : 2*self.K]
        gk_sq   = state[2*self.K   : 3*self.K]
        gamma_k = hk_sq * gk_sq / self.sigma2
        Ps, tau_a, beta = self._dinkelbach_ao(gamma_k, hk_sq, Ek)
        return np.concatenate([[Ps, tau_a], beta])

    def _dinkelbach_ao(self, gamma_k, hk_sq, Ek):
        alpha  = 0.0
        kappa  = 1.0
        Ps_opt = self.Pmax

   
        ps_needed  = self.Ptc_k / (self.eta * hk_sq)
        E_circuit  = self.Ptc_k * 1.0          # énergie circuit pour tau_a=1
        feasible   = (ps_needed <= self.Pmax) | (Ek >= E_circuit)
        #print(f"hk_sq={hk_sq}, Ek={Ek}, ps_needed={ps_needed.round(2)}, E_circuit={E_circuit}, feasible={feasible}")

        if not np.any(feasible):
            return 0.0, 1e-6, np.zeros(self.K)

        hk_sq_f   = hk_sq[feasible]
        gamma_k_f = gamma_k[feasible]

        # Ps_min uniquement sur les BNs dont le canal est insuffisant
        # mais dont l'énergie stockée compense
        canal_ok  = ps_needed[feasible] <= self.Pmax
        if np.any(canal_ok):
            Ps_min = np.max(self.Ptc_k / (self.eta * hk_sq_f[canal_ok]))
        else:
            Ps_min = 0.0   

        gk_sq_over_sigma = gamma_k / hk_sq  # = gk²/sigma²
    
        mu = (1.0 - np.sum(self.Ptc_k * gk_sq_over_sigma / self.eta)
              if np.any(canal_ok) else 1.0)

        for _ in range(self.n_iter):

            # Mode HoT 
            if alpha > 1e-10:
                Ps_0_i = self.eps / (alpha * np.log(2)) - mu / np.sum(gamma_k_f)
            else:
                Ps_0_i = self.Pmax

            Ps_hot = np.clip(Ps_0_i, Ps_min, self.Pmax)
            f_hot  = (np.log2(mu + np.sum(gamma_k_f * Ps_hot))
                      - alpha * (Ps_hot/self.eps + self.Psc + self.Prc))

            #Mode HtT
            Ps_min_ii = max(self.Pmax, Ps_min)
            Ps_max_ii = self.Pmax + (np.min(self.Ptc_k / (self.eta * hk_sq_f))
                                     if np.any(canal_ok) else self.Pmax)

            denom = alpha * np.log(2) * (1.0/self.eps + self.Psc/self.Pmax)
            Ps_0_ii = (1.0/denom - mu/np.sum(gamma_k_f)) if denom > 1e-10 else Ps_min_ii

            Ps_htt    = np.clip(Ps_0_ii, Ps_min_ii, Ps_max_ii)
            kappa_htt = Ps_htt / self.Pmax
            f_htt     = (np.log2(mu + np.sum(gamma_k_f * Ps_htt))
                         - alpha * (Ps_htt/self.eps + kappa_htt*self.Psc + self.Prc))

            #Sélection mode 
            if f_hot >= f_htt:
                Ps_opt = Ps_hot
                kappa  = 1.0
            else:
                Ps_opt = Ps_htt
                kappa  = kappa_htt

            # Mise à jour Dinkelbach 
            num   = np.log2(mu + np.sum(gamma_k_f * Ps_opt))
            denom = Ps_opt/self.eps + kappa*self.Psc + self.Prc

            f_alpha   = num - alpha * denom
            alpha_new = num / denom if denom > 1e-10 else 0.0

            if abs(f_alpha) < self.tol:
                alpha = alpha_new
                break
            alpha = alpha_new

        # Récupérer tau_a et beta
        tau_a = 1.0 / kappa
        Ps    = Ps_opt / kappa

        beta = np.zeros(self.K)
        beta[feasible] = np.clip(
            1.0/tau_a - self.Ptc_k / (self.eta * Ps * hk_sq_f)
            if Ps > 1e-10 else np.zeros(np.sum(feasible)),
            0.0, 1.0
        )

        return Ps, tau_a, beta


#Instanciation 
baseline = GreedyDinkelbach(
    K      = K,
    Pmax   = env.Pmax,
    eta    = env.eta,
    eps    = env.eps,
    Psc    = env.Psc,
    Prc    = env.Prc,
    Ptc_k  = env.Ptc_k,
    sigma2 = env.sigma2,
)

# Boucle d'évaluation 
ee_baseline = []

for episode in range(N_EPISODES):
    state = env.reset()

    for t in range(env.T):
        action                   = baseline.select_action(state)
        next_state, reward, done = env.step(action)
        state                    = next_state
        if done:
            break

    ee = env.sum_Rsum / env.sum_Etotal
    ee_baseline.append(ee)

    if episode % 100 == 0:
        print(f"[Dinkelbach] ep {episode:4d} | EE: {ee:.4f}")

#Plot
ee_arr    = np.array(ee_baseline)
ee_smooth = np.convolve(ee_arr, np.ones(WINDOW)/WINDOW, mode='valid')

plt.figure(figsize=(8, 5))
plt.plot(ee_arr, alpha=0.3, color='gray', label='EE brut')
plt.plot(range(WINDOW-1, N_EPISODES), ee_smooth,
         color='gray', linewidth=2, label=f'Moyenne glissante ({WINDOW} ep)')
plt.axhline(y=np.mean(ee_arr), color='black', linestyle='--',
            label=f'Moyenne globale = {np.mean(ee_arr):.4f}')
plt.xlabel('Épisodes')
plt.ylabel('EE (bits/Joule)')
plt.title('Baseline Greedy Dinkelbach')
plt.legend()
plt.tight_layout()
plt.savefig('dinkelbach.png', dpi=150)
plt.show()

print(f"\nEE moyenne Dinkelbach : {np.mean(ee_arr):.4f} bits/Joule")
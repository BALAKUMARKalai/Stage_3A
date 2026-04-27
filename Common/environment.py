import numpy as np
import math

class BackComEnv :
    def __init__(self) : 
        #On convertit les puissances dBm en Watts = P_watts = 10^(PdBm/10)/1000
        self.sigma2 = 1e-13 #W
        self.eta = 0.6 #Harvesting_efficiency
        self.n = 3 #path loss efficiency
        self.eps = 0.9 #Amplifier Efficiency 
        self.Psc = 0.1 #W
        self.Prc = 0.01 #W
        self.Ptc_k = 1e-3 #W
        self.Pmax = 1.0 #W
        self.Bk = 2e-3
        self.E_k_max = 2e-3 #J
        self.T = 100 #500 #number of time slots
        self.gamma = 0.95 #discount factor
        self.K = 2 # number of BNs
        self.Ek = np.zeros(self.K)
        self.max_distance = 40 #max distance between RF and BR (m)
        
        self.d_0k = np.random.uniform(5,20, self.K) #distance RF source à BN k
        self.dk = np.random.uniform(5,20,self.K) #distance BN k au BR
        self.t = 0
    
    def reset(self):
        #On réinistiallise les énergies de chaque BN, les accumulateurs, le compteur et les canaux
        self.Ek = np.zeros(self.K)
        self.sum_Rsum = 0.0
        self.sum_Etotal = 0.0
        self.t = 0
        self._generate_channels()
        return self._get_state()
            
    
    def _generate_channels(self):
        h_tilde = (np.random.randn(self.K) + 1j * np.random.randn(self.K))/ np.sqrt(2)
        g_tilde = (np.random.randn(self.K) + 1j * np.random.randn(self.K))/ np.sqrt(2)
        self.hk = h_tilde * self.d_0k **(-self.n)
        self.gk = g_tilde * self.dk ** (-self.n)
        return self.hk, self.gk
    
    def _compute_rewards(self,Ps, tau_a, beta):
        gamma_k = (np.abs(self.hk)**2 * np.abs(self.gk)**2) /self.sigma2
        s = np.sum(beta * gamma_k)
              
        R_sum = tau_a * np.log2(1+ Ps * s)
        E_total = (1-tau_a) * (Ps/self.eps+ self.Psc) + tau_a*(Ps/self.eps + self.Psc + self.Prc)
        self.sum_Rsum += R_sum
        self.sum_Etotal += E_total
        rt = R_sum/E_total
        return rt
        
    def _update_energy(self,Ps,tau_a, beta):
        tau_s = 1-tau_a
        Ek_sleep = self.eta * Ps * np.abs(self.hk)**2 * tau_s
        Ek_active = self.eta * (1- beta) * Ps * np.abs(self.hk)**2 * tau_a
        Ec_k = self.Ptc_k*tau_a
        self.Ek = self.Ek + Ek_sleep + Ek_active - Ec_k
        self.Ek = np.clip(self.Ek, 0, self.Bk)
    
    def _get_state(self):
        return np.concatenate([self.Ek, np.abs(self.hk)**2, np.abs(self.gk)**2])
    
    def _check_constraints(self,action):
        Ps, tau_a = action[0], action[1]
        beta = action [2:]
        Ps = np.clip(Ps,0, self.Pmax)
        tau_a = np.clip(tau_a, 1e-6, 1.0) #tau_a > 0 strictement
        beta = np.clip(beta,0,1)
        
        return np.concatenate([[Ps,tau_a],beta]) #on concatene les résultats sur un slot (Ps, tau et les coeff de refexions pour chaque BN)
    
    #optimisation dynamique
    def step(self,action):
        action = self._check_constraints(action)
        Ps = action[0]
        tau_a = action[1]
        beta = action[2:]
        
        rt = self._compute_rewards(Ps,tau_a, beta)
        self._update_energy(Ps,tau_a,beta)
        self.t += 1
        done = self.t >= self.T
        self._generate_channels()
        next_state = self._get_state()
        return next_state,rt, done
        

    
    
    
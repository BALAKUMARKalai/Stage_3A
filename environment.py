import numpy as np

class BackComEnv :
    def __init(self) : 
        self.sigma2 = -100 #dBm
        self.eta = 0.6 #Harvesting_efficiency
        self.n = 3 #path loss efficiency
        self.eps = 0.9 #Amplifier Efficiency
        self.Psc = 20 #dBm
        self.Prc = 10 #dBm
        self.Ptc_k = 0 #dBm
        self.E_k_max = 2 #mJ
        self.T = 500 #number of time slots
        self.gamma = 0.95 #discount factor
        self.K = 2 # number of BNs
        self.max_distance = 40 #max distance between RF and BR (m)
        
    def reset(self):
        self.hk = 0
        self.gk = 0
        
        return
            
        
    
    def _generate_channels(self):
        self.hk = np.random.exponential(scale = 1)
        self.gk = np.random.exponential(scale = 1) * np.random(self.max_distance)**self.n
        
    
    
    def _compute_rewards(Ps, tau_a, beta):
        R_sum = 1
        E_total = 1 
        rt = R_sum/ E_total
        
    def _update_energy(Ps,tau_a, beta):
        return
        
        

    
    
    
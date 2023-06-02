from cgitb import reset
import numpy as np

class OUNoise(object):
    def __init__(self,  space_action_dim, mu=0.0, theta=0.15, max_sigma=0.99, min_sigma=0.2, decay_period=8000000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   =  space_action_dim
        self.reset()
        
    def reset(self):
        self.action = np.ones(self.action_dim) * self.mu
        
    def evolve_action(self):
        x  = self.action
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.action = x + dx
        return self.action
    
    def get_noise(self, t=0): 
        ou_action = self.evolve_action()
        decaying = float(float(t)/ self.decay_period)
        self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
        return ou_action

    
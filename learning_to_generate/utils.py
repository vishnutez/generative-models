import torch
import numpy as np


def compute_exp(x):        
    if isinstance(x, torch.Tensor):  # Check if the input is a PyTorch tensor
        return torch.exp(x)
    else:
        return np.exp(x)
    

def compute_sqrt(x):        
    if isinstance(x, torch.Tensor):  # Check if the input is a PyTorch tensor
        return torch.sqrt(x)
    else:
        return np.sqrt(x) 
    

class VPNoiseSchedule:
    def __init__(self, 
                 beta_min=0.1, 
                 beta_max=20
        ):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t
        
    def alpha(self, t):
        return compute_exp(-(self.beta_min * t + (self.beta_max-self.beta_min) * t**2 / 2))
    
    def f(self, x, t):
        return - x * self.beta(t) / 2  # Forward drift term
    
    def g(self, t):
        return compute_sqrt(self.beta(t))  # Forward diffusion term
    

# A Mixture of Gaussians sampler
class MOGSampler:
    def __init__(self, means, covs, weights):
        self.means = means
        self.covs = covs
        self.weights = weights
        self.n_components = len(means)
        self.d = len(means[0])

    def sample(self, n_samples):
        components = np.random.choice(self.n_components, size=n_samples, p=self.weights)
        samples = np.zeros((n_samples, self.d))
        for i, c in enumerate(components):
            if self.d == 1:
                samples[i] = np.random.normal(self.means[c], np.sqrt(self.covs[c]))
            else:
                samples[i] = np.random.multivariate_normal(self.means[c], self.covs[c])
        return samples
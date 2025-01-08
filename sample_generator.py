import numpy as np
import torch

from models import DiffusionNet

# Running the reverse SDE with score approximated by a DiffusionNet incorporating VP-SDE with beta_min, beta_max affine NoiseSchedule

def sample_reverse(d_net: DiffusionNet,
                   n_steps=0.001, 
                   n_samples=100, 
                   eps=1e-5):
    
    dt = 1 / n_steps
    t = np.linspace(0, 1, n_steps)
    xT = np.random.randn(n_samples, d_net.dim)
    x = np.zeros((n_steps, n_samples, d_net.dim))
    x[n_steps-1] = xT

    for i in range(n_steps-1, -1, -1):
        score_t = d_net.score(torch.full((n_samples, 1), t[i]), torch.tensor(x[i]).float()).detach().numpy()
        x[i - 1] = x[i] - (d_net.f(x[i], t[i]) - d_net.g(t[i])**2 * score_t) * dt + d_net.g(t[i]) * np.sqrt(dt) * np.random.randn(*xT.shape)
    return x
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


def conditional_correction(y: torch.Tensor,
                        d_net: DiffusionNet,
                        x_t: torch.Tensor,
                        A: torch.Tensor,
                        t: torch.Tensor,
                        sigma_y=1e-3, 
                        eps=1e-5):
    
    x_t = x_t.detach().requires_grad_(True)
    # log_p_y_0 = -1 / (2 * sigma_y**2) * torch.linalg.norm(y - d_net.pred_x_0(t, x_t) @ A)**2  # Approximate likelihood score
    log_p_y_0 = - 1 / (2 * sigma_y**2) * torch.linalg.norm(y - d_net.pred_x_0(t, x_t) @ A)  # Approximate likelihood score
    grad_x_t = torch.autograd.grad(log_p_y_0, x_t, grad_outputs=torch.ones_like(log_p_y_0))[0]  # Gradient wrt to the likelihood score
    x_t = x_t.detach().requires_grad_(False)

    grad_x_t = grad_x_t.detach().numpy()

    # print('In conditional_correction')

    return grad_x_t


def sample_reverse_conditional_guidance(d_net: DiffusionNet,
                                        y: np.ndarray,
                                        A: np.ndarray,
                                        sigma_y=1e-3,
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
        score_y_t = conditional_correction(torch.tensor(y).float(), d_net, torch.tensor(x[i]).float(), torch.tensor(A).float(), torch.full((n_samples, 1), t[i]).float(), sigma_y)
        score_t_y = score_t + score_y_t
        x[i - 1] = x[i] - (d_net.f(x[i], t[i]) - d_net.g(t[i])**2 * score_t_y) * dt + d_net.g(t[i]) * np.sqrt(dt) * np.random.randn(*xT.shape)
    return x
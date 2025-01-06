import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor

from tqdm import tqdm

from math import sqrt


# Define the network
class ScoreNet(nn.Module):
    def __init__(self, dim: int = 2, h: int = 128, n_layers: int = 4):
        super().__init__()
        self.dim = dim
        self._in = nn.Linear(dim+1, h)
        self._block = nn.Sequential(nn.Linear(h, h), nn.ELU())
        self._out = nn.Linear(h, dim)
        self._backbone = nn.Sequential(*[self._block for _ in range(n_layers)])
        self.net = nn.Sequential(self._in, self._backbone, self._out)
    
    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))


def beta(t, beta_min=0.1, beta_max=20):
    return beta_min + (beta_max - beta_min) * t
    
def alpha(t, beta_min=0.1, beta_max=20):
    return torch.exp(-(beta_min * t + (beta_max-beta_min) * t**2 / 2))


# train the network
def train(net, sampler, dt = 0.001, epochs=1000, lr=1e-3, batch_size=128):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    n_steps = int(1/dt)
    alphas = alpha(torch.linspace(0, 1, n_steps))
    t_steps = torch.linspace(0, 1, n_steps)

    for epoch in tqdm(range(epochs)):
        t = torch.randint(n_steps, size=(batch_size, 1))  # time indices
        x = torch.tensor(sampler.sample(batch_size), dtype=torch.float32)  # data
        a_t = alphas[t]
        # print('a_t:', a_t[0])
        z_t = torch.randn(batch_size, sampler.d)  # noise
        x_t = torch.sqrt(a_t) * x + torch.sqrt(1-a_t) * z_t

        # # Predict score
        # denoising_score = -z_t / (torch.sqrt(1-a_t)+1e-3) # score = -(x_t - sqrt(a_t) * x) / (1-a_t) = -z_t / sqrt(1-a_t) <-> -pred_noise / sqrt(1-a_t)
        # pred_score = net(t_steps[t], x_t)

        # Predict noise

        pred_noise = net(t_steps[t], x_t)

        optimizer.zero_grad()

        # loss = criterion(pred_score, denoising_score) # Predict score
        
        loss = criterion(pred_noise, z_t) # Predict noise
        
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, loss {loss.item()}')


# Running the reverse SDE
def sample_reverse(score_net, f, g, dt=0.001, n_samples=100):
    n = int(1 / dt)
    t = np.linspace(0, 1, n)
    alphas = alpha(torch.from_numpy(t)).numpy()
    xT = np.random.randn(n_samples, score_net.dim)
    x = np.zeros((n, n_samples, score_net.dim))
    x[n-1] = xT
    for i in range(n - 1, -1, -1):
        z_t = score_net(torch.full((n_samples, 1), t[i]), torch.tensor(x[i]).float()).detach().numpy()
        score_i = -z_t / np.sqrt((1-alphas[i]) + 1e-3)
        x[i - 1] = x[i] - (f(x[i], t[i]) - g(t[i])**2 * score_i) * dt + g(t[i]) * np.sqrt(dt) * np.random.randn(*xT.shape)
    return x


# Runing DDPM sampler
def sample_reverse_DDPM(score_net, dt=0.001, n_samples=100):
    n = int(1 / dt)
    t = np.linspace(0, 1, n)
    alphas = alpha(t)
    xT = np.random.randn(n_samples, score_net.dim)
    x = np.zeros((n, n_samples, score_net.dim))
    x[n-1] = xT
    for i in range(n - 1, -1, -1):
        denoiser_i = score_net(torch.full((n_samples, 1), t[i]), torch.tensor(x[i]).float()).detach().numpy()
        x0 = (x[i] + denoiser_i) / sqrt(alphas[i]) # denoised input
        x[i - 1] = alphas[i-1] * x0 + np.sqrt(1-alphas[i-1]) * np.random.randn(*xT.shape)
    return x


# A Mixture of Gaussians sampler
class MOGSampler:
    def __init__(self, means, covs, weights):
        self.means = means
        self.covs = covs
        self.weights = weights
        self.n_components = len(means)
        self.d = len(means[0])

    def sample(self, n):
        components = np.random.choice(self.n_components, size=n, p=self.weights)
        samples = np.zeros((n, self.d))
        for i, c in enumerate(components):
            if self.d == 1:
                samples[i] = np.random.normal(self.means[c], np.sqrt(self.covs[c]))
            else:
                samples[i] = np.random.multivariate_normal(self.means[c], self.covs[c])
        return samples
    



# Generate some data
np.random.seed(0)

dt = 0.001

means = [np.array([-1, -1]), np.array([1, 1])]
covs = [np.array([[0.1, -0.05], [-0.05, 0.1]]), np.array([[0.2, 0.1], [0.1, 0.2]])]
weights = [0.7, 0.3]



# means = [np.zeros(2)]
# covs = [np.array([[0.1, -0.05], [-0.05, 0.1]])]
# weights = [1]



sampler = MOGSampler(means, covs, weights)  # Data sampler

n_samples = 1000

x0_true = sampler.sample(n_samples)

noise_pred_net = ScoreNet()

# train(noise_pred_net, sampler, dt=dt)

# torch.save(noise_pred_net.state_dict(), 'noise_pred_net.pt')

noise_pred_net.load_state_dict(torch.load('noise_pred_net.pt'))

x = sample_reverse(noise_pred_net, f=lambda x, t: -x * beta(t)/2, g=lambda t: np.sqrt(beta(t)), dt=dt, n_samples=n_samples)

x0_fake = x[0]

print("x0_reconstructed:", x0_fake)

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
ax[0].scatter(x0_true[:, 0], x0_true[:, 1], color='k', s=5, marker='o', label='True data')
ax[1].scatter(x0_fake[:, 0], x0_fake[:, 1], color='lightgrey', s=5, marker='s', label='Reconstructed data')
ax[0].set_xlim(-3, 3)
ax[0].set_ylim(-3, 3)

plt.show()
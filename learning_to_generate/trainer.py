from models import DiffusionNet
from utils import MOGSampler
import torch

from tqdm import tqdm


# Train a diffusion network
def train(d_net: DiffusionNet, 
          data_sampler: MOGSampler, 
          n_steps = 0.001, 
          epochs = 3000, 
          lr = 1e-3, 
          batch_size = 64, eps=1e-5):
    
    # Initialize network parameters
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(d_net.parameters(), lr=lr)

    t_steps = torch.linspace(0, 1, n_steps)

    # Get noise schedules from net
    alphas = d_net.alpha(torch.linspace(0, 1, n_steps))
    
    for epoch in tqdm(range(epochs)):

        t = torch.randint(n_steps, size=(batch_size, 1))  # sampler t uniform from [0, T]
        x = torch.tensor(data_sampler.sample(batch_size), dtype=torch.float32)  # sample data and convert to tensor

        alpha_t = alphas[t]
        z_t = torch.randn(batch_size, data_sampler.d)  # sample noise
        x_t = torch.sqrt(alpha_t) * x + torch.sqrt(1-alpha_t) * z_t

        pred_t = d_net(t_steps[t], x_t)

        optimizer.zero_grad()
        scale_t = 1  # Default no scaling

        loss = criterion(scale_t * pred_t, scale_t * z_t) # Predict noise
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, loss {loss.item()}')
import numpy as np
import matplotlib.pyplot as plt
import torch

from models import DiffusionNet
from utils import MOGSampler
from trainer import train
from sample_generator import sample_reverse, sample_reverse_conditional_guidance

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("A diffusion model to learn mixture of Gaussian distributions")

    parser.add_argument("--no_train", action="store_false", help="Train the model")
    parser.add_argument("--load_path", type=str, help="Load path for the saved models", default=None)
    parser.add_argument("--save_plot", action="store_true", help="Save the plot")

    args = parser.parse_args()

    
    # # Generate some data
    np.random.seed(0)

    means = [np.array([-1, -1]), np.array([1, 1])]
    covs = [np.array([[0.1, -0.09], [-0.09, 0.1]]), np.array([[0.2, 0.15], [0.15, 0.2]])]
    weights = [0.7, 0.3]

    data_sampler = MOGSampler(means, covs, weights)  # Data sampler
    n_samples = 1000
    x0_true = data_sampler.sample(n_samples)

    # Noise schedule
    beta_min, beta_max = 0.1, 10
    d_net = DiffusionNet(beta_min=beta_min, beta_max=beta_max)

    n_steps = 1000
    n_epochs = 2000

    if args.no_train:
        train(d_net, data_sampler=data_sampler, n_steps=n_steps, n_epochs=n_epochs)
        torch.save(d_net.state_dict(), f'trained_models/model.pt')
        print("Saved model in trained_models directory")
    elif args.load_path is not None:  
        d_net.load_state_dict(torch.load(args.load_path))
    else:
        print("Either train or choose a path for the load models")
        exit()


    # Generate observations from an observation model


    # Observation matrix

    A = np.array([1.0, 1.0])

    # Slope of the line 


    sigma_y = 1e-2

    # Generate measurements from the x0_true samples

    n_y_samples = 1  # Different y samples
    n_x0_samples_per_y = 100
    x0_y = data_sampler.sample(n_y_samples)
    x0_y_true = np.repeat(x0_y, n_x0_samples_per_y, axis=0)
    n_conditional_samples = n_x0_samples_per_y * n_y_samples

    y = x0_y_true @ A + sigma_y * np.random.randn(n_conditional_samples)
   

    x = sample_reverse(d_net=d_net, n_steps=n_steps, n_samples=n_conditional_samples)  # Unconditional

    x_y = sample_reverse_conditional_guidance(d_net=d_net, y=y, A=A, sigma_y=sigma_y, n_steps=n_steps, n_samples=n_conditional_samples)  # Conditional

    x0_y_fake = x_y[0]
    x0_fake = x[0]

    

    # Set the figure size
    plt.figure(figsize=(10, 10))

    

    plt.scatter(x0_y_fake[:, 0], x0_y_fake[:, 1], color='darkred', s=5, marker='o', label=r'Generated conditional data, $A = $' + f'{A}' + r', $\sigma_y = $' + f'{sigma_y}')
    plt.scatter(x0_fake[:, 0], x0_fake[:, 1], color='steelblue', s=5, marker='o', label='Generated unconditional data', alpha=0.5)


    if A[1] == 0:

        plt.axvline(x = y[0] / A[0], color='lightgrey', label=r'$A x = y$')

    else:
        # Points from lim to lim

        x0_pts = np.linspace(-3, 3, 100)

        # Need A[0] * x0_pts + A[1] * x1_pts = y

        x1_pts = (y - A[0] * x0_pts) / A[1]

        plt.plot(x0_pts, x1_pts, color='lightgrey', label=r'$A x = y$')

    plt.scatter(x0_y_true[:, 0], x0_y_true[:, 1], color='k', s=50, marker='x', label='True data')


    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # Common legend at the bottom
    plt.legend()

    plt.savefig(f"plots/DPS_implementation_true_vs_generated_conditional_A0_{A[0]}_A1_{A[1]}_sigma_y_{sigma_y}.png", bbox_inches="tight", dpi=300)
    
    print("Saved plot in plots directory")

    plt.show()
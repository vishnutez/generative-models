import numpy as np
import matplotlib.pyplot as plt
import torch

from models import DiffusionNet
from utils import MOGSampler
from trainer import train
from sample_generator import sample_reverse

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("A diffusion model to learn mixture of Gaussian distributions")

    parser.add_argument("--no_train", action="store_false", help="Train the model")
    parser.add_argument("--load_path", type=str, help="Load path for the saved models", default=None)
    parser.add_argument("--save_plot", action="store_true", help="Save the plot")

    args = parser.parse_args()

    
    # Generate some data
    np.random.seed(0)


    means = [np.array([-1, -1]), np.array([1, 1])]
    covs = [np.array([[0.1, -0.09], [-0.09, 0.1]]), np.array([[0.2, 0.15], [0.15, 0.2]])]
    weights = [0.7, 0.3]

    data_sampler = MOGSampler(means, covs, weights)  # Data sampler
    n_samples = 1000
    x0_true = data_sampler.sample(n_samples)

    beta_min, beta_max = 0.1, 10
    d_net = DiffusionNet(beta_min=beta_min, beta_max=beta_max)

    n_steps = 1000

    if args.no_train:
        train(d_net, data_sampler=data_sampler, n_steps=n_steps)
        torch.save(d_net.state_dict(), f'trained_models/model.pt')
        print("Saved model in trained_models directory")
    elif args.load_path is not None:  
        d_net.load_state_dict(torch.load(args.load_path))
    else:
        print("Either train or choose a path for the load models")
        exit()

    x = sample_reverse(d_net=d_net, n_steps=n_steps, n_samples=n_samples)

    x0_fake = x[0]


    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].scatter(x0_true[:, 0], x0_true[:, 1], color='k', s=5, marker='o', label='True data')
    ax[1].scatter(x0_fake[:, 0], x0_fake[:, 1], color='lightgrey', s=5, marker='o', label='Generated data')
    ax[0].set_xlim(-3, 3)
    ax[0].set_ylim(-3, 3)

    # Common legend at the bottom
    fig.legend(loc='lower center', ncol=2)

    plt.savefig("plots/true_vs_generated.png", bbox_inches="tight", dpi=300)
    
    print("Saved plot in plots directory")

    plt.show()
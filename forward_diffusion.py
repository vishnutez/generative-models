# Slider plot util using maplotlib

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

from utils import MOGSampler 


# Simulate a stochastic differential equation using the Euler-Maruyama method
def simulate_forward(f, g, x0, n_steps):
    dt = 1 / n_steps
    t = np.linspace(0, 1, n_steps)
    x = np.zeros((n_steps, len(x0), len(x0[0])))
    x[0] = x0
    for i in range(n_steps - 1):
        x[i + 1] = x[i] + f(x[i], t[i]) * dt + g(t[i]) * np.sqrt(dt) * np.random.randn(*x0.shape)
    return x


# Define the drift and diffusion functions
def VP_SDE(beta_min=0.1, beta_max=20):
    def beta(t):  # Choose such that beta(0) = 0 and beta(1) = 1
        return beta_min + (beta_max - beta_min) * t
    def f(x, t):
        return -x * beta(t)/2
    def g(t):
        return np.sqrt(beta(t))
    return f, g


def VE_SDE(sigma_min=0.01, sigma_max=7):
    def sigma(t):
        return sigma_min * (sigma_max / sigma_min) ** t
    def sigma_dot(t):
        return sigma_min * np.log(sigma_max / sigma_min) * (sigma_max / sigma_min) ** t
    def f(x, t):
        return 0
    def g(t):
        return np.sqrt(2*sigma(t)*sigma_dot(t))
    return f, g

# Simulate the SDE  

if __name__ == "__main__":

    # Generate some data
    np.random.seed(0)

    means = [np.array([-1, -1]), np.array([1, 1])]
    covs = [np.array([[0.1, -0.05], [-0.05, 0.1]]), np.array([[0.2, 0.1], [0.1, 0.2]])]
    weights = [0.7, 0.3]

    data_sampler = MOGSampler(means, covs, weights)

    n_samples = 1000

    x0 = data_sampler.sample(n_samples)

    n_steps = 100
    dt = 1 / n_steps

    f_E, g_E = VE_SDE()
    X_VE = simulate_forward(f=f_E, g=g_E, x0=x0, n_steps=n_steps)

    f_P, g_P = VP_SDE(beta_min=0.1, beta_max=10)
    X_VP = simulate_forward(f=f_P, g=g_P, x0=x0, n_steps=n_steps)


    # Plot the data using a slider update for time
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(
        ax=axtime,
        label='time (t)',
        valmin=0,
        valmax=1,
        valinit=0,
        color='grey'
    )

    lim = 10

    # Plot intial data
    ax[0].scatter(X_VE[0, :, 0], X_VE[0, :, 1], s=5, marker='o', c='k')
    ax[1].scatter(X_VP[0, :, 0], X_VP[0, :, 1], s=5, marker='o', c='k')
    ax[0].set_xlim(-lim, lim)
    ax[0].set_ylim(-lim, lim)
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[0].set_title('Variance-exploding SDE')
    ax[1].set_title('Variance-preserving SDE')


    # The function to be called anytime a slider's value changes
    def update(val):
        t_id = max(int(val / dt)-1, 0)
        ax[0].clear()
        ax[1].clear()
        ax[0].scatter(X_VE[t_id, :, 0], X_VE[t_id, :, 1], s=5, marker='o', c='k')
        ax[1].scatter(X_VP[t_id, :, 0], X_VP[t_id, :, 1], s=5, marker='o', c='k')
        ax[0].set_xlim(-lim, lim)
        ax[0].set_ylim(-lim, lim)
        ax[1].set_xlim(-lim, lim)
        ax[1].set_ylim(-lim, lim)
        ax[0].set_title('Variance-exploding SDE')
        ax[1].set_title('Variance-preserving SDE')
        fig.canvas.draw_idle()

    # register the update function with each slider
    time_slider.on_changed(update)

    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        time_slider.reset()
    button.on_clicked(reset)

    plt.show()



        


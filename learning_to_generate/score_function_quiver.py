import numpy as np
import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt

from matplotlib.widgets import Button, Slider
from models import DiffusionNet


lim = 2

# Define the grid of points
x0, x1 = np.meshgrid(np.linspace(-lim, lim, 50), np.linspace(-lim, lim, 50))

x_vecs = torch.from_numpy(np.array([x0, x1]).transpose(1, 2, 0)).float()


beta_min, beta_max = 0.1, 10

d_net = DiffusionNet(beta_min=beta_min, beta_max=beta_max)

d_net.load_state_dict(torch.load(f'../trained_models/model.pt'))


n_steps = 1000
t = torch.linspace(0, 1, n_steps)



# Define the vector field by score function

fig, ax = plt.subplots(figsize=(10, 10))
fig.subplots_adjust(left=0.25, bottom=0.25)


valinit = 0.025

# Make a horizontal slider to control the frequency.
ax_time = fig.add_axes([0.25, 0.1, 0.65, 0.03])
time_slider = Slider(
    ax=ax_time,
    label='time [t]',
    valmin=0.025,
    valmax=1,
    valinit=valinit,
    color='grey'
)

step = -1
t_vecs = torch.full((x_vecs.shape[0], x_vecs.shape[1], 1), t[step])
scores = d_net.score(t_vecs, x_vecs).detach().numpy() # (20, 20, 2)

u = scores[:, :, 0]  # X-component of the vector
v = scores[:, :, 1]  # Y-component of the vector

ax.quiver(x0, x1, u, v, color='k', angles='xy', scale_units='xy', scale=20)
ax.set_title('Score function')
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.grid()
ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)

def update(val):
    step = max(int(time_slider.val * n_steps)-1, 0)
    t_vecs = torch.full((x_vecs.shape[0], x_vecs.shape[1], 1), t[step])
    scores = d_net.score(t_vecs, x_vecs).detach().numpy()  # (20, 20, 2)

    u = scores[:, :, 0]  # X-component of the vector
    v = scores[:, :, 1]  # Y-component of the vector

    ax.clear()
    ax.quiver(x0, x1, u, v, color='k', angles='xy', scale_units='xy', scale=50)
    ax.set_title('Score function')
    ax.set_xlabel('X0')
    ax.set_ylabel('X1')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid()
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.draw()


time_slider.on_changed(update)

resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    time_slider.reset()

button.on_clicked(reset)


plt.show()

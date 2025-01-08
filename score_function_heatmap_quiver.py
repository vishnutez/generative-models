import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.widgets import Button, Slider

from models import DiffusionNet


# class NoiseSchedule:
#     def __init__(self, beta_min=0.1, beta_max=20):
#         self.beta_min = beta_min
#         self.beta_max = beta_max

#     def beta(self, t):
#         return self.beta_min + (self.beta_max - self.beta_min) * t
        
#     def alpha(self, t):
#         return torch.exp(-(self.beta_min * t + (self.beta_max-self.beta_min) * t**2 / 2))


# # def beta(t, beta_min=0.1, beta_max=20):
# #     return beta_min + (beta_max - beta_min) * t
    
# # def alpha(t, beta_min=0.1, beta_max=20):
# #     return torch.exp(-(beta_min * t + (beta_max-beta_min) * t**2 / 2))


# # Define the network
# class ScoreNet(nn.Module):
#     def __init__(self, dim: int = 2, h: int = 128, n_layers: int = 4):
#         super().__init__()
#         self.dim = dim
#         self._in = nn.Linear(dim+1, h)
#         self._block = nn.Sequential(nn.Linear(h, h), nn.ELU())
#         self._out = nn.Linear(h, dim)
#         self._backbone = nn.Sequential(*[self._block for _ in range(n_layers)])
#         self.net = nn.Sequential(self._in, self._backbone, self._out)
    
#     def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
#         return self.net(torch.cat((t, x_t), -1))



lim = 2


# Define the grid of points
x0, x1 = np.meshgrid(np.linspace(-lim, lim, 50), np.linspace(-lim, lim, 50))

x_vecs = torch.from_numpy(np.array([x0, x1]).transpose(1, 2, 0)).float()


beta_min, beta_max = 0.1, 10

d_net = DiffusionNet(beta_min=beta_min, beta_max=beta_max)

d_net.load_state_dict(torch.load(f'trained_models/model.pt'))


n_steps = 1000
t = torch.linspace(0, 1, n_steps)


# Define the vector field by score function

fig, ax = plt.subplots(figsize=(10, 8))
fig.subplots_adjust(left=0.25, bottom=0.25)

valinit = 0.005

# Make a horizontal slider to control the time
ax_time = fig.add_axes([0.25, 0.1, 0.65, 0.03])
time_slider = Slider(
    ax=ax_time,
    label='time [t]',
    valmin=0.005,
    valmax=1,
    valinit=valinit,
    color='grey'
)

step = -1
t_vecs = torch.full((x_vecs.shape[0], x_vecs.shape[1], 1), t[step])
scores = d_net.score(t_vecs, x_vecs).detach().numpy()  # (20, 20, 2)

u = scores[:, :, 0]  # X-component of the vector
v = scores[:, :, 1]  # Y-component of the vector

mag = np.sqrt(u**2 + v**2)

u_unit = u / mag
v_unit = v / mag

scale = 10

cmap = 'Greys'
arrow_color = 'k'

mesh = ax.pcolormesh(x0, x1, mag, shading='auto', cmap=cmap)
ax.quiver(x0, x1, u_unit, v_unit, color=arrow_color, angles='xy', scale_units='xy', scale=scale)
ax.set_title('Score function')
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.grid()
ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)

cbar = plt.colorbar(mesh)


def update(val):
    step = max(int(time_slider.val * n_steps)-1, 0)
    t_vecs = torch.full((x_vecs.shape[0], x_vecs.shape[1], 1), t[step])
    scores = d_net.score(t_vecs, x_vecs) .detach().numpy() # (20, 20, 2)

    u = scores[:, :, 0]  # X-component of the vector
    v = scores[:, :, 1]  # Y-component of the vector

    mag = np.sqrt(u**2 + v**2)

    u_unit = u / mag
    v_unit = v / mag

    ax.clear()
    mesh = ax.pcolormesh(x0, x1, mag, shading='auto', cmap=cmap)

    

    ax.quiver(x0, x1, u_unit, v_unit, color=arrow_color, angles='xy', scale_units='xy', scale=scale)
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

cbar.set_label(r"Magnitude", fontsize=12)

resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    time_slider.reset()

button.on_clicked(reset)



plt.show()

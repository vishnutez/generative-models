## Exploring simple examples to understand diffusion models

This repo is meant to understand SDE processes.

> Run `forward_diffusion.py` to generate a slider plot of two popular forward diffusion processes - (i) Variance preserving (VP) (ii) Variance exploding (VE)

> Run `main.py` to generate samples from a mixture of Gaussians using a diffusion model trained using VP process

> Run `noise_schedule.ipynb` to understand the effect of discretization in SDE and its relation with DDPM $\alpha_t$. Change `n_steps` to see the deviation.

> Run `score_function_heatmap_quiver.py` to plot the slider-based score function learnt by a trained diffusion model


Further explanation:


`models.py` : Code for diffusion model that uses Variance preserving (VP) process

`utils.py`: Code for `MOGSampler` used as the data distribution and some other utils

`sample_generator.py`: Code for running the reverse diffusion that takes a trained `DiffusionNet` as an input




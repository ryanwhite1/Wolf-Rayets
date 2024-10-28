# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:05:12 2024

@author: ryanw
"""
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, grad
import jax.lax as lax
import jax
import jax.scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import jax.scipy.signal as signal
from matplotlib import animation
import time
import emcee
from tqdm import tqdm
import sys

import jaxwt as jwt

import WR_Geom_Model as gm
import WR_binaries as wrb

def wavelet(image):
    face = image.copy().astype(jnp.float32)
    coeffs = jwt.wavedec2(face, "sym2", level=3)
    
    return coeffs

apep = wrb.apep.copy()
# apep['sigma'] = 8

### --- INFERENCE --- ###  
particles, weights = gm.dust_plume(apep)
    
X, Y, H = gm.smooth_histogram2d(particles, weights, apep)
xbins = X[0, :] * 1.3
ybins = Y[:, 0] * 1.3

X, Y, H = gm.smooth_histogram2d_w_bins(particles, weights, apep, xbins, ybins)
H = gm.add_stars(xbins, ybins, H, apep)
# X, Y, H = gm.spiral_grid(particles, weights, wrb.apep)
obs_err = 0.01 * np.max(H)
H += np.random.normal(0, obs_err, H.shape)
gm.plot_spiral(X, Y, H)


# image_wavelet = wavelet(H)

# fig, ax = plt.subplots()
# ax.imshow(image_wavelet[-2][0][0])
# ax.invert_yaxis()


params = {'eccentricity':[0., 0.95], 'inclination':[0, 180], 'open_angle':[0.1, 179]}
params_list = list(params.keys())

i = 0
param = 'eccentricity'
n = 100

# for i, param in enumerate(params):
    
def man_loglike(value):
    starcopy = apep.copy()
    starcopy[param] = value
    samp_particles, samp_weights = gm.dust_plume(starcopy)
    _, _, samp_H = gm.smooth_histogram2d_w_bins(samp_particles, samp_weights, starcopy, X[0, :], Y[:, 0])
    samp_H = gm.add_stars(xbins, ybins, samp_H, starcopy)
    
    # curr_wavelet = wavelet(samp_H)
    
    # horizontal = jnp.sum((curr_wavelet[-2][0][0] - image_wavelet[-2][0][0])**2)
    # vertical = jnp.sum((curr_wavelet[-2][1][0] - image_wavelet[-2][1][0])**2)
    
    # return -0.5 * (horizontal + vertical)
    
    return -0.5 * jnp.sum(((samp_H.flatten() - H.flatten()))**2)
    

like = jit(jax.value_and_grad(man_loglike))
numpyro_logLike = np.zeros(n)
manual_logLike = np.zeros(n)
param_vals = np.linspace(params[param][0], params[param][1], n)
dx = param_vals[1] - param_vals[0]

vals, grads = jnp.zeros(n), jnp.zeros(n)

for j in tqdm(range(n)):
    a, b = like(param_vals[j])
    vals = vals.at[j].set(a)
    grads = grads.at[j].set(b)


fig, axes = plt.subplots(ncols=2)

axes[0].plot(param_vals, vals)
axes[1].plot(param_vals, grads)











# every = 1
# length = 10
# # now calculate some parameters for the animation frames and timing
# # nt = int(stardata['period'])    # roughly one year per frame
# # nt = np.ceil(len(param_vals) / length).astype(int)
# nt = len(param_vals)
# frames = jnp.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
# fps = np.floor(len(frames) / length).astype(int)  # fps for the final animation

# phases = jnp.linspace(0, 1, nt)


# fig, axes = plt.subplots(ncols=3,figsize=(12, 4))
# ax1, ax2, ax3 = axes[0], axes[1], axes[2]

# # ax1.plot(param_vals, vals)
# # ax1.axvline(apep[param])
# ax3.plot(param_vals, grads, label='JAX Grad')
# ax3.plot(param_vals, np.gradient(vals, dx), label='Finite Diff Grad')
# ax3.axvline(apep[param], c='tab:purple', ls='--', label='True Value')
# ax3.axhline(0, c='k')
# param_line = ax3.axvline(param_vals[0], c='tab:red', label='Current Val')
# ax3.legend()
# ax3.set_title('Log Likelihood Gradient')
# ax3.set(xlabel=param)
# ax1.set(title='Pixel Log Likelihood')
# ax2.set(title='Pixel Log Likelihood Gradient')

# plot1 = ax1.pcolormesh(X, Y, iter_arrays[j], vmin=iter_min, vmax=iter_max, cmap='viridis')
# plot2 = ax2.pcolormesh(X, Y, grad_arrays[j], vmin=grad_min, vmax=grad_max, cmap='RdBu')

# print("Starting animation.")

# from matplotlib import animation
# def animate(j):
#     if j%(nt // 10) == 0:
#         print(j/nt * 100, "%", sep='')
#     # ax1.pcolormesh(X, Y, iter_arrays[j], vmin=iter_min, vmax=iter_max, cmap='viridis')
#     # # ax2.pcolormesh(X, Y, grad_arrays[j], vmin=grad_min, vmax=grad_max, cmap='RdBu')
#     # ax2.pcolormesh(X, Y, grad_arrays[j], vmin=grad_min, vmax=grad_max, cmap='RdBu')
#     plot1.set_array(iter_arrays[j])
#     plot2.set_array(grad_arrays[j])
#     fig.suptitle(f'{param}={param_vals[j]:.3f}')
#     param_line.set_xdata(param_vals[j])
#     return fig, 

# ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
# ani.save(f"gradienttest_{param}_test_{nt}_new_TEST.gif", writer='pillow', fps=fps)
    


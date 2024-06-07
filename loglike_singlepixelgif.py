# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:05:12 2024

@author: ryanw
"""
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, grad
import jax.lax as lax
import jax.scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import jax.scipy.signal as signal
from matplotlib import animation
import time
import emcee

import WR_Geom_Model as gm
import WR_binaries as wrb

apep = wrb.apep.copy()
# apep['sigma'] = 3

### --- INFERENCE --- ###  
particles, weights = gm.dust_plume(wrb.apep)
    
X, Y, H = gm.smooth_histogram2d(particles, weights, wrb.apep)
xbins = X[0, :]
ybins = Y[:, 0]
# X, Y, H = gm.spiral_grid(particles, weights, wrb.apep)
obs_err = 0.01 * np.max(H)
H += np.random.normal(0, obs_err, H.shape)
gm.plot_spiral(X, Y, H)
im_size, _ = H.shape


obs = H.flatten()
obs_err = obs_err * jnp.ones(len(obs))

fig, ax = plt.subplots()

ax.plot(jnp.arange(len(obs)), obs, lw=0.5)

import numpyro, chainconsumer, jax

params = {'eccentricity':[0, 0.95], 'inclination':[0, 180], 'open_angle':[0.1, 179]}
params_list = list(params.keys())

i = 0
param = 'inclination'
n = 50

xindices = {'eccentricity':0, 'inclination':int(im_size * 0.6), 'open_angle':int(im_size * 0.6)}
yindices = {'eccentricity':0, 'inclination':int(im_size * 0.29), 'open_angle':int(im_size * 0.75)}

# for i, param in enumerate(params):
    
def man_loglike(value):
    starcopy = apep.copy()
    starcopy[param] = value
    samp_particles, samp_weights = gm.dust_plume(starcopy)
    _, _, samp_H = gm.smooth_histogram2d_w_bins(samp_particles, samp_weights, starcopy, X[0, :], Y[:, 0])
    samp_H = samp_H.flatten()
    
    return -0.5 * jnp.sum(((samp_H - obs) / obs_err)**2)
@jit
def pixel_loglike(value):
    starcopy = apep.copy()
    starcopy[param] = value
    samp_particles, samp_weights = gm.dust_plume(starcopy)
    _, _, samp_H = gm.smooth_histogram2d_w_bins(samp_particles, samp_weights, starcopy, X[0, :], Y[:, 0])
    samp_H = samp_H.flatten()
    
    return -0.5 * ((samp_H - obs) / obs_err)**2
pixel_gradient = jax.jacfwd(pixel_loglike)
pixel_gradient = jax.jit(pixel_gradient)

# like = jit(vmap(jax.value_and_grad(man_loglike)))
 
# numpyro_logLike = np.zeros(n)
# manual_logLike = np.zeros(n)
# param_vals = np.linspace(params[param][0], params[param][1], n)
# dx = param_vals[1] - param_vals[0]

# vals, grads = like(param_vals)

like = jit(jax.value_and_grad(man_loglike))
numpyro_logLike = np.zeros(n)
manual_logLike = np.zeros(n)
param_vals = np.linspace(params[param][0], params[param][1], n)
dx = param_vals[1] - param_vals[0]


every = 1
length = 10
# now calculate some parameters for the animation frames and timing
# nt = int(stardata['period'])    # roughly one year per frame
# nt = np.ceil(len(param_vals) / length).astype(int)
nt = len(param_vals)
frames = jnp.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
fps = np.floor(len(frames) / length).astype(int)  # fps for the final animation

phases = jnp.linspace(0, 1, nt)

iter_arrays = []
grad_arrays = []

for j in range(n):
    iter_vals = pixel_loglike(param_vals[j])
    iter_vals = jnp.reshape(iter_vals, H.shape)
    grad_vals = pixel_gradient(param_vals[j])
    grad_vals = jnp.reshape(grad_vals, H.shape)
    
    iter_arrays.append(iter_vals)
    grad_arrays.append(grad_vals)
    # grad_arrays.append(jnp.sign(grad_vals) * jnp.log10(jnp.abs(grad_vals)))
    
xindex = xindices[param]
yindex = yindices[param]

xvalue = xbins[xindex]
yvalue = ybins[yindex]

vals = np.zeros(n)
grads = np.zeros(n)

for i in range(n):
    vals[i] = iter_arrays[i][xindex, yindex]
    grads[i] = grad_arrays[i][xindex, yindex]




iter_min, iter_max = jnp.min(jnp.array(iter_arrays)), jnp.max(jnp.array(iter_arrays))
grad_min, grad_max = jnp.min(jnp.array(grad_arrays)), jnp.max(jnp.array(grad_arrays))

fig, axes = plt.subplots(ncols=3,figsize=(12, 4))
ax1, ax2, ax3 = axes[0], axes[1], axes[2]

# ax1.plot(param_vals, vals)
# ax1.axvline(apep[param])
ax3.plot(param_vals, grads, label='JAX Grad')
ax3.plot(param_vals, np.gradient(vals, dx), label='Finite Diff Grad')
ax3.axvline(apep[param], c='tab:purple', ls='--', label='True Value')
ax3.axhline(0, c='k')
param_line = ax3.axvline(param_vals[0], c='tab:red', label='Current Val')
ax3.legend()
ax3.set_title('Log Likelihood Gradient')
ax3.set(xlabel=param)
ax1.set(title='Pixel Log Likelihood')
ax2.set(title='Pixel Log Likelihood Gradient')

from matplotlib import animation
def animate(j):
    ax1.pcolormesh(X, Y, iter_arrays[j], vmin=iter_min, vmax=iter_max, cmap='viridis')
    # ax2.pcolormesh(X, Y, grad_arrays[j], vmin=grad_min, vmax=grad_max, cmap='RdBu')
    ax2.pcolormesh(X, Y, grad_arrays[j], vmin=grad_min, vmax=grad_max, cmap='RdBu')
    for ax in [ax1, ax2]:
        ax.scatter(xvalue, yvalue, c='r', s=5)
    fig.suptitle(f'{param}={param_vals[j]:.3f}')
    param_line.set_xdata(param_vals[j])
    return fig, 

ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
ani.save(f"gradienttest_{param}_singlepixel.gif", writer='pillow', fps=fps)
    


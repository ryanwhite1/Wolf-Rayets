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
# apep['sigma'] = 0.01

### --- INFERENCE --- ###  
particles, weights = gm.dust_plume(wrb.apep)
    
X, Y, H = gm.smooth_histogram2d(particles, weights, wrb.apep)
xbins = X[0, :]
ybins = Y[:, 0]
# X, Y, H = gm.spiral_grid(particles, weights, wrb.apep)
obs_err = 0.01 * np.max(H)
H += np.random.normal(0, obs_err, H.shape)
gm.plot_spiral(X, Y, H)



obs = H.flatten()
obs_err = obs_err * jnp.ones(len(obs))

fig, ax = plt.subplots()

ax.plot(jnp.arange(len(obs)), obs, lw=0.5)

import numpyro, chainconsumer, jax

params = {'eccentricity':[0, 0.95], 'inclination':[0, 180], 'open_angle':[0.1, 179]}
params_list = list(params.keys())


n = 50
fig, axes = plt.subplots(ncols=2, nrows=len(params), figsize=(12, 4*len(params_list)))
for i, param in enumerate(params):
    
    def man_loglike(value):
        starcopy = apep.copy()
        starcopy[param] = value
        samp_particles, samp_weights = gm.dust_plume(starcopy)
        _, _, samp_H = gm.smooth_histogram2d_w_bins(samp_particles, samp_weights, starcopy, X[0, :], Y[:, 0])
        samp_H = samp_H.flatten()
        
        return -0.5 * jnp.sum(((samp_H - obs) / obs_err)**2)

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

    vals, grads = jnp.zeros(n), jnp.zeros(n)
    
    for j in range(n):
        a, b = like(param_vals[j])
        vals = vals.at[j].set(a)
        grads = grads.at[j].set(b)
    
    
    
    ax1, ax2 = axes[i, :]
    
    ax1.plot(param_vals, vals)
    ax1.axvline(apep[param])
    ax2.plot(param_vals, grads, label='JAX Grad')
    ax2.plot(param_vals, np.gradient(vals, dx), label='Finite Diff Grad')
    ax2.axvline(apep[param], c='tab:purple', ls='--', label='True Value')
    ax2.axhline(0, c='k')
    if i == 0:
        ax2.legend()
        ax1.set_title('Log Likelihood')
        ax2.set_title('Log Likelihood Gradient')
    for ax in [ax1, ax2]:
        ax.set(xlabel=param)
    


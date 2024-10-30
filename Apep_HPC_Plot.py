# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:10:05 2024

@author: ryanw
"""

import pickle
import corner
import matplotlib.pyplot as plt
import numpy as np
import WR_binaries as wrb
import WR_Geom_Model as gm 
# import pairplots

# set LaTeX font for our figures
plt.rcParams.update({"text.usetex": True})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

sim_run_2 = 'HPC/sim_run_2/173006276018319360_chains'
sim_run_1 = 'HPC/sim_run_1/173001196050679744'
cavity_sim_run_1 = 'HPC/cavity_sim_run_1/173001335740681824'

use_run = cavity_sim_run_1

with open(use_run, 'rb') as file:
    data = pickle.load(file)
    
param_labels = {"eccentricity":r"$e$",
          "open_angle":r"$\theta_{\rm OA}$",
          "phase":r"$\phi$",
          "arg_peri":r"$\omega$",
          "inclination":r"$i$",
          "asc_node":r"$\Omega$",
          "turn_on":r"$\nu_{\rm t\_on}$",
          "turn_off":r"$\nu_{\rm t\_off}$",
          "sigma":"Blur",
          "comp_az":r"$\alpha_{\rm tert}$",
          "comp_incl":r"$\beta_{\rm tert}$",
          "comp_open":r"$\theta_{\rm OA,tert}$"}

labels = [param_labels[label] for label in data.keys()]

Truth = True
truths = [wrb.apep[label] for label in data.keys()] if Truth else None

smooth = 0.8

corner_fig = corner.corner(data, 
              labels=labels,
              show_titles=True,
              smooth=smooth,
              title_fmt=".4f",
              color='tab:blue',
              use_math_text=True,
              levels=[0.393, 0.864],
              fill_contours=False,
              truths=truths,
              truth_color='tab:purple',
              labelpad=0.15)
              # quantiles=[0.16, 0.5, 0.84])

corner_fig.savefig(f'{use_run}_corner.png', dpi=400, bbox_inches='tight')
corner_fig.savefig(f'{use_run}_corner.pdf', dpi=400, bbox_inches='tight')

ndim = len(data.keys())
params = list(data.keys())

fig, axes = plt.subplots(figsize=(7, 2 * ndim), nrows=ndim, sharex=True, gridspec_kw={'hspace':0})

for i in range(ndim):
    param_vals = data[params[i]]
    if len(param_vals.shape) > 1:
        for j in range(param_vals.shape[0]):
            axes[i].scatter(np.arange(len(param_vals[j, :])), param_vals[j, :], s=1, rasterized=True)
    else:
        axes[i].scatter(np.arange(len(param_vals)), param_vals, s=1, rasterized=True)
    axes[i].set(ylabel=param_labels[params[i]])
axes[-1].set(xlabel='Walker Iteration')
    
fig.savefig(f'{use_run}_chains.png', dpi=400, bbox_inches='tight')
fig.savefig(f'{use_run}_chains.pdf', dpi=400, bbox_inches='tight')






apep = wrb.apep.copy()


np.random.seed(5842)

particles, weights = gm.dust_plume(wrb.apep)
    
X, Y, H_true = gm.smooth_histogram2d(particles, weights, wrb.apep)
xbins = X[0, :]
ybins = Y[:, 0]
# X, Y, H = gm.spiral_grid(particles, weights, wrb.apep)
obs_err = 0.05 * np.max(H_true)
H_true += np.random.normal(0, obs_err, H_true.shape)
fig, axes = plt.subplots(figsize=(8, 4), ncols=2)
for ax in axes:
    ax.set_facecolor('k')
    ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')
axes[0].pcolormesh(X, Y, H_true, cmap='hot', rasterized=True)

system_params = apep.copy()
for param in params:
    system_params[param] = np.mean(data[param])
particles, weights = gm.dust_plume(system_params)
X, Y, H = gm.smooth_histogram2d(particles, weights, system_params)
axes[1].pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    

fig.savefig(f'{use_run}_model.png', dpi=400, bbox_inches='tight')
fig.savefig(f'{use_run}_model.pdf', dpi=400, bbox_inches='tight')



from numpy.random import default_rng


N = 500
rng = default_rng()
chain_numbers = rng.choice(param_vals.shape[0], size=N)
samp_numbers = rng.choice(len(data[params[0]].flatten()), size=N, replace=False)

image_samples = np.zeros((N, H_true.shape[0], H_true.shape[1]))

for i in range(N):
    sample_params = apep.copy()
    for param in params:
        sample_params[param] = data[param][chain_numbers[i], samp_numbers[i]]
    particles, weights = gm.dust_plume(sample_params)
    _, _, H = gm.smooth_histogram2d(particles, weights, sample_params)
    image_samples[i, :, :] = H

difference_samples = image_samples - H_true
stds = np.std(difference_samples, axis=0)
means = np.mean(difference_samples, axis=0)

fig, ax = plt.subplots(figsize=(4, 4))
plot = ax.pcolormesh(X, Y, stds, cmap='Greys', rasterized=True)
fig.colorbar(mappable=plot, label='Diff. Standard Deviation', shrink=0.8)

ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')

fig.savefig(f'{use_run}_model_std.png', dpi=400, bbox_inches='tight')
fig.savefig(f'{use_run}_model_std.pdf', dpi=400, bbox_inches='tight')

max_val = np.max(abs(means))

fig, ax = plt.subplots(figsize=(4, 4))
plot = ax.pcolormesh(X, Y, means, cmap='bwr', rasterized=True, vmin=-max_val, vmax=max_val)
fig.colorbar(mappable=plot, label='Mean Difference', shrink=0.8)
ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')
fig.savefig(f'{use_run}_model_means.png', dpi=400, bbox_inches='tight')
fig.savefig(f'{use_run}_model_means.pdf', dpi=400, bbox_inches='tight')



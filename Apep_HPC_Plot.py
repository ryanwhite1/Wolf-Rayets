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
# import pairplots

sim_run_2 = 'HPC/sim_run_2/173006276018319360_chains'
sim_run_1 = 'HPC/sim_run_1/173001196050679744'

use_run = sim_run_1

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
          "sigma":"Blur"}

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

corner_fig.savefig(f'{use_run}_corner.png', dpi=400)

ndim = len(data.keys())
params = list(data.keys())

fig, axes = plt.subplots(figsize=(7, 2 * ndim), nrows=ndim, sharex=True, gridspec_kw={'hspace':0})

for i in range(ndim):
    param_vals = data[params[i]]
    if len(param_vals.shape) > 1:
        for j in range(param_vals.shape[0]):
            axes[i].scatter(np.arange(len(param_vals[j, :])), param_vals[j, :], s=1)
    else:
        axes[i].scatter(np.arange(len(param_vals)), param_vals, s=1)
    axes[i].set(ylabel=param_labels[params[i]])
    
fig.savefig(f'{use_run}_chains.png', dpi=400)



# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:10:05 2024

@author: ryanw
"""

import pickle
import corner
import matplotlib.pyplot as plt
import numpy as np
# import pairplots

sim_run_2 = 'HPC/sim_run_2/173006267427540800_flat'

use_run = sim_run_2

with open(use_run, 'rb') as file:
    data = pickle.load(file)
    
param_labels = {"eccentricity":r"$e$",
          "open_angle":r"$\theta_{\rm OA}$",
          "phase":r"$\phi$"}

labels = [param_labels[label] for label in data.keys()]

# pairplots.pairplot(data)

corner.corner(data, 
              labels=labels,
              show_titles=True,
              smooth=0.8,
              title_fmt=".4f",
              color='tab:blue',
              use_math_text=True,
              levels=[0.393, 0.864],
              fill_contours=False)
              # quantiles=[0.16, 0.5, 0.84])

ndim = len(data.keys())
params = list(data.keys())

fig, axes = plt.subplots(nrows=ndim, sharex=True, gridspec_kw={'hspace':0})

for i in range(ndim):
    param_vals = data[params[i]]
    axes[i].scatter(np.arange(len(param_vals)), param_vals, s=1)
    axes[i].set(ylabel=param_labels[params[i]])



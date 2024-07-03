# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:05:12 2024

@author: ryanw
"""
import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, grad
import jax.lax as lax
import jax.scipy.stats as stats
import matplotlib.pyplot as plt
import time
import pickle
import numpyro, jax
import numpyro.distributions as dists
from numpyro.infer.util import initialize_model
import blackjax

import WR_Geom_Model as gm
import WR_binaries as wrb

run_num = "3"
path = f'HPC/run_{run_num}/'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

chains = []
num_samples = 0
parameters = ['eccentricity', 'inclination']
display_params = ['$e$', '$i$']

for file in files:
    with open(path + file, 'rb') as f:
        a = pickle.load(f)
        chains.append(a)
        num_samples += len(a['states'].position[parameters[0]])
        
all_chains = {parameter:jnp.array([chains[i]['states'].position[parameter] for i in range(len(files))]) for parameter in parameters}
        
all_positions = np.zeros((num_samples, len(parameters)))

run_total = 0
fig, axes = plt.subplots(ncols=len(parameters))
for i in range(len(files)):
    for j, parameter in enumerate(parameters):
        states = chains[i]['states'].position[parameter]
        axes[j].plot(np.arange(len(states)), states)
        
        all_positions[run_total:run_total + len(states), j] = states
        
    run_total += len(states)
    
for parameter in parameters:
    ess = blackjax.diagnostics.effective_sample_size(all_chains[parameter], chain_axis=0, sample_axis=1)
    print(parameter, "effective sample size is", ess)



import corner
figure = corner.corner(all_positions, 
                       labels=display_params,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True)
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:03:33 2024

@author: ryanw
"""
import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import numpy as np
import jax_tqdm as jtqdm
import jax.numpy as jnp
from jax import jit, vmap, grad
import jax.lax as lax
import jax.scipy.stats as stats
import blackjax
import matplotlib.pyplot as plt
import time
import pickle
import numpyro, jax
import numpyro.distributions as dists
from numpyro.infer.util import initialize_model
from glob import glob
from astropy.io import fits

import WR_Geom_Model as gm
import WR_binaries as wrb

# apep = wrb.apep.copy()

numpyro.enable_x64()

system = wrb.apep.copy()

### --- INFERENCE --- ###  
# n = 256     # standard
n = 600     # VISIR
# n = 898     # JWST
@jit
def smooth_histogram2d(particles, weights, stardata):
    im_size = n
    
    x = particles[0, :]
    y = particles[1, :]
    
    xbound, ybound = jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))
    bound = jnp.max(jnp.array([xbound, ybound])) * (1. + 2. / im_size)
    
    xedges, yedges = jnp.linspace(-bound, bound, im_size+1), jnp.linspace(-bound, bound, im_size+1)
    return gm.smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
@jit
def smooth_histogram2d_w_bins(particles, weights, stardata, xbins, ybins):
    im_size = n
    return gm.smooth_histogram2d_base(particles, weights, stardata, xbins, ybins, im_size)


pscale = 1000 * 23/512 # mas/pixel, (Yinuo's email said 45mas/px, but I think the FOV is 23x23 arcsec for a 512x512 image?)
vlt_years = [2016, 2017, 2018, 2024]
vlt_data = {}
flattened_vlt_data = {}
directory = "Data\\VLT"
fnames = glob(directory + "\\*.fits")

for i, fname in enumerate(fnames):
    
    data = fits.open(fname)[0].data
    
    length = data.shape[0]
    
    X = jnp.linspace(-1., 1., length) * pscale * length/2 / 1000
    Y = X.copy()
    
    xs, ys = jnp.meshgrid(X, Y)
    
    # data[280:320, 280:320] = 0.
    data = jnp.array(data)
    
    # data = data - jnp.median(data)
    data = data - jnp.percentile(data, 84)
    data = data/jnp.max(data)
    data = jnp.maximum(data, 0)
    data = jnp.abs(data)**0.5
    # data = data.at[280:320, 280:320].set(0.)
    vlt_data[vlt_years[i]] = data
    flattened_vlt_data[vlt_years[i]] = data.flatten()

big_flattened_data = jnp.concatenate([flattened_vlt_data[year] for year in vlt_years])
xbins = X
ybins = Y


params = {'eccentricity':[0., 0.95], 'inclination':[0, 180], 'open_angle':[0.1, 179]}
params_list = list(params.keys())

i = 0
param = 'eccentricity'
N = 60

flattened_years = {year:vlt_data[year].flatten() for year in vlt_years}

# for i, param in enumerate(params):
    
def man_loglike(value):
    starcopy = wrb.apep.copy()
    starcopy[param] = value
    
    chisq = 0
    
    for year in vlt_years:
        year_params = starcopy.copy()
        year_params['phase'] -= (2024 - year) / starcopy['period']
        samp_particles, samp_weights = gm.dust_plume(year_params)
        _, _, samp_H = smooth_histogram2d_w_bins(samp_particles, samp_weights, year_params, xbins, ybins)
        samp_H = gm.add_stars(xbins, ybins, samp_H, year_params)
        # samp_H.at[280:320, 280:320].set(0.)
        samp_H = samp_H.flatten()
        
        chisq += jnp.sum(((samp_H - flattened_years[year]) / 0.05)**2)
    
    # year = 2024
    # year_params = starcopy.copy()
    # year_params['phase'] -= (2024 - year) / starcopy['period']
    # samp_particles, samp_weights = gm.dust_plume(year_params)
    # _, _, samp_H = smooth_histogram2d_w_bins(samp_particles, samp_weights, year_params, xbins, ybins)
    # # samp_H = gm.add_stars(xbins, ybins, samp_H, year_params)
    # samp_H.at[280:320, 280:320].set(0.)
    # samp_H = samp_H.flatten()
    
    # chisq += jnp.sum(((samp_H - flattened_years[year]) / 0.05)**2)
    
    return -0.5 * chisq

like = jit(jax.value_and_grad(man_loglike))
numpyro_logLike = np.zeros(N)
manual_logLike = np.zeros(N)
param_vals = np.linspace(params[param][0], params[param][1], N)
dx = param_vals[1] - param_vals[0]

vals, grads = jnp.zeros(N), jnp.zeros(N)

from tqdm import tqdm 

for j in tqdm(range(N)):
    a, b = like(param_vals[j])
    vals = vals.at[j].set(a)
    grads = grads.at[j].set(b)



fig, axes = plt.subplots(ncols=2)

axes[0].plot(param_vals, vals)
axes[1].plot(param_vals, grads)

axes[1].axhline(0, c='k')
axes[1].axvline(wrb.apep['eccentricity'], c='tab:red')
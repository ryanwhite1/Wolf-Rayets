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
import jax_tqdm as jtqdm
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
directory = "Data/VLT"
fnames = glob(directory + "/*.fits")

for i, fname in enumerate(fnames):
    
    data = fits.open(fname)[0].data
    
    length = data.shape[0]
    
    X = jnp.linspace(-1., 1., length) * pscale * length/2 / 1000
    Y = X.copy()
    
    xs, ys = jnp.meshgrid(X, Y)
    
    data = jnp.array(data)
    
    data = data - jnp.percentile(data, 84)
    data = data/jnp.max(data)
    data = jnp.maximum(data, 0)
    data = jnp.abs(data)**0. 
    
    vlt_data[vlt_years[i]] = data
    flattened_vlt_data[vlt_years[i]] = data.flatten()

big_flattened_data = jnp.concatenate([flattened_vlt_data[year] for year in vlt_years])
xbins = X
ybins = Y

print("Data loaded in well.")

system_params = system.copy()

def apep_model():
    params = system_params.copy()
    # m1 = numpyro.sample("m1", dists.Normal(apep['m1'], 5.))
    # m2 = numpyro.sample("m2", dists.Normal(apep['m2'], 5.))
    params['eccentricity'] = numpyro.sample("eccentricity", dists.Uniform(0.4, 0.95))
    # params['eccentricity'] = numpyro.sample("eccentricity", dists.Normal(apep['eccentricity'], 0.05, support=dists.constraints.interval(0., 0.95)))
    params['inclination'] = numpyro.sample("inclination", dists.Uniform(0., 50.))
    params["asc_node"] = numpyro.sample("asc_node", dists.Uniform(100., 210.))
    params["arg_peri"] = numpyro.sample("arg_peri", dists.Uniform(-30., 50.))
    # open_angle = numpyro.sample("open_angle", dists.Normal(apep['open_angle'], 10.))
    params['open_angle'] = numpyro.sample("open_angle", dists.Uniform(70, 140.))
    # period = numpyro.sample("period", dists.Normal(apep['period'], 40.))
    # distance = numpyro.sample("distance", dists.Normal(apep['distance'], 500.))
    # params["windspeed1"] = numpyro.sample("windspeed1", dists.Uniform(300., 1500.))
    # windspeed2 = numpyro.sample("windspeed2", dists.Normal(apep['windspeed2'], 200.))
    params['turn_on'] = numpyro.sample("turn_on", dists.Uniform(-150., -60.))
    params['turn_off'] = numpyro.sample("turn_off", dists.Uniform(50., 179.))
    # oblate = numpyro.sample("oblate", dists.Uniform(0., 1.))
    # orb_sd = numpyro.sample("orb_sd", dists.Exponential(1./10.))
    # orb_amp = numpyro.sample("orb_amp", dists.Exponential(1./0.1))
    # orb_min = numpyro.sample("orb_min", dists.Uniform(0., 360.))
    # az_sd = numpyro.sample("az_sd", dists.Exponential(1./10.))
    # az_amp = numpyro.sample("az_amp", dists.Exponential(1./0.1))
    # az_min = numpyro.sample("az_min", dists.Uniform(0., 360.))
    # comp_incl = numpyro.sample('comp_incl', dists.Normal(apep['comp_incl'], 10))
    # comp_az = numpyro.sample('comp_az', dists.Normal(apep['comp_az'], 10))
    # comp_open = numpyro.sample("comp_open", dists.Normal(apep['comp_open'], 10.))
    # comp_reduction = numpyro.sample("comp_reduction", dists.Uniform(0., 2.))
    # comp_plume = numpyro.sample("comp_plume", dists.Uniform(0., 2.))
    # phase = numpyro.sample("phase", dists.Uniform(0., 1.))
    params['phase'] = numpyro.sample("phase", dists.Uniform(0., 0.99))
    params['sigma'] = numpyro.sample("sigma", dists.Uniform(1., 10.))
    
    
    
    for year in vlt_years:
        year_params = params.copy()
        year_params['phase'] -= (2024 - year) / params['period']
        samp_particles, samp_weights = gm.dust_plume(year_params)
        _, _, samp_H = smooth_histogram2d_w_bins(samp_particles, samp_weights, year_params, xbins, ybins)
        samp_H = gm.add_stars(xbins, ybins, samp_H, year_params)
        # samp_H.at[280:320, 280:320].set(0.)
        samp_H = samp_H.flatten()
        # samp_H = jnp.nan_to_num(samp_H, 1e4)
        data = flattened_vlt_data[year]
        with numpyro.plate('plate', len(data)):
            numpyro.sample(f'obs_{year}', dists.Normal(samp_H, 0.08), obs=data)



rand_time = int(time.time() * 1e8)
# rng_key = jax.random.key(rand_time)
rng_key = jax.random.PRNGKey(rand_time)

init_val = wrb.apep.copy()

num_chains = min(10, len(jax.devices()))
print("Num Chains = ", num_chains)

sampler = numpyro.infer.MCMC(numpyro.infer.NUTS(apep_model,
                                                max_tree_depth=5,
                                                init_strategy=numpyro.infer.initialization.init_to_value(values=init_val)
                                                ),
                              num_chains=num_chains,
                              num_samples=1000,
                              num_warmup=400,
                              progress_bar=False)
t1 = time.time()
print("Running HMC Now.")
sampler.run(rng_key)
print("HMC Finished successfully.")
t2 = time.time()
print("Time taken = ", t2 - t1)

results = sampler.get_samples(group_by_chain=True)
results_flat = sampler.get_samples()

run_num = 1
with open(f'HPC/run_{run_num}/{rand_time}', 'wb') as file:
    pickle.dump(results, file)
with open(f'HPC/run_{run_num}/{rand_time}_flat', 'wb') as file:
    pickle.dump(results_flat, file)

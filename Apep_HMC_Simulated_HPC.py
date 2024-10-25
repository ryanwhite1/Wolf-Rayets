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
import pickle

import WR_Geom_Model as gm
import WR_binaries as wrb

apep = wrb.apep.copy()
system_params = apep.copy()
# apep['sigma'] = 0.01

### --- INFERENCE --- ###  
particles, weights = gm.dust_plume(wrb.apep)
    
X, Y, H = gm.smooth_histogram2d(particles, weights, wrb.apep)
xbins = X[0, :]
ybins = Y[:, 0]
# X, Y, H = gm.spiral_grid(particles, weights, wrb.apep)
obs_err = 0.05 * np.max(H)
H += np.random.normal(0, obs_err, H.shape)
gm.plot_spiral(X, Y, H)



obs = H.flatten()
obs_err = obs_err * jnp.ones(len(obs))

fig, ax = plt.subplots()

ax.plot(jnp.arange(len(obs)), obs, lw=0.5)


# ### --- NUMPYRO --- ###
import numpyro, chainconsumer, jax
import numpyro.distributions as dists

def apep_model(Y, E):
    params = system_params.copy()
    # m1 = numpyro.sample("m1", dists.Normal(apep['m1'], 5.))
    # m2 = numpyro.sample("m2", dists.Normal(apep['m2'], 5.))
    # bounded_norm = dists.Normal(apep['eccentricity'], 0.05)
    # bounded_norm.support = dists.constraints.interval(0., 0.95)
    # params['eccentricity'] = numpyro.sample("eccentricity", bounded_norm)
    params['eccentricity'] = numpyro.sample("eccentricity", dists.Uniform(0.4, 0.95))
    # params['eccentricity'] = numpyro.sample("eccentricity", dists.Normal(apep['eccentricity'], 0.05, support=dists.constraints.interval(0., 0.95)))
    # params['inclination'] = numpyro.sample("inclination", dists.Normal(apep['inclination'], 20.))
    # asc_node = numpyro.sample("asc_node", dists.Normal(apep['asc_node'], 20.))
    # arg_peri = numpyro.sample("arg_peri", dists.Normal(apep['arg_peri'], 20.))
    # open_angle = numpyro.sample("open_angle", dists.Normal(apep['open_angle'], 10.))
    params['open_angle'] = numpyro.sample("open_angle", dists.Uniform(70, 140.))
    # period = numpyro.sample("period", dists.Normal(apep['period'], 40.))
    # distance = numpyro.sample("distance", dists.Normal(apep['distance'], 500.))
    # windspeed1 = numpyro.sample("windspeed1", dists.Normal(apep['windspeed1'], 200.))
    # windspeed2 = numpyro.sample("windspeed2", dists.Normal(apep['windspeed2'], 200.))
    # turn_on = numpyro.sample("turn_on", dists.Normal(apep['turn_on'], 10.))
    # turn_off = numpyro.sample("turn_off", dists.Normal(apep['turn_off'], 10.))
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
    # sigma = numpyro.sample("sigma", dists.Uniform(0.01, 10.))
    # histmax = numpyro.sample("histmax", dists.Uniform(0., 1.))
        
    samp_particles, samp_weights = gm.dust_plume(params)
    _, _, samp_H = gm.smooth_histogram2d_w_bins(samp_particles, samp_weights, params, xbins, ybins)
    samp_H = samp_H.flatten()
    # samp_H = jnp.nan_to_num(samp_H, 1e4)
    with numpyro.plate('plate', len(obs)):
        numpyro.sample('obs', dists.Normal(samp_H, E), obs=Y)

rand_time = int(time.time() * 1e8)
rng_key = jax.random.key(rand_time)

init_params = apep.copy()

sampler = numpyro.infer.MCMC(numpyro.infer.NUTS(apep_model,
                                                init_strategy=numpyro.infer.initialization.init_to_value(values=init_params)
                                                ),
                              num_chains=5,
                              num_samples=10,
                              num_warmup=10)

sampler.run(rng_key, obs, obs_err)

results = sampler.get_samples()

run_num = 1
with open(f'HPC/sim_run_{run_num}/{rand_time}', 'wb') as file:
    pickle.dump(results, file)



    


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
directory = "Data\\VLT"
fnames = glob(directory + "\\*.fits")

for i, fname in enumerate(fnames):
    
    data = fits.open(fname)[0].data
    
    length = data.shape[0]
    
    X = jnp.linspace(-1., 1., length) * pscale * length/2 / 1000
    Y = X.copy()
    
    xs, ys = jnp.meshgrid(X, Y)
    
    data[280:320, 280:320] = 0.
    data = jnp.array(data)
    
    # data = data - jnp.median(data)
    data = data - jnp.percentile(data, 84)
    data = data/jnp.max(data)
    data = jnp.maximum(data, 0)
    vlt_data[vlt_years[i]] = jnp.abs(data)**0.5

xbins = X
ybins = Y

H = vlt_data[2018]
obs_err = 0.01

fig, ax = plt.subplots()
ax.imshow(H)
ax.invert_yaxis()

H = np.array(H)
# H[280:320, 280:320] = 0.

fig, ax = plt.subplots()
ax.imshow(H)
ax.invert_yaxis()


obs = H.flatten()
obs_err = obs_err * jnp.ones(len(obs))


fig, ax = plt.subplots()

ax.plot(jnp.arange(len(obs)), obs, lw=0.5)


# fig, ax = plt.subplots()

# ax.plot(jnp.arange(len(obs)), obs**3, lw=0.5)

system_params = system.copy()
system_params['histmax'] = 0.7

particles, weights = gm.dust_plume(system_params)
    
X, Y, H = smooth_histogram2d(particles, weights, system_params)
# xbins = X[0, :] * 1.5
# ybins = Y[:, 0] * 1.5
X, Y, H = smooth_histogram2d_w_bins(particles, weights, system_params, xbins, ybins)
# H = gm.add_stars(xbins, ybins, H, system_params)
# X, Y, H = gm.spiral_grid(particles, weights, wrb.apep)
# obs_err = 0.01 * np.max(H)
# H += np.random.normal(0, obs_err, H.shape)

H = np.array(H)
H[280:320, 280:320] = 0.

fig, ax = plt.subplots()
ax.plot(jnp.arange(len(obs)), H.flatten() - obs, lw=0.5)

fig, ax = plt.subplots()
ax.imshow(H - vlt_data[2024])
ax.invert_yaxis()


def apep_model(Y, E):
    params = system_params.copy()
    # m1 = numpyro.sample("m1", dists.Normal(apep['m1'], 5.))
    # m2 = numpyro.sample("m2", dists.Normal(apep['m2'], 5.))
    # params['eccentricity'] = numpyro.sample("eccentricity", dists.Normal(system_params['eccentricity'], 0.1))
    params['eccentricity'] = numpyro.sample("eccentricity", dists.Uniform(0.6, 0.9))
    # params['inclination'] = numpyro.sample("inclination", dists.Normal(system['inclination'], 5.))
    # asc_node = numpyro.sample("asc_node", dists.Normal(apep['asc_node'], 20.))
    # arg_peri = numpyro.sample("arg_peri", dists.Normal(apep['arg_peri'], 20.))
    # open_angle = numpyro.sample("open_angle", dists.Normal(apep['open_angle'], 10.))
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
    # phase = numpyro.sample("phase", dists.Normal(apep['phase'], 0.1))
    # sigma = numpyro.sample("sigma", dists.Uniform(0.01, 10.))
    # histmax = numpyro.sample("histmax", dists.Uniform(0., 1.))
        
    
    for year in vlt_years:
        year_params = params.copy()
        year_params['phase'] -= (2024 - year) / params['period']
        samp_particles, samp_weights = gm.dust_plume(year_params)
        _, _, samp_H = smooth_histogram2d_w_bins(samp_particles, samp_weights, year_params, xbins, ybins)
        # samp_H = gm.add_stars(xbins, ybins, samp_H, year_params)
        samp_H.at[280:320, 280:320].set(0.)
        samp_H = samp_H.flatten()
        # samp_H = jnp.nan_to_num(samp_H, 1e4)
        with numpyro.plate('plate', len(obs)):
            numpyro.sample(f'obs_{year}', dists.Normal(samp_H, E), obs=Y[year].flatten())

# h = numpyro.render_model(apep_model, model_args=(vlt_data, obs_err))
# h.view()



rand_time = int(time.time() * 1e8)
rng_key = jax.random.key(rand_time)

rng_key, init_key = jax.random.split(rng_key)
init_params, potential_fn_gen, *_ = initialize_model(
    init_key,
    apep_model,
    model_args=(obs, obs_err),
    dynamic_args=True,
    init_strategy=numpyro.infer.initialization.init_to_value(values=system_params)
)

logdensity_fn = lambda position: -potential_fn_gen(obs, obs_err)(position)
initial_position = init_params.z


### below is numpyro sampling

shifted = system_params.copy()
# shifted['eccentricity'] = 0.7

# sampler = numpyro.infer.MCMC(numpyro.infer.NUTS(apep_model, 
#                                                 target_accept_prob=0.3,
#                                                 # regularize_mass_matrix=False,
#                                                 find_heuristic_step_size=True,
#                                                 max_tree_depth=5,
#                                                 forward_mode_differentiation=True,
#                                                 init_strategy=numpyro.infer.initialization.init_to_value(values=shifted)),
#                               num_chains=1,
#                               num_samples=300,
#                               num_warmup=100,
#                               progress_bar=True)
# sampler = numpyro.infer.MCMC(numpyro.infer.BarkerMH(apep_model,
#                                                     target_accept_prob=0.6,
#                                                     step_size=0.01,
#                                                     init_strategy=numpyro.infer.initialization.init_to_value(values=shifted)),
#                               num_chains=1,
#                               num_samples=300,
#                               num_warmup=100,
#                               progress_bar=True)
sampler = numpyro.infer.MCMC(numpyro.infer.SA(apep_model,
                                              init_strategy=numpyro.infer.initialization.init_to_value(values=shifted)),
                              num_chains=1,
                              num_samples=300,
                              num_warmup=100,
                              progress_bar=True)
sampler.run(jax.random.PRNGKey(1), obs, obs_err)
results = sampler.get_samples()

import chainconsumer
C = chainconsumer.ChainConsumer()
C.add_chain(results, name='MCMC Results')
C.plotter.plot()
C.plotter.plot_walks()

fig, ax = plt.subplots()
ax.plot(C.chains[0].chain)


# ### below is blackjax sampling

# num_warmup = 300
# integration_steps = 4
# # adapt = blackjax.window_adaptation(
# #     blackjax.nuts, logdensity_fn, target_acceptance_rate=0.5, progress_bar=True,
# #     initial_step_size=1./integration_steps, max_num_doublings=3
# # )
# adapt = blackjax.window_adaptation(
#     blackjax.hmc, logdensity_fn, target_acceptance_rate=0.6, progress_bar=True, 
#     num_integration_steps=integration_steps, initial_step_size=1./integration_steps
# )
# rng_key, warmup_key = jax.random.split(rng_key)
# print("warm up")
# (last_state, parameters), _ = adapt.run(warmup_key, initial_position, num_warmup)
# print("warm up done")
# # kernel = blackjax.nuts(logdensity_fn, **parameters).step
# parameters['step_size'] = 1 / integration_steps
# # parameters['step_size'] *= 6
# kernel = blackjax.hmc(logdensity_fn, **parameters).step

# # print(a)

# def inference_loop(rng_key, kernel, initial_state, num_samples):
#     @jax.jit
#     def one_step(state, rng_key):
#         state, info = kernel(rng_key, state)
#         return state, (state, info)

#     keys = jax.random.split(rng_key, num_samples)
#     _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

#     return states, (
#         infos.acceptance_rate,
#         infos.is_divergent,
#         infos.num_integration_steps,
#     )

# # inference_loop_multiple_chains = jax.pmap(inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3))

# num_sample = 400

# ### single chain/core:
# rng_key, sample_key = jax.random.split(rng_key)
# t1 = time.time()
# print("Starting HMC...")
# states, infos = inference_loop(sample_key, kernel, last_state, num_sample)
# print("HMC Completed in ", time.time() - t1, "s")
# _ = states.position["eccentricity"].block_until_ready()


# ### pmapped:
# # num_chains = min(max(1, len(jax.devices())), 10)
# # rng_key, sample_key = jax.random.split(rng_key)
# # sample_keys = jax.random.split(sample_key, num_chains)

# # hmc = blackjax.hmc(logdensity_fn, **parameters)
# # initial_positions = {"eccentricity": jnp.ones(num_chains)*apep['eccentricity'], 
# #                      "inclination": jnp.ones(num_chains)*apep['inclination']}
# # initial_states = jax.vmap(hmc.init, in_axes=(0))(initial_positions)
# # t1 = time.time()
# # print("Starting HMC...")
# # pmap_states = inference_loop_multiple_chains(
# #     sample_keys, kernel, initial_states, num_sample
# # )
# # print("HMC Completed in ", time.time() - t1, "s")
# # states, infos = pmap_states
# # _ = states.position["eccentricity"].block_until_ready()


# acceptance_rate = np.mean(infos[0])
# num_divergent = np.mean(infos[1])

# print(f"Average acceptance rate: {acceptance_rate:.2f}")
# print(f"There were {100*num_divergent:.2f}% divergent transitions")

# run_num = 6
# pickle_samples = {"states":states, "infos":infos}
# with open(f'HPC/run_{run_num}/{rand_time}', 'wb') as file:
#     pickle.dump(pickle_samples, file)

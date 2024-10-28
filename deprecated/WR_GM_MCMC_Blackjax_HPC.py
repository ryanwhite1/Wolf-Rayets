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

import WR_Geom_Model as gm
import WR_binaries as wrb

# apep = wrb.apep.copy()

system = wrb.apep_aniso2.copy()

### --- INFERENCE --- ###  
particles, weights = gm.dust_plume(system)
    
X, Y, H = gm.smooth_histogram2d(particles, weights, system)
xbins = X[0, :] * 1.5
ybins = Y[:, 0] * 1.5
X, Y, H = gm.smooth_histogram2d_w_bins(particles, weights, system, xbins, ybins)
H = gm.add_stars(xbins, ybins, H, system)
# X, Y, H = gm.spiral_grid(particles, weights, wrb.apep)
obs_err = 0.01 * np.max(H)
H += np.random.normal(0, obs_err, H.shape)
gm.plot_spiral(X, Y, H)



obs = H.flatten()
obs_err = obs_err * jnp.ones(len(obs))

fig, ax = plt.subplots()

ax.plot(jnp.arange(len(obs)), obs, lw=0.5)


# fig, ax = plt.subplots()

# ax.plot(jnp.arange(len(obs)), obs**3, lw=0.5)

system_params = system.copy()


def apep_model(Y, E):
    params = system_params.copy()
    # m1 = numpyro.sample("m1", dists.Normal(apep['m1'], 5.))
    # m2 = numpyro.sample("m2", dists.Normal(apep['m2'], 5.))
    params['eccentricity'] = numpyro.sample("eccentricity", dists.Normal(system['eccentricity'], 0.08))
    params['inclination'] = numpyro.sample("inclination", dists.Normal(system['inclination'], 10.))
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
        
    samp_particles, samp_weights = gm.dust_plume(params)
    _, _, samp_H = gm.smooth_histogram2d_w_bins(samp_particles, samp_weights, params, xbins, ybins)
    samp_H = gm.add_stars(xbins, ybins, samp_H, params)
    samp_H = samp_H.flatten()
    # samp_H = jnp.nan_to_num(samp_H, 1e4)
    with numpyro.plate('plate', len(obs)):
        numpyro.sample('obs', dists.Normal(samp_H, E), obs=Y)



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



num_warmup = 300
integration_steps = 10
adapt = blackjax.window_adaptation(
    blackjax.nuts, logdensity_fn, target_acceptance_rate=0.5, progress_bar=True,
    initial_step_size=1./integration_steps, max_num_doublings=5
)
# adapt = blackjax.window_adaptation(
#     blackjax.hmc, logdensity_fn, target_acceptance_rate=0.6, progress_bar=True, 
#     num_integration_steps=integration_steps, initial_step_size=1./integration_steps
# )
rng_key, warmup_key = jax.random.split(rng_key)
print("warm up")
(last_state, parameters), _ = adapt.run(warmup_key, initial_position, num_warmup)
print("warm up done")
kernel = blackjax.nuts(logdensity_fn, **parameters).step
# parameters['step_size'] = 1 / integration_steps
# parameters['step_size'] *= 6
# kernel = blackjax.hmc(logdensity_fn, **parameters).step

# print(a)

def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, (
        infos.acceptance_rate,
        infos.is_divergent,
        infos.num_integration_steps,
    )

# inference_loop_multiple_chains = jax.pmap(inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3))

num_sample = 400

### single chain/core:
rng_key, sample_key = jax.random.split(rng_key)
t1 = time.time()
print("Starting HMC...")
states, infos = inference_loop(sample_key, kernel, last_state, num_sample)
print("HMC Completed in ", time.time() - t1, "s")
_ = states.position["eccentricity"].block_until_ready()


### pmapped:
# num_chains = min(max(1, len(jax.devices())), 10)
# rng_key, sample_key = jax.random.split(rng_key)
# sample_keys = jax.random.split(sample_key, num_chains)

# hmc = blackjax.hmc(logdensity_fn, **parameters)
# initial_positions = {"eccentricity": jnp.ones(num_chains)*apep['eccentricity'], 
#                      "inclination": jnp.ones(num_chains)*apep['inclination']}
# initial_states = jax.vmap(hmc.init, in_axes=(0))(initial_positions)
# t1 = time.time()
# print("Starting HMC...")
# pmap_states = inference_loop_multiple_chains(
#     sample_keys, kernel, initial_states, num_sample
# )
# print("HMC Completed in ", time.time() - t1, "s")
# states, infos = pmap_states
# _ = states.position["eccentricity"].block_until_ready()


acceptance_rate = np.mean(infos[0])
num_divergent = np.mean(infos[1])

print(f"Average acceptance rate: {acceptance_rate:.2f}")
print(f"There were {100*num_divergent:.2f}% divergent transitions")

run_num = 5
pickle_samples = {"states":states, "infos":infos}
with open(f'HPC/run_{run_num}/{rand_time}', 'wb') as file:
    pickle.dump(pickle_samples, file)

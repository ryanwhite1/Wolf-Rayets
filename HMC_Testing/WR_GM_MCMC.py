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


# fig, ax = plt.subplots()

# ax.plot(jnp.arange(len(obs)), obs**3, lw=0.5)


### --- EMCEE --- ###


# # @jit
# def log_prior(state):
#     # m1 = jnp.heaviside(state['m1'], 0) * jnp.heaviside(200 - state['m1'], 1)
#     # m2 = jnp.heaviside(state['m2'], 0) * jnp.heaviside(200 - state['m2'], 1)
#     # period = jnp.heaviside(state['period'], 0) * jnp.heaviside(1e3 - state['period'], 1)
#     # eccentricity = jnp.heaviside(state['eccentricity'], 1) * jnp.heaviside(1 - state['eccentricity'], 0)
#     # inclination = jnp.heaviside(360 - state['inclination'], 1) * (1 - jnp.heaviside(-state['inclination'] - 360, 1))
#     # asc_node = jnp.heaviside(360 - state['asc_node'], 1) * (1 - jnp.heaviside(-state['asc_node'] - 360, 1))
#     # arg_peri = jnp.heaviside(360 - state['arg_peri'], 1) * (1 - jnp.heaviside(-state['arg_peri'] - 360, 1))
#     # open_angle = jnp.heaviside(180 - state['open_angle'], 0) * jnp.heaviside(state['open_angle'], 0)
#     # distance = jnp.heaviside(state['distance'], 0)
#     # turn_on = jnp.heaviside(180 + state['turn_on'], 1) * (1 - jnp.heaviside(-state['turn_on'] - 180, 0))
#     # turn_off = jnp.heaviside(180 + state['turn_off'], 1) * (1 - jnp.heaviside(-state['turn_off'] - 180, 0))
    
#     # return (1. - m1*m2*period*eccentricity*inclination*asc_node*arg_peri*open_angle*distance*turn_on*
#     #         turn_off) * -jnp.inf
    
#     array = jnp.array([0., -jnp.inf])
#     eccentricity = jnp.heaviside(state[0], 1) * jnp.heaviside(1 - state[0], 0)
#     inclination = jnp.heaviside(360 - state[1], 1) * (1 - jnp.heaviside(-state[1] - 360, 1))
#     asc_node = jnp.heaviside(360 - state[2], 1) * (1 - jnp.heaviside(-state[2] - 360, 1))
#     open_angle = jnp.heaviside(180 - state[3], 0) * jnp.heaviside(state[3], 0)
    
#     # # print(eccentricity, inclination, asc_node, open_angle)
#     # if not (1 - eccentricity*inclination*asc_node*open_angle):
#     #     return 0. 
#     # else:
#     #     return -jnp.inf
#     # a = (1. - eccentricity*inclination*asc_node*open_angle) * -jnp.inf
#     num = 1 - [eccentricity*inclination*asc_node*open_angle][0]
#     num = jnp.array(num, int)
#     return array[num]
    
#     # a = (1. - eccentricity*inclination*asc_node*open_angle) * -jnp.inf
#     # return -np.min([np.nan_to_num(a), np.inf])
# # @jit 
# def log_likelihood(state, obs, obs_err):
    
#     data_dict = apep.copy()
#     data_dict['eccentricity'] = state[0]
#     data_dict['inclination'] = state[1]
#     data_dict['asc_node'] = state[2]
#     data_dict['open_angle'] = state[3]
    
#     particles, weights = dust_plume(data_dict)
#     _, _, model = spiral_grid(particles, weights, data_dict)
#     model = model.flatten()
#     return -0.5 * jnp.sum((obs - model)**2 / obs_err**2)

# @jit 
# def log_prob(state, obs, obs_err):
#     lp = log_prior(state)
#     isfinite = jnp.array(jnp.isfinite(lp), int)
#     return_arr = jnp.array([-jnp.inf, lp + log_likelihood(state, obs, obs_err)])
#     return return_arr[isfinite]


# nwalkers = 10

# pos = np.array([apep['eccentricity'], apep['inclination'], apep['asc_node'], apep['open_angle']])
# ndim = len(pos)
# pos = pos * np.ones((nwalkers, ndim))
# pos += 1e-1 * np.random.normal(0, 0.5, pos.shape)

# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(obs, obs_err))
# sampler.run_mcmc(pos, 1000, progress=True);


# fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
# samples = sampler.get_chain()
# labels = ["e", "i", "an", "oa"]
# for i in range(ndim):
#     ax = axes[i]
#     ax.plot(samples[:, :, i], "k", alpha=0.3)
#     ax.set_xlim(0, len(samples))
#     ax.set_ylabel(labels[i])
#     ax.yaxis.set_label_coords(-0.1, 0.5)
    
    
# flat_samples = sampler.get_chain(discard=300, flat=True)
# import corner
# import jax
# labels = ['ecc', 'incl', 'asc_node', 'op_ang']
# truths = np.array([apep['eccentricity'], apep['inclination'], apep['asc_node'], apep['open_angle']])
# fig = corner.corner(flat_samples, labels=labels, truths=truths)





# ### --- BLACKJAX --- ###
# def log_prior(state):
#     array = jnp.array([0., -jnp.inf])
#     m1 = jnp.heaviside(state['m1'], 0) * jnp.heaviside(200 - state['m1'], 1)
#     m2 = jnp.heaviside(state['m2'], 0) * jnp.heaviside(200 - state['m2'], 1)
#     period = jnp.heaviside(state['period'], 0) * jnp.heaviside(1e3 - state['period'], 1)
#     eccentricity = jnp.heaviside(state['eccentricity'], 1) * jnp.heaviside(1 - state['eccentricity'], 0)
#     inclination = jnp.heaviside(360 - state['inclination'], 1) * (1 - jnp.heaviside(-state['inclination'] - 360, 1))
#     asc_node = jnp.heaviside(360 - state['asc_node'], 1) * (1 - jnp.heaviside(-state['asc_node'] - 360, 1))
#     arg_peri = jnp.heaviside(360 - state['arg_peri'], 1) * (1 - jnp.heaviside(-state['arg_peri'] - 360, 1))
#     open_angle = jnp.heaviside(180 - state['open_angle'], 0) * jnp.heaviside(state['open_angle'], 0)
#     distance = jnp.heaviside(state['distance'], 0)
#     turn_on = jnp.heaviside(180 + state['turn_on'], 1) * (1 - jnp.heaviside(-state['turn_on'] - 180, 0))
#     turn_off = jnp.heaviside(180 + state['turn_off'], 1) * (1 - jnp.heaviside(-state['turn_off'] - 180, 0))
    
#     num = 1 - [m1*m2*period*eccentricity*inclination*asc_node*arg_peri*open_angle*distance*turn_on*turn_off][0]
#     num = jnp.array(num, int)
    
#     return array[num]

# def log_likelihood(state, obs, obs_err):
#     particles, weights = dust_plume(state)
#     _, _, model = spiral_grid(particles, weights, state)
#     model = model.flatten()
#     return -0.5 * jnp.sum((obs - model)**2 / obs_err**2)

# def log_prob(state, obs=obs, obs_err=obs_err):
#     lp = log_prior(state)
#     isfinite = jnp.array(jnp.isfinite(lp), int)
#     return_arr = jnp.array([-jnp.inf, lp + log_likelihood(state, obs, obs_err)])
#     return return_arr[isfinite]

# import blackjax 
# inverse_mass_matrix = jnp.ones(len(apep)) * 0.05
# step_size = 1e-3
# hmc = blackjax.nuts(log_prob, step_size, inverse_mass_matrix)

# initial_position = apep
# state = hmc.init(initial_position)
# import jax
# rng_key = jax.random.key(0)
# step = jit(hmc.step)


# def inference_loop(rng_key, kernel, initial_state, num_samples):

#     @jax.jit
#     def one_step(state, rng_key):
#         state, _ = kernel(rng_key, state)
#         return state, state

#     keys = jax.random.split(rng_key, num_samples)
#     _, states = jax.lax.scan(one_step, initial_state, keys)

#     return states

# states = inference_loop(rng_key, step, state, 1000)

# mcmc_samples = states.position

# samples = np.ones((len(list(mcmc_samples.keys())), len(mcmc_samples[list(mcmc_samples.keys())[0]])))
# for i, key in enumerate(mcmc_samples.keys()):
#     samples[i, :] = mcmc_samples[key]

# import corner
# corner.corner(samples)






# ### --- NUMPYRO --- ###
import numpyro, chainconsumer, jax
import numpyro.distributions as dists

num_chains = 1

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



init_params = apep.copy()
# # init_params_arr = init_params.copy()
# # for key in init_params.keys():
# #     init_params_arr[key] = jnp.ones(num_chains) * init_params_arr[key]


init_params = numpyro.infer.util.constrain_fn(apep_model, (obs, obs_err), {}, init_params)
# # sampler = numpyro.infer.MCMC(numpyro.infer.NUTS(apep_model, 
# #                                                 target_accept_prob=0.2,
# #                                                 init_strategy=numpyro.infer.initialization.init_to_value(values=init_params)),
# #                               num_chains=1,
# #                               num_samples=300,
# #                               num_warmup=20,
# #                               progress_bar=True)
# # sampler = numpyro.infer.MCMC(numpyro.infer.HMC(apep_model, 
# #                                                 target_accept_prob=0.3,
# #                                                 step_size=2*jnp.pi,
# #                                                 adapt_step_size=False,
# #                                                 init_strategy=numpyro.infer.initialization.init_to_value(values=init_params),
# #                                                 forward_mode_differentiation=True),
# #                               num_chains=1,
# #                               num_samples=300,
# #                               num_warmup=20,
# #                               progress_bar=True)
# sampler = numpyro.infer.MCMC(numpyro.infer.HMC(apep_model,
#                                                init_strategy=numpyro.infer.initialization.init_to_value(values=init_params)),
#                               num_chains=1,
#                               num_samples=300,
#                               num_warmup=20,
#                               progress_bar=True)
# # sampler = numpyro.infer.MCMC(numpyro.infer.HMC(apep_model, 
# #                                                 target_accept_prob=0.3,
# #                                                 init_strategy=numpyro.infer.initialization.init_to_value(values=init_params),
# #                                                 forward_mode_differentiation=True,
# #                                                 step_size=1e-3),
# #                               num_chains=1,
# #                               num_samples=300,
# #                               num_warmup=20,
# #                               progress_bar=True)
sampler = numpyro.infer.MCMC(numpyro.infer.NUTS(apep_model),
                              num_chains=1,
                              num_samples=1000,
                              num_warmup=300)
# sampler = numpyro.infer.MCMC(numpyro.infer.SA(apep_model, init_strategy=numpyro.infer.initialization.init_to_value(values=init_params)),
#                               num_chains=num_chains,
#                               num_samples=300,
#                               num_warmup=100)
# sampler = numpyro.infer.MCMC(numpyro.infer.SA(apep_model,
#                                               init_strategy=numpyro.infer.initialization.init_to_value(values={'eccentricity':0.5})),
#                               num_chains=num_chains,
#                               num_samples=1000,
#                               num_warmup=300)
# sampler = numpyro.infer.MCMC(numpyro.infer.SA(apep_model),
#                               num_chains=num_chains,
#                               num_samples=1000,
#                               num_warmup=300)
sampler.run(jax.random.PRNGKey(1), obs, obs_err)

results = sampler.get_samples()
C = chainconsumer.ChainConsumer()
C.add_chain(results, name='MCMC Results')
C.plotter.plot(truth=apep)

# C.plotter.plot_walks()

nparams = len(results.keys())
param_names = list(results.keys())

fig, axes = plt.subplots(nrows=nparams, sharex=True, gridspec_kw={'hspace':0})

for i in range(nparams):
    vals = results[param_names[i]]
    axes[i].scatter(np.arange(len(vals)), vals, s=0.1)
    axes[i].set(ylabel=param_names[i])

# maxlike = apep.copy()
# for key in results.keys():
#     maxlike[key] = np.median(results[key])


# samp_particles, samp_weights = gm.dust_plume(maxlike)
# X, Y, samp_H = gm.spiral_grid(samp_particles, samp_weights, maxlike)
# gm.plot_spiral(X, Y, samp_H)

# a = jax.jit(numpyro.infer.util.potential_energy(apep_model, (obs, obs_err), {}, {"eccentricity":0.7}))
# @jit
# def a(e):
#     blah = numpyro.infer.util.log_likelihood(apep_model, {"eccentricity":e}, obs, obs_err)
#     print(blah['y'])
#     return blah



# def man_loglike(e):
#     starcopy = apep.copy()
#     starcopy['eccentricity'] = e
#     samp_particles, samp_weights = gm.dust_plume(starcopy)
#     _, _, samp_H = gm.smooth_histogram2d(samp_particles, samp_weights, starcopy)
#     # _, _, samp_H = gm.spiral_grid(samp_particles, samp_weights, starcopy)
#     samp_H = samp_H.flatten()
    
#     return -0.5 * jnp.sum(jnp.square((samp_H - obs) / obs_err))

# params = {'eccentricity':[0, 0.95], 'inclination':[0, 180], 'open_angle':[0.1, 179]}
# params_list = list(params.keys())



# n = 50
# fig, axes = plt.subplots(ncols=2, nrows=len(params), figsize=(12, 4*len(params_list)))
# for i, param in enumerate(params):
    
#     def man_loglike(value):
#         starcopy = apep.copy()
#         starcopy[param] = value
#         samp_particles, samp_weights = gm.dust_plume(starcopy)
#         _, _, samp_H = gm.smooth_histogram2d_w_bins(samp_particles, samp_weights, starcopy, X[0, :], Y[:, 0])
#         # _, _, samp_H = gm.spiral_grid(samp_particles, samp_weights, starcopy)
#         samp_H = samp_H.flatten()
        
#         return -0.5 * jnp.sum(((samp_H - obs) / obs_err)**2)

#     like = jit(vmap(jax.value_and_grad(man_loglike)))
    
#     numpyro_logLike = np.zeros(n)
#     manual_logLike = np.zeros(n)
#     param_vals = np.linspace(params[param][0], params[param][1], n)
#     dx = param_vals[1] - param_vals[0]

#     vals, grads = like(param_vals)
    
#     # normalize = lambda x: (x-x.min())/(x.max()-x.min())
    
#     ax1, ax2 = axes[i, :]
    
#     ax1.plot(param_vals, vals)
#     ax1.axvline(apep[param])
#     ax2.plot(param_vals, grads, label='JAX Grad')
#     ax2.plot(param_vals, np.gradient(vals, dx), label='Finite Diff Grad')
#     ax2.axvline(apep[param], c='tab:purple', ls='--', label='True Value')
#     ax2.axhline(0, c='k')
#     if i == 0:
#         ax2.legend()
#         ax1.set_title('Log Likelihood')
#         ax2.set_title('Log Likelihood Gradient')
#     for ax in [ax1, ax2]:
#         ax.set(xlabel=param)
    


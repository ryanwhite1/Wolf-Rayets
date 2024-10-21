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


def log_prior(theta):
    [e, phase, open_ang] = theta
    
    lp = 0.
    
    # eccentricity check
    if e > 0.95 or e < 0.:
        return -np.inf
    # orbital phase check
    if phase > 1. or phase < 0.:
        return -np.inf
    # open angle check
    if open_ang > 170. or open_ang < 10.:
        return -np.inf
    open_ang_unc = 15
    lp += np.log(1.0 / (np.sqrt(2 * np.pi) * open_ang_unc)) - 0.5 * (open_ang - apep['open_angle'])**2 / (open_ang_unc**2)
    
    return lp
    
def log_likelihood(theta):
    [e, phase, open_ang] = theta
    
    data_dict = apep.copy()
    
    data_dict['eccentricity'] = e
    data_dict['phase'] = phase
    data_dict['open_ang'] = open_ang
    
    particles, weights = gm.dust_plume(data_dict)
    _, _, model = gm.smooth_histogram2d_w_bins(particles, weights, data_dict, xbins, ybins)
    model = model.flatten()
    
    return -0.5 * jnp.sum((obs - model)**2 / obs_err**2)

def log_prob(theta, obs, obs_err):
    lp = log_prior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


nwalkers = 10

params = ['eccentricity', 'phase', 'open_angle']

pos = np.array([apep[param] for param in params])
ndim = len(pos)
pos = pos * np.ones((nwalkers, ndim))
pos += 1e-4 * np.random.normal(0, 1, pos.shape)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(obs, obs_err))
sampler.run_mcmc(pos, 1000, progress=True);


fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = params
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
    
    
flat_samples = sampler.get_chain(discard=300, flat=True)
import corner
import jax
# labels = ['ecc', 'incl', 'asc_node', 'op_ang']
truths = np.array([apep[param] for param in params])
fig = corner.corner(flat_samples, 
                    labels=labels, 
                    truths=truths,
                    show_titles=True)











# # ### --- NUMPYRO --- ###
# import numpyro, chainconsumer, jax
# import numpyro.distributions as dists

# num_chains = 1

# def apep_model(Y, E):
#     params = system_params.copy()
#     # m1 = numpyro.sample("m1", dists.Normal(apep['m1'], 5.))
#     # m2 = numpyro.sample("m2", dists.Normal(apep['m2'], 5.))
#     # bounded_norm = dists.Normal(apep['eccentricity'], 0.05)
#     # bounded_norm.support = dists.constraints.interval(0., 0.95)
#     # params['eccentricity'] = numpyro.sample("eccentricity", bounded_norm)
#     params['eccentricity'] = numpyro.sample("eccentricity", dists.Uniform(0.4, 0.95))
#     # params['eccentricity'] = numpyro.sample("eccentricity", dists.Normal(apep['eccentricity'], 0.05, support=dists.constraints.interval(0., 0.95)))
#     # params['inclination'] = numpyro.sample("inclination", dists.Normal(apep['inclination'], 20.))
#     # asc_node = numpyro.sample("asc_node", dists.Normal(apep['asc_node'], 20.))
#     # arg_peri = numpyro.sample("arg_peri", dists.Normal(apep['arg_peri'], 20.))
#     # open_angle = numpyro.sample("open_angle", dists.Normal(apep['open_angle'], 10.))
#     params['open_angle'] = numpyro.sample("open_angle", dists.Uniform(70, 140.))
#     # period = numpyro.sample("period", dists.Normal(apep['period'], 40.))
#     # distance = numpyro.sample("distance", dists.Normal(apep['distance'], 500.))
#     # windspeed1 = numpyro.sample("windspeed1", dists.Normal(apep['windspeed1'], 200.))
#     # windspeed2 = numpyro.sample("windspeed2", dists.Normal(apep['windspeed2'], 200.))
#     # turn_on = numpyro.sample("turn_on", dists.Normal(apep['turn_on'], 10.))
#     # turn_off = numpyro.sample("turn_off", dists.Normal(apep['turn_off'], 10.))
#     # oblate = numpyro.sample("oblate", dists.Uniform(0., 1.))
#     # orb_sd = numpyro.sample("orb_sd", dists.Exponential(1./10.))
#     # orb_amp = numpyro.sample("orb_amp", dists.Exponential(1./0.1))
#     # orb_min = numpyro.sample("orb_min", dists.Uniform(0., 360.))
#     # az_sd = numpyro.sample("az_sd", dists.Exponential(1./10.))
#     # az_amp = numpyro.sample("az_amp", dists.Exponential(1./0.1))
#     # az_min = numpyro.sample("az_min", dists.Uniform(0., 360.))
#     # comp_incl = numpyro.sample('comp_incl', dists.Normal(apep['comp_incl'], 10))
#     # comp_az = numpyro.sample('comp_az', dists.Normal(apep['comp_az'], 10))
#     # comp_open = numpyro.sample("comp_open", dists.Normal(apep['comp_open'], 10.))
#     # comp_reduction = numpyro.sample("comp_reduction", dists.Uniform(0., 2.))
#     # comp_plume = numpyro.sample("comp_plume", dists.Uniform(0., 2.))
#     # phase = numpyro.sample("phase", dists.Uniform(0., 1.))
#     params['phase'] = numpyro.sample("phase", dists.Uniform(0., 0.99))
#     # sigma = numpyro.sample("sigma", dists.Uniform(0.01, 10.))
#     # histmax = numpyro.sample("histmax", dists.Uniform(0., 1.))
        
#     samp_particles, samp_weights = gm.dust_plume(params)
#     _, _, samp_H = gm.smooth_histogram2d_w_bins(samp_particles, samp_weights, params, xbins, ybins)
#     samp_H = samp_H.flatten()
#     # samp_H = jnp.nan_to_num(samp_H, 1e4)
#     with numpyro.plate('plate', len(obs)):
#         numpyro.sample('obs', dists.Normal(samp_H, E), obs=Y)



# init_params = apep.copy()
# # # init_params_arr = init_params.copy()
# # # for key in init_params.keys():
# # #     init_params_arr[key] = jnp.ones(num_chains) * init_params_arr[key]


# init_params = numpyro.infer.util.constrain_fn(apep_model, (obs, obs_err), {}, init_params)
# # # sampler = numpyro.infer.MCMC(numpyro.infer.NUTS(apep_model, 
# # #                                                 target_accept_prob=0.2,
# # #                                                 init_strategy=numpyro.infer.initialization.init_to_value(values=init_params)),
# # #                               num_chains=1,
# # #                               num_samples=300,
# # #                               num_warmup=20,
# # #                               progress_bar=True)
# # # sampler = numpyro.infer.MCMC(numpyro.infer.HMC(apep_model, 
# # #                                                 target_accept_prob=0.3,
# # #                                                 step_size=2*jnp.pi,
# # #                                                 adapt_step_size=False,
# # #                                                 init_strategy=numpyro.infer.initialization.init_to_value(values=init_params),
# # #                                                 forward_mode_differentiation=True),
# # #                               num_chains=1,
# # #                               num_samples=300,
# # #                               num_warmup=20,
# # #                               progress_bar=True)
# # sampler = numpyro.infer.MCMC(numpyro.infer.HMC(apep_model,
# #                                                init_strategy=numpyro.infer.initialization.init_to_value(values=init_params)),
# #                               num_chains=1,
# #                               num_samples=300,
# #                               num_warmup=20,
# #                               progress_bar=True)
# # # sampler = numpyro.infer.MCMC(numpyro.infer.HMC(apep_model, 
# # #                                                 target_accept_prob=0.3,
# # #                                                 init_strategy=numpyro.infer.initialization.init_to_value(values=init_params),
# # #                                                 forward_mode_differentiation=True,
# # #                                                 step_size=1e-3),
# # #                               num_chains=1,
# # #                               num_samples=300,
# # #                               num_warmup=20,
# # #                               progress_bar=True)
# sampler = numpyro.infer.MCMC(numpyro.infer.NUTS(apep_model),
#                               num_chains=1,
#                               num_samples=1000,
#                               num_warmup=300)
# # sampler = numpyro.infer.MCMC(numpyro.infer.SA(apep_model, init_strategy=numpyro.infer.initialization.init_to_value(values=init_params)),
# #                               num_chains=num_chains,
# #                               num_samples=300,
# #                               num_warmup=100)
# # sampler = numpyro.infer.MCMC(numpyro.infer.SA(apep_model,
# #                                               init_strategy=numpyro.infer.initialization.init_to_value(values={'eccentricity':0.5})),
# #                               num_chains=num_chains,
# #                               num_samples=1000,
# #                               num_warmup=300)
# # sampler = numpyro.infer.MCMC(numpyro.infer.SA(apep_model),
# #                               num_chains=num_chains,
# #                               num_samples=1000,
# #                               num_warmup=300)
# sampler.run(jax.random.PRNGKey(1), obs, obs_err)

# results = sampler.get_samples()
# C = chainconsumer.ChainConsumer()
# C.add_chain(results, name='MCMC Results')
# C.plotter.plot(truth=apep)

# # C.plotter.plot_walks()

# nparams = len(results.keys())
# param_names = list(results.keys())

# fig, axes = plt.subplots(nrows=nparams, sharex=True, gridspec_kw={'hspace':0})

# for i in range(nparams):
#     vals = results[param_names[i]]
#     axes[i].scatter(np.arange(len(vals)), vals, s=0.1)
#     axes[i].set(ylabel=param_names[i])

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

    


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
from glob import glob
from astropy.io import fits

import WR_Geom_Model as gm
import WR_binaries as wrb

apep = wrb.apep.copy()
system_params = apep.copy()
# apep['sigma'] = 0.01

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
max_vals = {}
directory = "Data\\VLT"
fnames = glob(directory + "\\*.fits")

for i, fname in enumerate(fnames):
    
    data = fits.open(fname)[0].data
    
    length = data.shape[0]
    
    X = jnp.linspace(-1., 1., length) * pscale * length/2 / 1000
    Y = X.copy()
    
    xs, ys = jnp.meshgrid(X, Y)
    
    # data[280:320, 280:320] = 0.
    data = np.array(data)
    
    # data = data - jnp.median(data)
    data = data - np.percentile(data, 84)
    data = data / np.max(data)
    data = np.maximum(data, 0)
    data = np.abs(data)**0.5
    # data[280:320, 280:320] = 0.
    vlt_data[vlt_years[i]] = data
    flattened_vlt_data[vlt_years[i]] = data.flatten()
    max_vals[vlt_years[i]] = np.max(data)

big_flattened_data = np.concatenate([flattened_vlt_data[year] for year in vlt_years])
xbins = X
ybins = Y

# ### --- INFERENCE --- ###  
# particles, weights = gm.dust_plume(wrb.apep)
    
# X, Y, H = gm.smooth_histogram2d(particles, weights, wrb.apep)
# xbins = X[0, :]
# ybins = Y[:, 0]
# # X, Y, H = gm.spiral_grid(particles, weights, wrb.apep)
# obs_err = 0.05 * np.max(H)
# H += np.random.normal(0, obs_err, H.shape)
# gm.plot_spiral(X, Y, H)



# obs = H.flatten()
# # obs_err = obs_err * jnp.ones(len(obs))

# fig, ax = plt.subplots()

# ax.plot(jnp.arange(len(obs)), obs, lw=0.5)

obs_err = 0.05
# fig, ax = plt.subplots()

# ax.plot(jnp.arange(len(obs)), obs**3, lw=0.5)


def log_prior(theta):
    [e, phase, open_angle] = theta
    
    lp = 0.
    
    # eccentricity check
    if e > 0.95 or e < 0.:
        return -np.inf
    # orbital phase check
    if phase > 1. or phase < 0.:
        return -np.inf
    # open angle check
    if open_angle > 170. or open_angle < 10.:
        return -np.inf
    # open_angle_unc = 15
    # lp += np.log(1.0 / (np.sqrt(2 * np.pi) * open_angle_unc)) - 0.5 * (open_angle - apep['open_angle'])**2 / (open_angle_unc**2)
    
    return lp
    
def log_likelihood(theta):
    [e, phase, open_angle] = theta
    # print(theta)
    
    data_dict = apep.copy()
    
    data_dict['eccentricity'] = e
    data_dict['phase'] = phase
    data_dict['open_angle'] = open_angle
    
    # particles, weights = gm.dust_plume(data_dict)
    # _, _, model = gm.smooth_histogram2d_w_bins(particles, weights, data_dict, xbins, ybins)
    # model = model.flatten()
    
    # return -0.5 * jnp.sum((obs - model)**2 / obs_err**2)
    chisq = 0. 
    
    # year_model = {}
    # for year in [2024]:
    #     year_params = data_dict.copy()
    #     year_params['phase'] -= (2024 - year) / data_dict['period']
    #     samp_particles, samp_weights = gm.dust_plume(year_params)
    #     _, _, samp_H = smooth_histogram2d_w_bins(samp_particles, samp_weights, year_params, xbins, ybins)
    #     samp_H = gm.add_stars(xbins, ybins, samp_H, year_params)
    #     # samp_H.at[280:320, 280:320].set(0.)
    #     samp_H = samp_H.flatten()
    #     # samp_H = jnp.nan_to_num(samp_H, 1e4)
    #     # year_model[year] = samp_H
        
    #     chisq += np.sum(((flattened_vlt_data[year] - samp_H) / obs_err)**2)
    
    # return -0.5 * chisq
    
    # year_model = {}
    for year in vlt_years:
        year_params = data_dict.copy()
        year_params['phase'] -= (2024 - year) / data_dict['period']
        samp_particles, samp_weights = gm.dust_plume(year_params)
        _, _, samp_H = smooth_histogram2d_w_bins(samp_particles, samp_weights, year_params, xbins, ybins)
        samp_H = gm.add_stars(xbins, ybins, samp_H, year_params)
        # samp_H.at[280:320, 280:320].set(0.)
        samp_H = samp_H.flatten()
        # samp_H = jnp.nan_to_num(samp_H, 1e4)
        # year_model[year] = samp_H
        
        chisq += np.sum((flattened_vlt_data[year] - samp_H)**2 / obs_err**2) / max_vals[year]
    
    return -0.5 * chisq
    
    

def log_prob(theta):
    lp = log_prior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


nwalkers = 10

params = ['eccentricity', 'phase', 'open_angle']


params_jitter = {'eccentricity':0.01, 'phase':0.01, 'open_angle':3.}

pos = np.array([apep[param] for param in params])
ndim = len(pos)
pos = pos * np.ones((nwalkers, ndim))
# pos += 1e-4 * np.random.normal(0, 1, pos.shape)
pos += np.random.normal(np.zeros(ndim), [params_jitter[param] for param in params], pos.shape)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(pos, 500, progress=True);

param_labels = {"eccentricity":r"$e$", 'phase':r'$\phi$', 'open_angle':r"$\theta_{\rm OA}$"}
labels = [param_labels[param] for param in params]

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
    
    
flat_samples = sampler.get_chain(discard=400, flat=True)
import corner
import jax
# labels = ['ecc', 'incl', 'asc_node', 'op_ang']
truths = np.array([apep[param] for param in params])
fig = corner.corner(flat_samples, 
                    labels=labels,
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

    


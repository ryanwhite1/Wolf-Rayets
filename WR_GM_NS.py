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
import jaxns as jns
import matplotlib.pyplot as plt

import WR_Geom_Model as gm
import WR_binaries as wrb

# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

apep = wrb.apep.copy()



### --- INFERENCE --- ###
particles, weights = gm.dust_plume(wrb.apep)
    
X, Y, H = gm.spiral_grid(particles, weights, wrb.apep)
obs_err = 0.01 * np.max(H)
H += np.random.normal(0, obs_err, H.shape)
gm.plot_spiral(X, Y, H)



obs = H.flatten()
obs_err = obs_err * jnp.ones(len(obs))

fig, ax = plt.subplots()

ax.plot(jnp.arange(len(obs)), obs, lw=0.5)


### --- NUMPYRO --- ###
import numpyro, chainconsumer, jax
import numpyro.distributions as dists
from numpyro.contrib.nested_sampling import NestedSampler
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

def log_normal(params):
    samp_particles, samp_weights = gm.dust_plume(params)
    _, _, samp_H = gm.spiral_grid(samp_particles, samp_weights, params)
    samp_H = samp_H.flatten()
    samp_H = jnp.nan_to_num(samp_H, 1e4)
    dx = (obs - samp_H) / obs_err
    return -0.5 * jnp.log(2.*jnp.pi) - jnp.log(obs_err) - 0.5 * dx*dx

def log_likelihood(params):
    return jnp.sum(log_normal(params))

# def prior_model():
#     # m1 = numpyro.sample("m1", dists.Normal(apep['m1'], 5.))
#     # m2 = numpyro.sample("m2", dists.Normal(apep['m2'], 5.))
#     eccentricity = numpyro.sample("eccentricity", dists.Normal(apep['eccentricity'], 0.05))
#     inclination = numpyro.sample("inclination", dists.Normal(apep['inclination'], 20.))
#     asc_node = numpyro.sample("asc_node", dists.Normal(apep['asc_node'], 20.))
#     arg_peri = numpyro.sample("arg_peri", dists.Normal(apep['arg_peri'], 20.))
#     open_angle = numpyro.sample("open_angle", dists.Normal(apep['open_angle'], 10.))
#     period = numpyro.sample("period", dists.Normal(apep['period'], 40.))
#     distance = numpyro.sample("distance", dists.Normal(apep['distance'], 500.))
#     windspeed1 = numpyro.sample("windspeed1", dists.Normal(apep['windspeed1'], 200.))
#     # windspeed2 = numpyro.sample("windspeed2", dists.Normal(apep['windspeed2'], 200.))
#     turn_on = numpyro.sample("turn_on", dists.Normal(apep['turn_on'], 10.))
#     turn_off = numpyro.sample("turn_off", dists.Normal(apep['turn_off'], 10.))
#     # oblate = numpyro.sample("oblate", dists.Uniform(0., 1.))
#     orb_sd = numpyro.sample("orb_sd", dists.Exponential(1./10.))
#     orb_amp = numpyro.sample("orb_amp", dists.Exponential(1./0.1))
#     # orb_min = numpyro.sample("orb_min", dists.Uniform(0., 360.))
#     az_sd = numpyro.sample("az_sd", dists.Exponential(1./10.))
#     az_amp = numpyro.sample("az_amp", dists.Exponential(1./0.1))
#     # az_min = numpyro.sample("az_min", dists.Uniform(0., 360.))
#     # comp_incl = numpyro.sample('comp_incl', dists.Normal(apep['comp_incl'], 10))
#     # comp_az = numpyro.sample('comp_az', dists.Normal(apep['comp_az'], 10))
#     # comp_open = numpyro.sample("comp_open", dists.Normal(apep['comp_open'], 10.))
#     # comp_reduction = numpyro.sample("comp_reduction", dists.Uniform(0., 2.))
#     # comp_plume = numpyro.sample("comp_plume", dists.Uniform(0., 2.))
#     # phase = numpyro.sample("phase", dists.Uniform(0., 1.))
#     # sigma = numpyro.sample("sigma", dists.Uniform(0.01, 10.))
#     # histmax = numpyro.sample("histmax", dists.Uniform(0., 1.))
#     m1 = apep['m1']
#     m2 = apep['m2']
#     # open_angle = apep['open_angle']
#     # period = apep['period']
#     # distance = apep['distance']
#     # windspeed1 = apep['windspeed1']
#     windspeed2 = apep['windspeed2']
#     # turn_on = apep['turn_on']
#     # turn_off = apep['turn_off']
#     oblate = apep['oblate']
#     # orb_sd = apep['orb_sd']
#     # orb_amp = apep['orb_amp']
#     orb_min = apep['orb_min']
#     # az_sd = apep['az_sd']
#     # az_amp = apep['az_amp']
#     az_min = apep['az_min']
#     comp_incl = apep['comp_incl']
#     comp_az = apep['comp_az']
#     comp_open = apep['comp_open']
#     comp_reduction = apep["comp_reduction"]
#     comp_plume = apep["comp_plume"]
#     phase = apep['phase']
#     sigma = apep['sigma']
#     histmax = apep['histmax']
#     params = {"m1":m1, "m2":m2,                # solar masses
#             "eccentricity":eccentricity, 
#             "inclination":inclination, "asc_node":asc_node, "arg_peri":arg_peri,           # degrees
#             "open_angle":open_angle,       # degrees (full opening angle)
#             "period":period, "distance":distance,        # pc
#             "windspeed1":windspeed1, "windspeed2":windspeed2,      # km/s
#             "turn_on":turn_on, "turn_off":turn_off,     # true anomaly (degrees)
#             "oblate":oblate,
#             "orb_sd":orb_sd, "orb_amp":orb_amp, "orb_min":orb_min, 
#             "az_sd":az_sd, "az_amp":az_amp, "az_min":az_min,
#             "comp_incl":comp_incl, "comp_az":comp_az, "comp_open":comp_open, "comp_reduction":comp_reduction, "comp_plume":comp_plume,
#             "phase":phase, "sigma":sigma, "histmax":histmax}
#     return params
    
def prior_model():
    # m1 = numpyro.sample("m1", dists.Normal(apep['m1'], 5.))
    # m2 = numpyro.sample("m2", dists.Normal(apep['m2'], 5.))
    # eccentricity = numpyro.sample("eccentricity", dists.Normal(apep['eccentricity'], 0.05))
    eccentricity = yield jns.Prior(tfpd.Normal(apep['eccentricity'], 0.05), name='eccentricity')
    # inclination = numpyro.sample("inclination", dists.Normal(apep['inclination'], 20.))
    inclination = yield jns.Prior(tfpd.Normal(apep['inclination'], 20.), name='inclination')
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
    # phase = numpyro.sample("phase", dists.Uniform(0., 1.))
    # sigma = numpyro.sample("sigma", dists.Uniform(0.01, 10.))
    # histmax = numpyro.sample("histmax", dists.Uniform(0., 1.))
    m1 = apep['m1']
    m2 = apep['m2']
    asc_node = apep['asc_node']
    arg_peri = apep['arg_peri']
    open_angle = apep['open_angle']
    period = apep['period']
    distance = apep['distance']
    windspeed1 = apep['windspeed1']
    windspeed2 = apep['windspeed2']
    turn_on = apep['turn_on']
    turn_off = apep['turn_off']
    oblate = apep['oblate']
    orb_sd = apep['orb_sd']
    orb_amp = apep['orb_amp']
    orb_min = apep['orb_min']
    az_sd = apep['az_sd']
    az_amp = apep['az_amp']
    az_min = apep['az_min']
    comp_incl = apep['comp_incl']
    comp_az = apep['comp_az']
    comp_open = apep['comp_open']
    comp_reduction = apep["comp_reduction"]
    comp_plume = apep["comp_plume"]
    phase = apep['phase']
    sigma = apep['sigma']
    histmax = apep['histmax']
    nuc_dist = apep['nuc_dist']
    opt_thin_dist = apep['opt_thin_dist']
    acc_max = apep['acc_max']
    params = {"m1":m1, "m2":m2,                # solar masses
            "eccentricity":eccentricity, 
            "inclination":inclination, "asc_node":asc_node, "arg_peri":arg_peri,           # degrees
            "open_angle":open_angle,       # degrees (full opening angle)
            "period":period, "distance":distance,        # pc
            "windspeed1":windspeed1, "windspeed2":windspeed2,      # km/s
            "turn_on":turn_on, "turn_off":turn_off,     # true anomaly (degrees)
            "oblate":oblate,
            "nuc_dist":nuc_dist, "opt_thin_dist":opt_thin_dist,           # nucleation and optically thin distance (AU)
            "acc_max":acc_max,
            "orb_sd":orb_sd, "orb_amp":orb_amp, "orb_min":orb_min, 
            "az_sd":az_sd, "az_amp":az_amp, "az_min":az_min,
            "comp_incl":comp_incl, "comp_az":comp_az, "comp_open":comp_open, "comp_reduction":comp_reduction, "comp_plume":comp_plume,
            "phase":phase, "sigma":sigma, "histmax":histmax}
    return params

model = jns.Model(prior_model=prior_model, log_likelihood=log_likelihood)
model.sanity_check(jax.random.PRNGKey(0), S=100)

ns = jns.DefaultNestedSampler(model=model, max_samples=1e3, difficult_model=True, parameter_estimation=True)
termination_reason, state = jit(ns)(jax.random.PRNGKey(12345))
results = ns.to_results(termination_reason=termination_reason, state=state)
ns.plot_cornerplot(results)







# NS = NestedSampler(model=apep_model,
#                    constructor_kwargs={'num_live_points': 10000, 'max_samples': 2000},
#                    termination_kwargs={'live_evidence_frac': 0.01})
# NS.run(jax.random.PRNGKey(1), obs, obs_err*10)

# results = NS.get_samples(jax.random.PRNGKey(1), int(1e3))
# C = chainconsumer.ChainConsumer()
# C.add_chain(results, name='MCMC Results')
# C.plotter.plot(truth=apep)

# maxlike = apep.copy()
# for key in results.keys():
#     maxlike[key] = np.median(results[key])


# samp_particles, samp_weights = gm.dust_plume(maxlike)
# X, Y, samp_H = gm.spiral_grid(samp_particles, samp_weights, maxlike)
# gm.plot_spiral(X, Y, samp_H)
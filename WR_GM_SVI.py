# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 20:57:29 2024

@author: ryanw
"""

import numpyro  
from numpyro.distributions import constraints  
from numpyro.infer import Predictive, SVI, Trace_ELBO  
  
import jax.numpy as jnp  
import jax, jaxopt  
from jax import random  
  
import numpy as np  
import matplotlib.pyplot as plt  

from jax import jit, vmap, grad
import jax.lax as lax
import jax.scipy.stats as stats
import blackjax
import matplotlib.pyplot as plt
import time
import pickle
import numpyro.distributions as dists
from numpyro.infer.util import initialize_model

import WR_Geom_Model as gm
import WR_binaries as wrb

apep = wrb.apep.copy()

### --- INFERENCE --- ###  
particles, weights = gm.dust_plume(wrb.apep)
    
X, Y, H = gm.smooth_histogram2d(particles, weights, wrb.apep)
xbins = X[0, :]
ybins = Y[:, 0]
H = gm.add_stars(xbins, ybins, H, wrb.apep)
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

system_params = apep.copy()


def apep_model(Y, E):
    params = system_params.copy()
    # m1 = numpyro.sample("m1", dists.Normal(apep['m1'], 5.))
    # m2 = numpyro.sample("m2", dists.Normal(apep['m2'], 5.))
    params['eccentricity'] = numpyro.sample("eccentricity", dists.Normal(apep['eccentricity'], 0.05))
    params['inclination'] = numpyro.sample("inclination", dists.Normal(apep['inclination'], 20.))
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

def manual_guide(Y, E):
    e_mean = numpyro.param('e_mean', apep['eccentricity'], constraint=constraints.interval(0, 0.95))
    e_sigma = numpyro.param('e_sigma', 0.02, constraint=constraints.positive)
    inc_mean = numpyro.param('i_mean', apep['inclination'], constraint=constraints.interval(0, 180))
    inc_sigma = numpyro.param('i_sigma', 10, constraint=constraints.positive)
    numpyro.sample('eccentricity', dists.Normal(e_mean, e_sigma))
    numpyro.sample('inclination', dists.Normal(inc_mean, inc_sigma))
        

optimizer_forauto = numpyro.optim.Adam(step_size=1e-3)  
# optimizer_forauto = numpyro.optim.Minimize()  
svi_samples = 30
# guide_linreg_diag = numpyro.infer.autoguide.AutoMultivariateNormal(apep_model)  
guide_linreg_diag = manual_guide
svi_linreg_diag = SVI(apep_model, guide_linreg_diag, optim = optimizer_forauto, loss=Trace_ELBO(num_particles=8))  
result_linreg_diag = svi_linreg_diag.run(random.PRNGKey(1), svi_samples, obs, obs_err) 

fig, ax = plt.subplots()
ax.plot(result_linreg_diag.losses)
ax.set(xlabel='Iteration No.', ylabel='Loss') 


from chainconsumer import ChainConsumer  
svi_pred_manual = Predictive(manual_guide, params = result_linreg_diag.params, num_samples = 20000)(rng_key = jax.random.PRNGKey(1), Y=obs, E=obs_err)  

c = ChainConsumer()  

c.add_chain(svi_pred_manual, name="Manual guide")  
c.plotter.plot(parameters = ['eccentricity', 'inclination'],   
               truth = {'eccentricity':apep['eccentricity'], 'inclination':apep['inclination']}) 



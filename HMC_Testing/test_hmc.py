# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:43:40 2024

@author: ryanw
"""

import os 
import multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
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
import pickle

np.random.seed(10)

n = 100

sigma = 2

times = np.linspace(0, 10, n)
data_vals = 5 * times**2 + 10 * times - 50 + np.random.normal(loc=0, scale=10, size=n)

fig, ax = plt.subplots()
ax.scatter(times, data_vals)

# ### --- NUMPYRO --- ###
import numpyro, chainconsumer, jax
import numpyro.distributions as dists

def model(Y):
    
    a1 = numpyro.sample("a1", dists.Uniform(0, 50))
    p1 = numpyro.sample("p1", dists.Uniform(-2, 5))
    
    a2 = numpyro.sample("a2", dists.Uniform(0, 50))
    
    c = numpyro.sample("c1", dists.Uniform(-500, 500))
    
    sampled_Y = a1 * times**p1 + a2 * times + c
    
    error = numpyro.sample("e1", dists.Uniform(0, 100))
    
    with numpyro.plate('plate', len(Y)):
        numpyro.sample('obs', dists.Normal(sampled_Y, error), obs=Y)
        

num_chains = 20
print("Num Chains = ", num_chains)

rng_key = jax.random.PRNGKey(10)

sampler = numpyro.infer.MCMC(numpyro.infer.NUTS(model,
                                                ),
                              num_chains=num_chains,
                              num_samples=1000,
                              num_warmup=1000,
                              progress_bar=True)
t1 = time.time()
print("Running HMC Now.")
sampler.warmup(rng_key, data_vals, collect_warmup=True)

# sampler.run(rng_key, data_vals)

results = sampler.get_samples(group_by_chain=True)
results_flat = sampler.get_samples()

print("HMC Finished successfully.")



import corner

corner.corner(results)




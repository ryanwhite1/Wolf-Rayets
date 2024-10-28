# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:41:38 2024

@author: ryanw
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, jit

orb_min = 180.
orb_amp = 0.
val_orb_sd = 40

def func(nu):
    transf_nu = (nu - jnp.pi)%(2. * jnp.pi) - jnp.pi
    periastron_dist = ((transf_nu*180./jnp.pi + 180.) - orb_min)
    
    return jnp.heaviside(val_orb_sd - 1, 1.) * (1. - (1. - orb_amp) * jnp.exp(-0.5 * (periastron_dist / val_orb_sd)**2))

grad_func = jit(grad(func))

xs = np.linspace(-jnp.pi, jnp.pi, 1000)
# xs = np.linspace(0., 2 * jnp.pi, 1000)

ys = func(xs)
grads = [grad_func(x) for x in xs]

fig, axes = plt.subplots(nrows=2)

axes[0].plot(xs, ys)
axes[1].plot(xs, grads)

axes[0].set(ylabel='Function Value')
axes[1].set(xlabel='True Anomaly', ylabel='Function Derivative')
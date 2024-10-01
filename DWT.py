# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:11:20 2024

@author: ryanw


https://jax-wavelet-toolbox.readthedocs.io/en/latest/jaxwt.html
"""

import scipy.datasets
import jaxwt as jwt
import jax.numpy as jnp

import matplotlib.pyplot as plt

data = scipy.datasets.face()
face = jnp.transpose(data, [2, 0, 1])
face = face.astype(jnp.float32)
coeffs = jwt.wavedec2(face, "haar", level=6)

fig, ax = plt.subplots()
ax.imshow(data)

fig, ax = plt.subplots()
ax.imshow(coeffs[2][0][0])



import numpy as np
from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import WR_Geom_Model as gm
import WR_binaries as wrb
starcopy = wrb.apep.copy()
starcopy['n_orbits'] = 1



n = 600     # standard
# n = 600     # VISIR
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

particles, weights = gm.dust_plume(starcopy)
X, Y, H_original = smooth_histogram2d(particles, weights, starcopy)
H_original = gm.add_stars(X[0, :], Y[:, 0], H_original, starcopy)


data = H_original
# face = jnp.transpose(data, [1, 0])
face = data
face = face.astype(jnp.float32)
coeffs = jwt.wavedec2(face, "haar", level=3)

fig, ax = plt.subplots()
ax.imshow(data)
ax.invert_yaxis()

fig, ax = plt.subplots()
ax.imshow(coeffs[-1][0][0])
ax.invert_yaxis()
fig, ax = plt.subplots()
ax.imshow(coeffs[-3][1][0])
ax.invert_yaxis()

fig, ax = plt.subplots()
ax.imshow(coeffs[-3][0][0] + coeffs[-3][1][0])
ax.invert_yaxis()

fig, ax = plt.subplots()
ax.imshow(coeffs[0][0])
ax.invert_yaxis()

print(coeffs[-3][0][0])



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

# data = scipy.datasets.face()
# face = jnp.transpose(data, [2, 0, 1])
# face = face.astype(jnp.float32)
# coeffs = jwt.wavedec2(face, "haar", level=6)

# fig, ax = plt.subplots()
# ax.imshow(data)

# fig, ax = plt.subplots()
# ax.imshow(coeffs[2][0][0])



import numpy as np
from jax import jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
import WR_Geom_Model as gm
import WR_binaries as wrb
starcopy = wrb.apep.copy()
starcopy['n_orbits'] = 1

starcopy['sigma'] = 5



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

# particles, weights = gm.dust_plume(starcopy)
# X, Y, H_original = smooth_histogram2d(particles, weights, starcopy)
# H_original = gm.add_stars(X[0, :], Y[:, 0], H_original, starcopy)


# data = H_original
# # face = jnp.transpose(data, [1, 0])
# face = data
# face = face.astype(jnp.float32)
# coeffs = jwt.wavedec2(face, "db2", level=3)

# fig, ax = plt.subplots()
# ax.imshow(data)
# ax.invert_yaxis()

# fig, ax = plt.subplots()
# ax.imshow(coeffs[-1][0][0])
# ax.invert_yaxis()

# fig, ax = plt.subplots()
# ax.imshow(coeffs[-1][1][0])
# ax.invert_yaxis()

# fig, ax = plt.subplots()
# ax.imshow(coeffs[-3][1][0])
# ax.invert_yaxis()

# fig, ax = plt.subplots()
# ax.imshow(coeffs[-3][0][0] + coeffs[-3][1][0])
# ax.invert_yaxis()

# fig, ax = plt.subplots()
# ax.imshow(coeffs[0][0])
# ax.invert_yaxis()

# print(coeffs[-3][0][0])



from astropy.io import fits
from glob import glob

def Apep_VISIR_reference(year):
    pscale = 1000 * 23/512 # mas/pixel, (Yinuo's email said 45mas/px, but I think the FOV is 23x23 arcsec for a 512x512 image?)
    
    years = {2016:0, 2017:1, 2018:2, 2024:3}
    directory = "Data\\VLT"
    fnames = glob(directory + "\\*.fits")
    
    vlt_data = fits.open(fnames[years[year]])    # for the 2024 epoch
    
    data = vlt_data[0].data
    length = data.shape[0]
    
    X = jnp.linspace(-1., 1., length) * pscale * length/2 / 1000
    Y = X.copy()
    
    xs, ys = jnp.meshgrid(X, Y)
    
    data = jnp.array(data)
    # data = data - jnp.median(data)
    data = data - jnp.percentile(data, 84)
    data = data/jnp.max(data)
    data = jnp.maximum(data, 0)
    data = jnp.abs(data)**0.5
    
    
    return xs, ys, data

X_ref, Y_ref, data = Apep_VISIR_reference(2016)

face = data
face = face.astype(jnp.float32)
coeffs = jwt.wavedec2(face, "db2", level=3)

fig, ax = plt.subplots()
ax.imshow(data)
ax.invert_yaxis()

fig, ax = plt.subplots()
ax.imshow(coeffs[-1][0][0])
ax.invert_yaxis()

fig, ax = plt.subplots()
ax.imshow(coeffs[-1][1][0])
ax.invert_yaxis()

fig, ax = plt.subplots()
ax.imshow(coeffs[-3][1][0])
ax.invert_yaxis()

fig, ax = plt.subplots()
ax.imshow(coeffs[-3][1][0] * jnp.heaviside(abs(coeffs[-3][1][0]) - 0.04, 1))
ax.invert_yaxis()

fig, ax = plt.subplots()
ax.imshow(coeffs[-3][0][0] + coeffs[-3][1][0])
ax.invert_yaxis()

fig, ax = plt.subplots()
ax.imshow(coeffs[0][0])
ax.invert_yaxis()

print(coeffs[-3][0][0])






















particles, weights = gm.dust_plume(starcopy)
X, Y, H_original = smooth_histogram2d_w_bins(particles, weights, starcopy, X_ref[0, :], Y_ref[:, 0])
# H_original = gm.add_stars(X[0, :], Y[:, 0], H_original, starcopy)


data = H_original
# face = jnp.transpose(data, [1, 0])
face = data
face = face.astype(jnp.float32)
coeffs = jwt.wavedec2(face, "db2", level=3)

fig, ax = plt.subplots()
ax.imshow(data)
ax.invert_yaxis()

fig, ax = plt.subplots()
ax.imshow(coeffs[-1][0][0])
ax.invert_yaxis()

fig, ax = plt.subplots()
ax.imshow(coeffs[-1][1][0])
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


# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:20:48 2024

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

def smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size):
    '''
    '''
    x = particles[0, :]
    y = particles[1, :]
    
    side_width = xedges[1] - xedges[0]
    
    # xpos = jnp.round(x - jnp.min(xedges), 12)
    # ypos = jnp.round(y - jnp.min(yedges), 12)
    
    xpos = x - jnp.min(xedges)
    ypos = y - jnp.min(yedges)
    # xpos = xpos + jnp.where(xpos%side_width == 0., -1e-8, 0)
    # ypos = ypos + jnp.where(ypos%side_width == 0., -1e-8, 0)
    
    x_indices = jnp.floor(xpos / side_width).astype(int)
    y_indices = jnp.floor(ypos / side_width).astype(int)
    
    alphas = xpos%side_width
    betas = ypos%side_width
    
    # alphas = jnp.where(jnp.isclose(alphas / side_width, 1., atol=1e-4), 0., alphas)
    # betas = jnp.where(jnp.isclose(betas / side_width, 1., atol=1e-4), 0., betas)
    
    a_s = jnp.minimum(alphas, side_width - alphas) + side_width / 2
    b_s = jnp.minimum(betas, side_width - betas) + side_width / 2
    
    one_minus_a_indices = x_indices + jnp.where(alphas > side_width / 2, 1, -1)
    one_minus_b_indices = y_indices + jnp.where(betas > side_width / 2, 1, -1)
    
    one_minus_a_indices = one_minus_a_indices.astype(int)
    one_minus_b_indices = one_minus_b_indices.astype(int)
    
    # now check the indices that are out of bounds
    x_edge_check = jnp.heaviside(one_minus_a_indices, 1) * jnp.heaviside(im_size - one_minus_a_indices, 0)
    y_edge_check = jnp.heaviside(one_minus_b_indices, 1) * jnp.heaviside(im_size - one_minus_b_indices, 0)
    
    x_edge_check = x_edge_check.astype(int)
    y_edge_check = y_edge_check.astype(int)
    
    main_quadrant = a_s * b_s * weights
    horizontal_quadrant = (side_width - a_s) * b_s * weights * x_edge_check
    vertical_quadrant = a_s * (side_width - b_s) * weights * y_edge_check
    corner_quadrant = (side_width - a_s) * (side_width - b_s) * weights * x_edge_check * y_edge_check

    # The below few lines rely fundamentally on the following line sourced from https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html :
    # Unlike NumPy in-place operations such as x[idx] += y, if multiple indices refer to the same location, all updates will be applied (NumPy would only apply the last update, rather than applying all updates.)
    
    H = jnp.zeros((im_size, im_size))
    H = H.at[x_indices, y_indices].add(main_quadrant)
    H = H.at[one_minus_a_indices, y_indices].add(x_edge_check * horizontal_quadrant)
    H = H.at[x_indices, one_minus_b_indices].add(y_edge_check * vertical_quadrant)
    H = H.at[one_minus_a_indices, one_minus_b_indices].add(x_edge_check * y_edge_check * corner_quadrant)

    X, Y = jnp.meshgrid(xedges, yedges)
    H = H.T
    
    shape = 30 // 2  # choose just large enough grid for our gaussian
    gx, gy = jnp.meshgrid(jnp.arange(-shape, shape+1, 1), jnp.arange(-shape, shape+1, 1))
    gxy = jnp.exp(- (gx*gx + gy*gy) / (2 * stardata['sigma']**2))
    gxy /= gxy.sum()
    
    H = signal.convolve(H, gxy, mode='same', method='fft')
    
    H /= jnp.max(H)
    H = H**stardata['lum_power']
    
    H = jnp.minimum(H, jnp.ones((im_size, im_size)) * stardata['histmax'])
    H /= jnp.max(H)
    
    return X, Y, H
n = 26
def smooth_histogram2d(particles, weights, stardata):
    im_size = n
    
    x = particles[0, :]
    y = particles[1, :]
    
    xbound, ybound = jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))
    bound = jnp.max(jnp.array([xbound, ybound])) * (1. + 2. / im_size)
    
    xedges, yedges = jnp.linspace(-bound, bound, im_size+1), jnp.linspace(-bound, bound, im_size+1)
    return smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
def smooth_histogram2d_w_bins(particles, weights, stardata, xbins, ybins):
    im_size = n
    return smooth_histogram2d_base(particles, weights, stardata, xbins, ybins, im_size)


stardata = wrb.apep
stardata['sigma'] = 0.01

p1 = np.array([1., 1., 1.])
p2 = np.array([2., 3., 2.])
p3 = np.array([0., 0., 0.])
p4 = np.array([-3., -5., 1.])
p5 = np.array([2., -3., 0.])
p6 = np.array([0., -3., 0.])
particles = np.column_stack((p1, p2, p3, p4, p5, p6))

weights = np.ones(particles.shape[1])
weights[0] = 0.5
X, Y, H = smooth_histogram2d(particles, weights, stardata)

for add in [0, 0.5, 1, 1.5]:
    p1 = np.array([1., 1., 1.])
    p2 = np.array([2., 3., 2.])
    p3 = np.array([0., 0., 0.])
    p4 = np.array([-3., -5., 1.])
    p5 = np.array([2., -3., 0.])
    p6 = np.array([0., -3., 0.])
    
    p1[0] += add
    p4[1] += add
    
    particles = np.column_stack((p1, p2, p3, p4, p5, p6))
    
    _, _, H = smooth_histogram2d_w_bins(particles, weights, stardata, X[0, :], Y[:, 0])
    
    ax = gm.plot_spiral(X, Y, H)
    ax.scatter(particles[0, :], particles[1, :])
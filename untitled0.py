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

def smooth_histogram2d(particles, weights, stardata):
    '''
    '''
    
    im_size = 15
    
    x = particles[0, :]
    y = particles[1, :]
    
    xbound, ybound = jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))
    bound = jnp.max(jnp.array([xbound, ybound])) * (1. + 2. / im_size)
    _, xedges, yedges = jnp.histogram2d(x, y, bins=im_size, weights=weights, range=jnp.array([[-bound, bound], [-bound, bound]]))
    
    x_indices = jnp.digitize(x, xedges) - 1
    y_indices = jnp.digitize(y, yedges) - 1
    
    side_width = xedges[1] - xedges[0]
    
    alphas = x%side_width
    betas = y%side_width
    
    a_s = jnp.minimum(alphas, side_width - alphas) + side_width / 2
    b_s = jnp.minimum(betas, side_width - betas) + side_width / 2 
    # one_minus_a_indices = x_indices - 1 + 2 * jnp.heaviside(alphas - side_width / 2, 0)
    # one_minus_b_indices = y_indices - 1 + 2 * jnp.heaviside(betas - side_width / 2, 0)
    
    one_minus_a_indices = x_indices - 1 + 2 * jnp.heaviside(side_width / 2 - alphas, 0)
    one_minus_b_indices = y_indices - 1 + 2 * jnp.heaviside(side_width / 2 - betas, 0)
    
    # one_minus_a_indices = x_indices + 1 - 2 * jnp.heaviside(alphas - side_width / 2, 0)
    # one_minus_b_indices = y_indices + 1 - 2 * jnp.heaviside(betas - side_width / 2, 0)
    
    one_minus_a_indices = one_minus_a_indices.astype(int)
    one_minus_b_indices = one_minus_b_indices.astype(int)
    
    # now check the indices that are out of bounds
    x_edge_check = jnp.heaviside(one_minus_a_indices, 1) * jnp.heaviside(im_size - one_minus_a_indices, 0)
    y_edge_check = jnp.heaviside(one_minus_b_indices, 1) * jnp.heaviside(im_size - one_minus_b_indices, 0)
    
    x_edge_check = x_edge_check.astype(int)
    y_edge_check = y_edge_check.astype(int)
    
    main_quadrant = a_s * b_s * weights
    horizontal_quadrant = (side_width - a_s) * b_s * weights
    vertical_quadrant = a_s * (side_width - b_s) * weights
    corner_quadrant = (side_width - a_s) * (side_width - b_s) * weights

    # The below few lines rely fundamentally on the following line sourced from https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html :
    # Unlike NumPy in-place operations such as x[idx] += y, if multiple indices refer to the same location, all updates will be applied (NumPy would only apply the last update, rather than applying all updates.)
    
    H = jnp.zeros((im_size, im_size))
    
    H = H.at[x_indices, y_indices].add(main_quadrant)
    H = H.at[one_minus_a_indices, y_indices].add(x_edge_check * horizontal_quadrant)
    H = H.at[x_indices, one_minus_b_indices].add(y_edge_check * vertical_quadrant)
    H = H.at[one_minus_a_indices, one_minus_b_indices].add(x_edge_check * y_edge_check * corner_quadrant)
    
    X, Y = jnp.meshgrid(xedges, yedges)
    # H = H.T
    H /= jnp.max(H)
    
    H = jnp.minimum(H, jnp.ones((im_size, im_size)) * stardata['histmax'])
    
    shape = 30 // 2  # choose just large enough grid for our gaussian
    gx, gy = jnp.meshgrid(jnp.arange(-shape, shape+1, 1), jnp.arange(-shape, shape+1, 1))
    gxy = jnp.exp(- (gx*gx + gy*gy) / (2 * stardata['sigma']**2))
    gxy /= gxy.sum()
    
    H = signal.convolve(H, gxy, mode='same', method='fft')
    
    H /= jnp.max(H)
    H = H**stardata['lum_power']
    
    return X, Y, H
def smooth_histogram2d_w_bins(particles, weights, stardata, xbins, ybins):
    ''' Takes in the particle positions and weights and calculates the 2D histogram, ignoring those points at (0,0,0), and
        applying a Gaussian blur.
    Parameters
    ----------
    particles : ndarray (Ndim, Nparticles)
        Particle positions in cartesian coordinates
    weights : array (Nparticles)
        Weight of each particle in the histogram (for orbital/azimuthal variations)
    sigma : 
    '''
    im_size = 15
    
    x = particles[0, :]
    y = particles[1, :]
    
    _, xedges, yedges = jnp.histogram2d(x, y, bins=[xbins, ybins], weights=weights)
    
    x_indices = jnp.digitize(x, xedges) - 1
    y_indices = jnp.digitize(y, yedges) - 1
    
    side_width = xedges[1] - xedges[0]
    
    alphas = x%side_width
    betas = y%side_width
    
    a_s = jnp.minimum(alphas, side_width - alphas) + side_width / 2
    b_s = jnp.minimum(betas, side_width - betas) + side_width / 2 
    # one_minus_a_indices = x_indices - 1 + 2 * jnp.heaviside(alphas - side_width / 2, 0)
    # one_minus_b_indices = y_indices - 1 + 2 * jnp.heaviside(betas - side_width / 2, 0)
    
    one_minus_a_indices = x_indices - 1 + 2 * jnp.heaviside(side_width / 2 - alphas, 0)
    one_minus_b_indices = y_indices - 1 + 2 * jnp.heaviside(side_width / 2 - betas, 0)
    
    # one_minus_a_indices = x_indices + 1 - 2 * jnp.heaviside(alphas - side_width / 2, 0)
    # one_minus_b_indices = y_indices + 1 - 2 * jnp.heaviside(betas - side_width / 2, 0)
    
    one_minus_a_indices = one_minus_a_indices.astype(int)
    one_minus_b_indices = one_minus_b_indices.astype(int)
    
    # now check the indices that are out of bounds
    x_edge_check = jnp.heaviside(one_minus_a_indices, 1) * jnp.heaviside(im_size - one_minus_a_indices, 0)
    y_edge_check = jnp.heaviside(one_minus_b_indices, 1) * jnp.heaviside(im_size - one_minus_b_indices, 0)
    
    x_edge_check = x_edge_check.astype(int)
    y_edge_check = y_edge_check.astype(int)
    
    main_quadrant = a_s * b_s * weights
    horizontal_quadrant = (side_width - a_s) * b_s * weights
    vertical_quadrant = a_s * (side_width - b_s) * weights
    corner_quadrant = (side_width - a_s) * (side_width - b_s) * weights
    
    # The below few lines rely fundamentally on the following line sourced from https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html :
    # Unlike NumPy in-place operations such as x[idx] += y, if multiple indices refer to the same location, all updates will be applied (NumPy would only apply the last update, rather than applying all updates.)
    
    H = jnp.zeros((im_size, im_size))
    
    H = H.at[x_indices, y_indices].add(main_quadrant)
    H = H.at[one_minus_a_indices, y_indices].add(x_edge_check * horizontal_quadrant)
    H = H.at[x_indices, one_minus_b_indices].add(y_edge_check * vertical_quadrant)
    H = H.at[one_minus_a_indices, one_minus_b_indices].add(x_edge_check * y_edge_check * corner_quadrant)
    
    X, Y = jnp.meshgrid(xedges, yedges)
    H = H.T
    H /= jnp.max(H)
    
    H = jnp.minimum(H, jnp.ones((im_size, im_size)) * stardata['histmax'])
    
    shape = 30 // 2  # choose just large enough grid for our gaussian
    gx, gy = jnp.meshgrid(jnp.arange(-shape, shape+1, 1), jnp.arange(-shape, shape+1, 1))
    gxy = jnp.exp(- (gx*gx + gy*gy) / (2 * stardata['sigma']**2))
    gxy /= gxy.sum()
    
    H = signal.convolve(H, gxy, mode='same', method='fft')
    
    H /= jnp.max(H)
    H = H**stardata['lum_power']
    
    return X, Y, H


stardata = wrb.apep
stardata['sigma'] = 0.01

p1 = np.array([1., 1., 1.])
p2 = np.array([2., 3., 2.])
p3 = np.array([0., 0., 0.])
p4 = np.array([-3., -5., 1.])
p5 = np.array([2., -3., 0.])
particles = np.column_stack((p1, p2, p3, p4, p5))

weights = np.ones(particles.shape[1])
# weights[0] = 0.5
X, Y, H = smooth_histogram2d(particles, weights, stardata)

for add in [0, 0.5, 1, 1.5]:
    p1 = np.array([1., 1., 1.])
    p2 = np.array([2., 3., 2.])
    p3 = np.array([0., 0., 0.])
    p4 = np.array([-3., -5., 1.])
    p5 = np.array([2., -3., 0.])
    
    p1[0] += add
    p4[1] += add
    
    particles = np.column_stack((p1, p2, p3, p4, p5))
    
    _, _, H = smooth_histogram2d_w_bins(particles, weights, stardata, X[0, :], Y[:, 0])
    
    ax = gm.plot_spiral(X, Y, H)
    ax.scatter(particles[0, :], particles[1, :])
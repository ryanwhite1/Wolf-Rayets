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
    # print(x)
    
    side_width = xedges[1] - xedges[0]

    xpos = np.round(x - np.min(xedges), 10)
    ypos = np.round(y - np.min(yedges), 10)
    
    xpos = xpos + np.where(xpos%side_width == 0., 0.000001, 0.)
    ypos = ypos + np.where(ypos%side_width == 0., 0.000001, 0.)
    
    
    # print(xpos)
    # print(np.fmod(xpos, 1))

    x_indices = np.floor(xpos / side_width).astype(int)
    y_indices = np.floor(ypos / side_width).astype(int)
    
    # x_indices = xpos.astype(int)
    # y_indices = ypos.astype(int)
    
    # x_indices = jnp.where(xpos%1. == 0, jnp.ceil(xpos), jnp.floor(xpos)).astype(int)
    # y_indices = jnp.where(ypos%1. == 0, jnp.ceil(ypos), jnp.floor(ypos)).astype(int)
    # print(xpos)
    # print(jnp.mod(xpos, 1))
    # x_indices = x_indices - im_size%2
    # y_indices = y_indices - im_size%2
    
    alphas = xpos%side_width
    betas = ypos%side_width
    
    
    
    # alphas = (1 - jnp.heaviside(im_size%2, 0)) * (-x%side_width) + jnp.heaviside(im_size%2, 0) * (x%side_width)
    # betas = (1 - jnp.heaviside(im_size%2, 0)) * (-y%side_width) + jnp.heaviside(im_size%2, 0) * (y%side_width)
    
    a_s = np.minimum(alphas, side_width - alphas) + side_width / 2
    b_s = np.minimum(betas, side_width - betas) + side_width / 2
    # print(alphas / side_width)
    # print(a_s / side_width)
    
    # a_s = np.minimum(alphas, side_width - alphas) + side_width / 2 + np.heaviside(-alphas, side_width / 2)
    # b_s = np.minimum(betas, side_width - betas) + side_width / 2 + np.heaviside(-betas, side_width / 2)
    
    # one_minus_a_indices = x_indices - 1 + 2 * np.heaviside(alphas - side_width / 2, 1)
    # one_minus_b_indices = y_indices - 1 + 2 * np.heaviside(betas - side_width / 2, 1)
    
    one_minus_a_indices = x_indices + np.where(alphas > side_width / 2, 1, -1)
    one_minus_b_indices = y_indices + np.where(betas > side_width / 2, 1, -1)
    
    
    # one_minus_a_indices = x_indices - 1 + 2 * jnp.heaviside(a_s - side_width / 2, 1)
    # one_minus_b_indices = y_indices - 1 + 2 * jnp.heaviside(b_s - side_width / 2, 1)
    # one_minus_a_indices = x_indices + (- 1 + 2 * jnp.heaviside(alphas - side_width / 2, 0)) * (1 - 2 * (im_size%2))
    # one_minus_b_indices = y_indices + (- 1 + 2 * jnp.heaviside(betas - side_width / 2, 0)) * (1 - 2 * (im_size%2))
    
    # one_minus_a_indices = x_indices - 1 + 2 * jnp.heaviside(side_width / 2 - alphas, 0)
    # one_minus_b_indices = y_indices - 1 + 2 * jnp.heaviside(side_width / 2 - betas, 0)
    
    # one_minus_a_indices = x_indices + 1 - 2 * jnp.heaviside(alphas - side_width / 2, 0)
    # one_minus_b_indices = y_indices + 1 - 2 * jnp.heaviside(betas - side_width / 2, 0)
    
    
    one_minus_a_indices = one_minus_a_indices.astype(int)
    one_minus_b_indices = one_minus_b_indices.astype(int)
    
    # now check the indices that are out of bounds
    x_edge_check = np.heaviside(one_minus_a_indices, 1) * np.heaviside(im_size - one_minus_a_indices, 0)
    y_edge_check = np.heaviside(one_minus_b_indices, 1) * np.heaviside(im_size - one_minus_b_indices, 0)
    
    x_edge_check = x_edge_check.astype(int)
    y_edge_check = y_edge_check.astype(int)
    
    main_quadrant = a_s * b_s * weights
    horizontal_quadrant = (side_width - a_s) * b_s * weights * x_edge_check
    vertical_quadrant = a_s * (side_width - b_s) * weights * y_edge_check
    corner_quadrant = (side_width - a_s) * (side_width - b_s) * weights * x_edge_check * y_edge_check
    
    # horizontal_quadrant_new = im_size%2 * vertical_quadrant + (1 - im_size%2) * horizontal_quadrant
    # vertical_quadrant_new = im_size%2 * horizontal_quadrant + (1 - im_size%2) * vertical_quadrant
    # horizontal_quadrant = horizontal_quadrant_new
    # vertical_quadrant = vertical_quadrant_new

    # The below few lines rely fundamentally on the following line sourced from https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html :
    # Unlike NumPy in-place operations such as x[idx] += y, if multiple indices refer to the same location, all updates will be applied (NumPy would only apply the last update, rather than applying all updates.)
    
    H = np.zeros((im_size, im_size))
    
    # need to add the horizontal quadrant actually on the vertical one (and vice versa) because of the transposing later on
    # H = H.at[x_indices, y_indices].add(main_quadrant)
    # H = H.at[one_minus_a_indices, y_indices].add(y_edge_check * vertical_quadrant)
    # H = H.at[x_indices, one_minus_b_indices].add(x_edge_check * horizontal_quadrant)
    # H = H.at[one_minus_a_indices, one_minus_b_indices].add(x_edge_check * y_edge_check * corner_quadrant)
    for i in range(len(x_indices)):
        H[x_indices[i], y_indices[i]] += main_quadrant[i]
        H[one_minus_a_indices[i], y_indices[i]] += horizontal_quadrant[i]
        H[x_indices[i], one_minus_b_indices[i]] += vertical_quadrant[i]
        H[one_minus_a_indices[i], one_minus_b_indices[i]] += corner_quadrant[i]
    
    # for i in range(len(x_indices)):
    #     H[x_indices[i], y_indices[i]] += main_quadrant[i]
    #     H[one_minus_a_indices[i], y_indices[i]] += vertical_quadrant[i]
    #     H[x_indices[i], one_minus_b_indices[i]] += horizontal_quadrant[i]
    #     H[one_minus_a_indices[i], one_minus_b_indices[i]] += corner_quadrant[i]


    
    X, Y = np.meshgrid(xedges, yedges)
    H = H.T
    
    shape = 30 // 2  # choose just large enough grid for our gaussian
    gx, gy = np.meshgrid(np.arange(-shape, shape+1, 1), np.arange(-shape, shape+1, 1))
    gxy = np.exp(- (gx*gx + gy*gy) / (2 * stardata['sigma']**2))
    gxy /= gxy.sum()
    
    H = signal.convolve(H, gxy, mode='same', method='fft')
    
    H /= np.max(H)
    H = H**stardata['lum_power']
    
    H = np.minimum(H, jnp.ones((im_size, im_size)) * stardata['histmax'])
    H /= np.max(H)
    
    return X, Y, H
n = 20
def smooth_histogram2d(particles, weights, stardata):
    im_size = n
    
    x = particles[0, :]
    y = particles[1, :]
    
    xbound, ybound = np.max(np.abs(x)), np.max(np.abs(y))
    bound = np.max(np.array([xbound, ybound])) * (1. + 2. / im_size)
    
    xedges, yedges = np.linspace(-bound, bound, im_size+1), np.linspace(-bound, bound, im_size+1)
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
# weights[0] = 0.5
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
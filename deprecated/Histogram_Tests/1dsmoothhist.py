# -*- coding: utf-8 -*-
"""
Created on Fri May 31 21:46:37 2024

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt


def smooth_histogram1d_base(particles, weights, xedges, im_size):
    
    x = particles[:]
    dx = xedges[1] - xedges[0]
    
    H = np.zeros(im_size)
    print(xedges / dx)
    for i in range(len(x)):
        # print((x[i] - np.min(xedges)) / dx)
        use_x = (x[i] - np.min(xedges)) / dx
        use_x = round(use_x, 13)
        # if use_x%1. == 0.:
        #     use_x += 0.000000001
        j = np.floor(use_x).astype(int)
        
        
        alpha = (x[i] - np.min(xedges))%dx
        a_s = np.min([alpha, dx - alpha]) + dx/2
        print(use_x, j, alpha / dx)
        
        if alpha > dx / 2:
            border_ind = j + 1
        else:
            border_ind = j - 1
        border_ind = border_ind.astype(int) 
        print(alpha/dx, border_ind - j)
        # border_ind = border_ind
        H[j] += a_s
        if border_ind < im_size and border_ind >= 0:
            H[border_ind] += dx - a_s
    H /= dx
    return xedges, H
n = 11
def smooth_histogram1d(particles, weights):
    im_size = n
    
    x = particles[:]
    
    xbound = np.max(np.abs(x))
    bound = xbound * (1. + 2. / im_size)
    
    xedges = np.linspace(-bound, bound, im_size+1)
    return smooth_histogram1d_base(particles, weights, xedges, im_size)
def smooth_histogram2d_w_bins(particles, weights, stardata, xbins):
    im_size = n
    return smooth_histogram1d_base(particles, weights, xbins, im_size)

particles = np.array([0., 1., 4., 10., 8.5, -6.3])
weights = np.ones(len(particles))

X, H = smooth_histogram1d(particles, weights)

fig, ax = plt.subplots()
ax.hist(X[:-1], X, weights=H, edgecolor='k')
for x in particles:
    ax.axvline(x, c='tab:purple', ls='--')
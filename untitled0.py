# -*- coding: utf-8 -*-
"""
Created on Fri May 31 21:46:37 2024

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt


def smooth_histogram1d_base(particles, weights, xedges, im_size):
    
    x = particles[:, 0]
    dx = xedges[1] - xedges[0]
    
    H = np.ones(len(x))
    
    for i in range(len(x)):
        j = np.floor(x[i] / dx).astype(int)
        H[j] += 1
    
    return xedges, H
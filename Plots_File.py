# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 07:57:21 2024

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

def apep_plot():
    star = wrb.apep
    
    particles, weights = gm.dust_plume(star)
    X, Y, H = gm.smooth_histogram2d(particles, weights, star)
    H = gm.add_stars(X[0, :], Y[:, 0], H, star)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')
    
    fig.savefig('Images/Apep_Plot.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Apep_Plot.pdf', dpi=400, bbox_inches='tight')

def apep_cone_plot():
    def turning_point(data):
        ''' Finds the indices of the turning points when there are exactly two turning points in a 1d array. '''
        indices = np.zeros(2)
        deriv = np.diff(data)
        sign = np.sign(data[0])
        j = 0
        for i in range(len(deriv)):
            if np.sign(deriv[i]) != sign:
                indices[j] = i 
                j += 1 
                sign = np.sign(deriv[i])
        return indices.astype(int)
            
    star = wrb.apep
    
    particles, weights = gm.dust_plume(star)
    X, Y, H = gm.smooth_histogram2d(particles, weights, star)
    H = gm.add_stars(X[0, :], Y[:, 0], H, star)

    # now display a circle around the cavity from the ternary star
    u = np.linspace(0, 2 * np.pi, 100)
    open_ang = np.deg2rad(star['comp_open']) / 2
    incl = np.deg2rad(star['comp_incl'])
    az = np.deg2rad(star['comp_az'])
    # formula from https://stackoverflow.com/questions/42068073/python-plotting-points-and-circles-on-a-sphere
    x = np.sin(open_ang) * np.cos(incl) * np.cos(az) * np.cos(u) + np.cos(open_ang) * np.sin(incl) * np.cos(az) - np.sin(open_ang) * np.sin(az) * np.sin(u)
    y = np.sin(open_ang) * np.cos(incl) * np.sin(az) * np.cos(u) + np.cos(open_ang) * np.sin(incl) * np.sin(az) + np.sin(open_ang) * np.cos(az) * np.sin(u)
    z = -np.sin(open_ang) * np.sin(incl) * np.cos(u) + np.cos(open_ang) * np.cos(incl)

    cone_circ = np.array([x, y, z])
    
    # get the distance to the edge (bottom) of the cone
    distance = star['windspeed1'] * star['period'] * star['phase'] * gm.yr2s
    cone_circ *= distance
    
    cone_circ, _ = gm.transform_orbits(cone_circ, np.zeros(cone_circ.shape), star)
    
    turn_x = turning_point(cone_circ[0, :])     # get the turning point indices in each of the x and y directions
    turn_y = turning_point(cone_circ[1, :])
    
    y_turn_1 = cone_circ[1, turn_y[0]]  # y-values of each turning point for the y array
    y_turn_2 = cone_circ[1, turn_y[1]] 
    
    
    point_1, point_2 = np.zeros(2), np.zeros(2)
    arg_min = np.argmin([cone_circ[0, turn_x[0]], cone_circ[0, turn_x[1]]])
    other_arg = int(not arg_min)
    point_1[0] = cone_circ[0, turn_x[arg_min]]
    point_1[1] = cone_circ[1, turn_x[arg_min]]
    point_2[0] = cone_circ[0, turn_x[other_arg]]
    point_2[1] = cone_circ[1, turn_x[other_arg]]
        
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')
    ax.plot(cone_circ[0, :], cone_circ[1, :], c='w', rasterized=True)
    ax.plot([0, np.mean(cone_circ[0, :])], [0, np.mean(cone_circ[1, :])], ls='--', c='w', rasterized=True)
    ax.plot([0, point_1[0]], [0, point_1[1]], c='w', rasterized=True)
    ax.plot([0, point_2[0]], [0, point_2[1]], c='w', rasterized=True)
    
    fig.savefig('Images/Apep_Cone.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Apep_Cone.pdf', dpi=400, bbox_inches='tight')
    
    
def main():
    apep_plot()
    apep_cone_plot()
    



if __name__ == "__main__":
    main()
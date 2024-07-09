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

def apep_plot(filename, custom_params={}):
    star = wrb.apep.copy()
    
    for param in custom_params:
        star[param] = custom_params[param]
    
    particles, weights = gm.dust_plume(star)
    X, Y, H = gm.smooth_histogram2d(particles, weights, star)
    H = gm.add_stars(X[0, :], Y[:, 0], H, star)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')
    
    fig.savefig(f'Images/{filename}.png', dpi=400, bbox_inches='tight')
    fig.savefig(f'Images/{filename}.pdf', dpi=400, bbox_inches='tight')

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
            
    star = wrb.apep.copy()
    
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
    

def smooth_hist_demo():
    im_size = 16
    
    x = np.array([-1.1, 0, 0.5, 0.54, -0.536, -0.6])
    y = np.array([0, 0, 0.67, -0.698, -0.6, 0.7])
    
    particles = np.array([x, y])
    weights = np.ones(len(x))
    
    xbound, ybound = jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))
    bound = jnp.max(jnp.array([xbound, ybound])) * (1. + 2. / im_size)
    
    stardata = wrb.test_system.copy()
    stardata['sigma'] = 0.1
    
    xedges, yedges = jnp.linspace(-bound, bound, im_size+1), jnp.linspace(-bound, bound, im_size+1)
    X, Y, H = gm.smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
    
    fig, ax = plt.subplots()
    ax.set_facecolor('k')
    ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    ax.scatter(x, y, rasterized=True)
    for i in range(len(xedges)):
        ax.axhline(xedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
        ax.axvline(yedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
    ax.set(aspect='equal', xlabel=r'$x$', ylabel=r'$y$')
    
    fig.savefig('Images/Smooth_Hist_Demo.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Smooth_Hist_Demo.pdf', dpi=400, bbox_inches='tight')
    
def smooth_hist_gif():
    im_size = 10
    
    x = np.array([-0.8])
    y = np.array([0])
    
    particles = np.array([x, y])
    weights = np.ones(len(x))
    
    xbound = ybound = 1
    bound = jnp.max(jnp.array([xbound, ybound])) * (1. + 2. / im_size)
    
    stardata = wrb.test_system.copy()
    stardata['sigma'] = 0.1
    
    xedges, yedges = jnp.linspace(-bound, bound, im_size+1), jnp.linspace(-bound, bound, im_size+1)
    X, Y, H = gm.smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
    
    fig, ax = plt.subplots()
    ax.set_facecolor('k')
    mesh = ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    scatter = ax.scatter(x, y, rasterized=True)
    ax.set(aspect='equal', xlabel=r'$x$', ylabel=r'$y$')
    for i in range(len(xedges)):
        ax.axhline(xedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
        ax.axvline(yedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
    
    
    every = 1
    length = 10
    # now calculate some parameters for the animation frames and timing
    # nt = int(stardata['period'])    # roughly one year per frame
    nt = 100
    # nt = 10
    frames = jnp.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
    fps = len(frames) // length  # fps for the final animation
    
    xs = np.linspace(x[0], 1, nt)
    ys = xs**2
    
    Hs = []
    for i in range(nt):
        current_xs = np.array([xs[i], 0])
        current_ys = np.array([ys[i], 0])
        particles = np.array([current_xs, current_ys])
        weights = np.array([1, 0])
        X, Y, H = gm.smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
        Hs.append(H)
    
    def animate(i):
        if i%(nt // 10) == 0:
            print(i/nt * 100, "%", sep='')
        # current_xs = np.array([xs[i], 0])
        # current_ys = np.array([ys[i], 0])
        # particles = np.array([current_xs, current_ys])
        # weights = np.array([1, 0])
        # X, Y, H = gm.smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
        # ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
        # ax.scatter(xs[i], ys[i], rasterized=True)
        
        mesh.set_array(Hs[i])
        scatter.set_offsets(np.c_[xs[i], ys[i]])
        return fig, 

    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
    ani.save(f"Images/Smooth_Hist_Gif.gif", writer='pillow', fps=fps)
    
    
def main():
    # apep_plot('Apep_Plot')
    # apep_plot('Apep_Plot_No_Photodiss', custom_params={'comp_reduction':0})
    # apep_cone_plot()
    
    # smooth_hist_demo()
    smooth_hist_gif()
    



if __name__ == "__main__":
    main()
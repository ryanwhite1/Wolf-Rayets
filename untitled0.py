# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:14:43 2024

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

@jit
def dust_plume_for_gif(stardata):
    '''
    Parameters
    ----------
    stardata : dict
    '''
    phase = stardata['phase']%1
    
    period_s = stardata['period'] * 365.25 * 24 * 60 * 60
    
    n_orbits = 1
    n_t = 1000       # circles per orbital period
    n_points = 400   # points per circle
    n_particles = n_points * n_t * n_orbits
    n_time = n_t * n_orbits
    theta = 2 * jnp.pi * jnp.linspace(0, 1, n_points)
    times = period_s * jnp.linspace(phase, n_orbits + phase, n_time)
    particles, weights = gm.dust_plume_sub(theta, times, n_orbits, period_s, stardata)
    return particles, weights

def orbital_positions(stardata):
    
    phase = stardata['phase']%1
    
    period_s = stardata['period'] * 365.25 * 24 * 60 * 60
    
    n_orbits = 1
    n_t = 100       # circles per orbital period
    n_points = 40   # points per circle
    n_particles = n_points * n_t * n_orbits
    n_time = n_t * n_orbits
    theta = 2 * jnp.pi * jnp.linspace(0, 1, n_points)
    times = period_s * jnp.linspace(phase, n_orbits + phase, n_time)
    n_time = len(times)
    n_t = n_time / n_orbits
    ecc = stardata['eccentricity']
    # E, true_anomaly = kepler_solve(times, period_s, ecc)
    
    E, true_anomaly = gm.kepler(2 * jnp.pi * times / period_s, jnp.array([ecc]))
    
    a1, a2 = gm.calculate_semi_major(period_s, stardata['m1'], stardata['m2'])
    r1 = a1 * (1 - ecc * jnp.cos(E)) * 1e-3     # radius in km 
    r2 = a2 * (1 - ecc * jnp.cos(E)) * 1e-3
    # ws_ratio = stardata['windspeed1'] / stardata['windspeed2']
    
    positions1 = jnp.array([jnp.cos(true_anomaly), 
                            jnp.sin(true_anomaly), 
                            jnp.zeros(n_time)])
    positions2 = jnp.copy(positions1)
    positions1 *= r1      # position in the orbital frame
    positions2 *= -r2     # position in the orbital frame
    
    return positions1, positions2

def transform_orbits(pos1, pos2, stardata):
    pos1 = gm.euler_angles(pos1, stardata['asc_node'], stardata['inclination'], stardata['arg_peri'])
    pos2 = gm.euler_angles(pos2, stardata['asc_node'], stardata['inclination'], stardata['arg_peri'])
    pos1 = 60 * 60 * 180 / jnp.pi * jnp.arctan(pos1 / (stardata['distance'] * 3.086e13))
    pos2 = 60 * 60 * 180 / jnp.pi * jnp.arctan(pos2 / (stardata['distance'] * 3.086e13))
    return pos1, pos2

def transform_pole(pole, stardata):
    pole = gm.rotate_x(jnp.deg2rad(stardata['spin_inc'])) @ (gm.rotate_z(jnp.deg2rad(stardata['spin_Omega'])) @ pole)
    # pole = gm.rotate_z(jnp.deg2rad(stardata['spin_Omega'])) @ (gm.rotate_x(jnp.deg2rad(stardata['spin_inc'])) @ pole)
    pole = gm.euler_angles(pole, stardata['asc_node'], stardata['inclination'], stardata['arg_peri'])
    pole = 60 * 60 * 180 / jnp.pi * jnp.arctan(pole / (stardata['distance'] * 3.086e13))
    return pole


# @jit
def orbit_spiral_gif(stardata):
    '''
    '''
    starcopy = stardata.copy()
    fig, ax = plt.subplots(figsize=(6, 6))
    
    every = 1
    length = 10
    # now calculate some parameters for the animation frames and timing
    # nt = int(stardata['period'])    # roughly one year per frame
    nt = 30
    # nt = 10
    frames = jnp.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
    fps = len(frames) // length  # fps for the final animation
    
    phases = jnp.linspace(0, 1, nt)
    pos1, pos2 = orbital_positions(test_system)
    pos1, pos2 = transform_orbits(pos1, pos2, starcopy)
    
    
    lim = 2 * max(np.max(np.abs(pos1)), np.max(np.abs(pos2)))
    xbins = np.linspace(-lim, lim, 257)
    ybins = np.linspace(-lim, lim, 257)
    ax.set_aspect('equal')
    
    a1, _ = gm.calculate_semi_major(stardata['period'] * 365.25 * 24 * 60 * 60, stardata['m1'], stardata['m2'])
    
    # @jit
    def animate(i):
        ax.cla()
        # if i%20 == 0:
        #     print(i)
        print(i)
        starcopy['phase'] = phases[i] + 0.5
        particles, weights = dust_plume_for_gif(starcopy)
        
        pos1, pos2 = orbital_positions(starcopy)
        pole1 = pos1[:, -1] + jnp.array([0, 0, 0.0005 * a1])
        pole2 = pos1[:, -1] - jnp.array([0, 0, 0.0005 * a1])
        pos1, pos2 = transform_orbits(pos1, pos2, starcopy)

        X, Y, H = gm.spiral_grid_w_bins(particles, weights, starcopy, xbins, ybins)
        ax.pcolormesh(X, Y, H, cmap='hot')
        
        
        ax.plot(pos1[0, :], pos1[1, :], c='w')
        ax.plot(pos2[0, :], pos2[1, :], c='w')
        ax.scatter([pos1[0, -1], pos2[0, -1]], [pos1[1, -1], pos2[1, -1]], c=['tab:cyan', 'w'], s=100)
        
        pole1 = transform_pole(pole1, starcopy)
        pole2 = transform_pole(pole2, starcopy)
        ax.plot([pos1[0, -1], pole1[0]], [pos1[1, -1], pole1[1]], c='tab:blue')
        ax.plot([pos1[0, -1], pole2[0]], [pos1[1, -1], pole2[1]], c='tab:red')
        
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
        ax.set_facecolor('k')
        ax.set_axis_off()
        ax.text(0.3 * lim, -0.8 * lim, f"Phase = {starcopy['phase']%1:.2f}", c='w', fontsize=14)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        return fig, 

    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
    ani.save(f"orbit_spiral.gif", writer='pillow', fps=fps)
    
# test_system = wrb.apep.copy()

test_system = {"m1":22.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0.5, 
        "inclination":60.,       # degrees
        "asc_node":254.1,         # degrees
        "arg_peri":10.6,           # degrees
        "open_angle":40.,       # degrees (full opening angle)
        "period":1.,           # years
        "distance":10.,        # pc
        "windspeed1":0.1,       # km/s
        "windspeed2":2400.,      # km/s
        "turn_on":-180.,         # true anomaly (degrees)
        "turn_off":180.,         # true anomaly (degrees)
        "oblate":0.,
        "nuc_dist":0.0001, "opt_thin_dist":2.,           # nucleation and optically thin distance (AU)
        "acc_max":0.1,                                 # maximum acceleration (km/s/yr)
        "orb_sd":0., "orb_amp":0., "orb_min":180., "az_sd":30., "az_amp":0., "az_min":270.,
        "comp_incl":127.1, "comp_az":116.5, "comp_open":0., "comp_reduction":0., "comp_plume":1.,
        "phase":0.6, 
        "sigma":1.5,              # sigma for gaussian blur
        "histmax":1., "lum_power":1, 
        "spin_inc":45., "spin_Omega":90., "spin_oa_mult":0.5, "spin_vel_mult":1., "spin_oa_sd":60., "spin_vel_sd":60.}

orbit_spiral_gif(test_system)

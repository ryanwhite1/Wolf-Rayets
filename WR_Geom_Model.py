# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 08:36:43 2024

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

M_odot = 1.98e30
G = 6.67e-11
c = 299792458
yr2day = 365.25
kms2pcyr = 60*60*24*yr2day / (3.086e13) # km/s to pc/yr


def rotate_x(angle):
    arr = jnp.array([[1, 0, 0],
                     [0, jnp.cos(angle), jnp.sin(angle)],
                     [0, -jnp.sin(angle), jnp.cos(angle)]])
    return arr

def rotate_y(angle):
    arr = jnp.array([[jnp.cos(angle), 0, -jnp.sin(angle)],
                     [0, 1, 0],
                     [jnp.sin(angle), 0, jnp.cos(angle)]])
    return arr

def rotate_z(angle):
    arr = jnp.array([[jnp.cos(angle), jnp.sin(angle), 0],
                     [-jnp.sin(angle), jnp.cos(angle), 0],
                     [0, 0, 1]])
    return arr



def kepler_solve_sub_sub(i, E0_ecc_mi):
    '''
    '''
    E0, ecc, mi = E0_ecc_mi
    return (E0 - (E0 - ecc * jnp.sin(E0) - mi) / (1 - ecc * jnp.cos(E0)), ecc, mi)
def kepler_solve_sub(i, ecc, tol, M):
    ''' This is the main kepler equation solving step. 
    '''
    E0 = M[i]
    # Newton's formula to solve for eccentric anomaly
    E0 = lax.fori_loop(0, 20, kepler_solve_sub_sub, (E0, ecc, M[i]))[0]
    return E0
def kepler_solve(t, P, ecc):
    ''' Solver for Kepler's 2nd law giving the angle of an orbiter (rel. to origin) over time
    '''
    # follow the method in https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
    # to get true anomaly
    M = 2 * jnp.pi / P * t
    tol = 1e-8

    E = vmap(lambda i: kepler_solve_sub(i, ecc, tol, M))(jnp.arange(len(t)))
    # now output true anomaly (rad)
    return E, 2 * jnp.arctan2(jnp.sqrt(1 + ecc) * jnp.sin(E / 2), jnp.sqrt(1 - ecc) * jnp.cos(E / 2))


def dust_circle(i_nu, stardata, theta, plume_direction, widths):
    '''
    '''
    i, nu = i_nu
    x = nu / (2 * jnp.pi)
    transf_nu = 2 * jnp.pi * (x + jnp.floor(0.5 - x))
    turned_on = jnp.heaviside(transf_nu - jnp.deg2rad(stardata['turn_on']), 0)
    turned_off = jnp.heaviside(jnp.deg2rad(stardata['turn_off']) - transf_nu, 0)
    direction = plume_direction[:, i] / jnp.linalg.norm(plume_direction[:, i])
    
    half_angle = jnp.deg2rad(stardata['open_angle']) / 2

    circle = jnp.array([jnp.ones(len(theta)) * jnp.cos(half_angle), 
                        jnp.sin(half_angle) * jnp.sin(theta), 
                        jnp.sin(half_angle) * jnp.cos(theta)])
    
    circle *= widths[i]
    angle_x = jnp.arctan2(direction[1], direction[0])
    circle = rotate_z(angle_x) @ circle
    
    circle *= turned_on * turned_off
    
    # now calculate the weights of each point according the their orbital variation
    prop_orb = 1 - (1 - stardata['orb_amp']) * jnp.exp(-0.5 * (((transf_nu*180/jnp.pi + 180) - 180) / stardata['orb_sd'])**2) # weight proportion from orbital variation
    
    # now from azimuthal variation
    prop_az = 1 - (1 - stardata['az_amp']) * jnp.exp(-0.5 * ((theta * 180/jnp.pi - 270) / (stardata['az_sd']))**2)
    
    weights = jnp.ones(len(theta)) * jnp.max(jnp.array([prop_orb, 0])) * prop_az
    
    circle = jnp.array([circle[0, :], 
                        circle[1, :], 
                        circle[2, :],
                        weights])
    
    return circle

def calculate_semi_major(period_s, m1, m2):
    '''
    '''
    m1_kg = m1 * M_odot                                 # mass of stars in kg
    m2_kg = m2 * M_odot
    M_kg = m1_kg + m2_kg                   # total mass in kg
    M = m1 + m2                             # total mass in solar masses
    mu = G * M
    a = jnp.cbrt((period_s / (2 * jnp.pi))**2 * mu)         # semi-major axis of the system (total separation)
    a1 = m2_kg / M_kg * a                                   # semi-major axis of first body (meters)
    a2 = a - a1                                             # semi-major axis of second body
    return a1, a2

@jit
def dust_plume(stardata):
    '''
    Parameters
    ----------
    stardata : dict
    '''
    phase = stardata['phase']%1
    ecc = stardata['eccentricity']
    period_s = stardata['period'] * 365.25 * 24 * 60 * 60
    
    n_orbits = 1
    n_t = 1000       # circles per orbital period
    n_points = 400   # points per circle
    n_particles = n_points * n_t * n_orbits
    n_time = n_t * n_orbits
    
    theta = 2 * jnp.pi * jnp.linspace(0, 1, n_points)
    
    times = period_s * jnp.linspace(phase, n_orbits + phase, n_time)
    
    E, true_anomaly = kepler_solve(times, period_s, ecc)
    
    a1, a2 = calculate_semi_major(period_s, stardata['m1'], stardata['m2'])
    r1 = a1 * (1 - ecc * jnp.cos(E)) * 1e-3     # radius in km 
    r2 = a2 * (1 - ecc * jnp.cos(E)) * 1e-3
    # ws_ratio = stardata['windspeed1'] / stardata['windspeed2']
    
    positions1 = jnp.array([jnp.cos(true_anomaly), 
                            jnp.sin(true_anomaly), 
                            jnp.zeros(n_time)])
    positions2 = jnp.copy(positions1)
    positions1 *= -r1      # position in the orbital frame
    positions2 *=  r2     # position in the orbital frame
    
    widths = stardata['windspeed1'] * period_s * (n_orbits - jnp.arange(n_time) / n_t)
    
    plume_direction = positions1 - positions2               # get the line of sight from first star to the second in the orbital frame
    
        
    particles = vmap(lambda i_nu: dust_circle(i_nu, stardata, theta, plume_direction, widths))((jnp.arange(n_time), true_anomaly))

    weights = particles[:, 3, :].flatten()
    particles = particles[:, :3, :]
    
    
    particles = jnp.array([jnp.ravel(particles[:, 0, :]),
                           jnp.ravel(particles[:, 1, :]),
                           jnp.ravel(particles[:, 2, :])])

    particles = rotate_z(jnp.deg2rad(- stardata['asc_node'])) @ (
            rotate_x(jnp.deg2rad(- stardata['inclination'])) @ (
            rotate_z(jnp.deg2rad(- stardata['arg_peri'])) @ particles))

    return 60 * 60 * 180 / jnp.pi * jnp.arctan(particles / (stardata['distance'] * 3.086e13)), weights

@jit
def spiral_grid(particles, weights, stardata):
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
    im_size = 256
    
    x = particles[0, :]
    y = particles[1, :]
    
    weights = jnp.where((x != 0) & (y != 0), weights, 0)
    

    H, xedges, yedges = jnp.histogram2d(y, x, bins=im_size, weights=weights)
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
    
    return X, Y, H

def plot_spiral(X, Y, H):
    ''' Plots the histogram given by X, Y edges and H densities
    '''
    fig, ax = plt.subplots()
    
    ax.pcolormesh(X, Y, H, cmap='hot')
    # import matplotlib.colors as cols
    # ax.pcolormesh(X, Y, H, norm=cols.LogNorm(vmin=1, vmax=H.max()))
    # ax.pcolormesh(X, Y, H, norm=cols.PowerNorm(gamma=1/2), cmap='hot')
    ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')


# @jit
def spiral_gif(stardata):
    '''
    '''
    starcopy = stardata.copy()
    fig, ax = plt.subplots()
    
    # im_size = 256
    # im = np.zeros((im_size, im_size))
    starcopy['phase'] = 0.01
    starcopy['sigma'] = 2
    particles, weights = dust_plume(stardata)
    X, Y, H = spiral_grid(particles, weights, starcopy)
    xmin, xmax = jnp.min(X), jnp.max(X)
    ymin, ymax = jnp.min(Y), jnp.max(Y)
    # border = [[xmin, xmax], [ymin, ymax]]
    # bins = [X, Y]
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal', 
           xlabel='Relative RA (")', ylabel='Relative Dec (")')
    
    starcopy['phase'] = 0.5
    particles, weights = dust_plume(starcopy)
    X, Y, H = spiral_grid(particles, weights, starcopy)
    # vmin, vmax = jnp.min(H), jnp.max(H)
    
    every = 1
    length = 10
    # now calculate some parameters for the animation frames and timing
    nt = int(stardata['period'])    # roughly one year per frame
    # nt = 10
    frames = jnp.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
    fps = len(frames) // length  # fps for the final animation
    
    phases = jnp.linspace(0, 1, nt)
    
    # @jit
    def animate(i):
        if (i // every)%20 == 0:
            print(f"{i // every} / {len(frames)}")
        # print(i)
        starcopy['phase'] = phases[i] + 0.5
        particles, weights = dust_plume(starcopy)
        X, Y, H = spiral_grid(particles, weights, starcopy)
        # ax.imshow(H, extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax, cmap='Greys')
        # ax.pcolormesh(xedges, yedges[::-1], H, vmax=vmax)
        ax.pcolormesh(X, Y, H, cmap='hot')
        return fig, 

    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
    ani.save(f"animation.gif", writer='pillow', fps=fps)
    
def plot_3d(particles):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    n = 23
    ax.scatter(particles[0, ::n], particles[1, ::n], particles[2, ::n], alpha=0.1)
    
def plot_orbit(stardata):
    ## plots orbits
    theta = np.linspace(0, 2 * np.pi, 100)
    r1 = stardata['p1'] / (1 + stardata['eccentricity'] * np.cos(theta))
    r2 = stardata['p2'] / (1 + stardata['eccentricity'] * np.cos(theta))

    x1, y1 = r1 * np.cos(theta), r1 * np.sin(theta)
    x2, y2 = -r2 * np.cos(theta), -r2 * np.sin(theta)

    fig, ax = plt.subplots()

    ax.plot(x1, y1)
    ax.plot(x2, y2)
    ax.set_aspect('equal')
    
# @jit
# def calculate_orbit(stardata):
#     ''' Fills out our binary data dictionary with calculated quantities.
#     Parameters
#     ----------
#     stardata : dict
#     '''
#     stardata['period_s'] = stardata['period'] * yr2day * 24 * 60 * 60           # orbital period in s
#     stardata['m1_kg'] = stardata['m1'] * M_odot                                 # mass of stars in kg
#     stardata['m2_kg'] = stardata['m2'] * M_odot
#     stardata['M_kg'] = stardata['m1_kg'] + stardata['m2_kg']                    # total mass in kg
#     stardata['M'] = stardata['m1'] + stardata['m2']                             # total mass in solar masses
#     mu = G * stardata['M']
#     # kms2masyr = np.arctan(kms2pcyr / stardata['distance']) * 180/np.pi * 60 * 60 * 1000     # conversion from km/s to mas/yr at the system distance
#     # ws1 = stardata['windspeed1'] * kms2masyr
#     # ws2 = stardata['windspeed2'] * kms2masyr
#     stardata['a'] = np.cbrt((stardata['period_s'] / (2 * np.pi))**2 * mu)       # semi-major axis of the system (total separation)
#     stardata['a1'] = stardata['m2_kg']  / stardata['M_kg'] * stardata['a']      # semi-major axis of first body (meters)
#     stardata['a2'] = stardata['a'] - stardata['a1']                             # semi-major axis of second body
    
#     stardata['p1'] = stardata['a1'] * (1 - stardata['eccentricity']**2)         # periastron (meters) of first star from barycenter
#     stardata['p2'] = stardata['a2'] * (1 - stardata['eccentricity']**2)


# below are rough params for Apep 
apep = {"m1":15.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0.7, 
        "inclination":25.,       # degrees
        "asc_node":-88.,         # degrees
        "arg_peri":0.,           # degrees
        "open_angle":125.,       # degrees (full opening angle)
        "period":125.,           # years
        "distance":2400.,        # pc
        "windspeed1":700.,       # km/s
        "windspeed2":2400.,      # km/s
        "turn_on":-114.,         # true anomaly (degrees)
        "turn_off":150.,         # true anomaly (degrees)
        "orb_sd":0., "orb_amp":0., "az_sd":0., "az_amp":0., 
        "phase":0.6, 
        "sigma":3.,              # sigma for gaussian blur
        "histmax":1.}

# below are rough params for WR 48a
WR48a = {"m1":15.,                  # solar masses
        "m2":10.,                   # solar masses
        "eccentricity":0.1, 
        "inclination":75,           # degrees
        "asc_node":0,               # degrees
        "arg_peri":20,              # degrees
        "open_angle":110,           # degrees (full opening angle)
        "period":32.5,              # years
        "distance":3500,            # pc
        "windspeed1":700,           # km/s
        "windspeed2":2400,          # km/s
        "turn_on":-140,             # true anomaly (degrees)
        "turn_off":140,             # true anomaly (degrees)
        "orb_sd":0, "orb_amp":0, "az_sd":0, "az_amp":0, 
        "phase":0.6, 
        "sigma":2,                  # sigma for gaussian blur
        "histmax":1}


# below are rough params for WR 112
WR112 = {"m1":15.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0., 
        "inclination":100.,       # degrees
        "asc_node":75.,         # degrees
        "arg_peri":170.,           # degrees
        "open_angle":110.,       # degrees (full opening angle)
        "period":19,           # years
        "distance":2400,        # pc
        "windspeed1":700,       # km/s
        "windspeed2":2400,      # km/s
        "turn_on":-180,         # true anomaly (degrees)
        "turn_off":180,         # true anomaly (degrees)
        "orb_sd":0, "orb_amp":0, "az_sd":0, "az_amp":0, 
        "phase":0.6, 
        "sigma":2,              # sigma for gaussian blur
        "histmax":1}

# below are rough params for WR 140
WR140 = {"m1":8.4,                # solar masses
        "m2":20,                # solar masses
        "eccentricity":0.8964, 
        "inclination":119.6,       # degrees
        "asc_node":0,         # degrees
        "arg_peri":180-46.8,           # degrees
        "open_angle":80,       # degrees (full opening angle)
        "period":2896.35/365.25,           # years
        "distance":1670,        # pc
        "windspeed1":2600,       # km/s
        "windspeed2":2400,      # km/s
        "turn_on":-135,         # true anomaly (degrees)
        "turn_off":135,         # true anomaly (degrees)
        "orb_sd":80, "orb_amp":0, "az_sd":60, "az_amp":0, 
        "phase":0.6, 
        "sigma":2,              # sigma for gaussian blur
        "histmax":1}

# below are rough params for WR 104
WR104 = {"m1":10,                # solar masses
        "m2":20,                # solar masses
        "eccentricity":0.06, 
        "inclination":180-15,       # degrees
        "asc_node":90,         # degrees
        "arg_peri":0,           # degrees
        "open_angle":60,       # degrees (full opening angle)
        "period":241.5/365.25,           # years
        "distance":2580,        # pc
        "windspeed1":1200,       # km/s
        "windspeed2":2000,      # km/s
        "turn_on":-180,         # true anomaly (degrees)
        "turn_off":180,         # true anomaly (degrees)
        "orb_sd":0, "orb_amp":0, "az_sd":0, "az_amp":0, 
        "phase":0.7, 
        "sigma":6,              # sigma for gaussian blur
        "histmax":0.2}


# # for i in range(10):
# t1 = time.time()
# particles, weights = dust_plume(apep)

# X, Y, H = spiral_grid(particles, weights, apep)
# print(time.time() - t1)
# plot_spiral(X, Y, H)


# spiral_gif(apep)



# ### --- INFERENCE --- ###
# particles, weights = dust_plume(apep)
    
# X, Y, H = spiral_grid(particles, weights, apep)
# obs_err = 0.01 * np.max(H)
# H += np.random.normal(0, obs_err, H.shape)
# plot_spiral(X, Y, H)



# obs = H.flatten()
# obs_err = obs_err * jnp.ones(len(obs))

# fig, ax = plt.subplots()

# ax.plot(jnp.arange(len(obs)), obs, lw=0.5)


# # fig, ax = plt.subplots()

# # ax.plot(jnp.arange(len(obs)), obs**3, lw=0.5)


# ### --- EMCEE --- ###


# # # @jit
# # def log_prior(state):
# #     # m1 = jnp.heaviside(state['m1'], 0) * jnp.heaviside(200 - state['m1'], 1)
# #     # m2 = jnp.heaviside(state['m2'], 0) * jnp.heaviside(200 - state['m2'], 1)
# #     # period = jnp.heaviside(state['period'], 0) * jnp.heaviside(1e3 - state['period'], 1)
# #     # eccentricity = jnp.heaviside(state['eccentricity'], 1) * jnp.heaviside(1 - state['eccentricity'], 0)
# #     # inclination = jnp.heaviside(360 - state['inclination'], 1) * (1 - jnp.heaviside(-state['inclination'] - 360, 1))
# #     # asc_node = jnp.heaviside(360 - state['asc_node'], 1) * (1 - jnp.heaviside(-state['asc_node'] - 360, 1))
# #     # arg_peri = jnp.heaviside(360 - state['arg_peri'], 1) * (1 - jnp.heaviside(-state['arg_peri'] - 360, 1))
# #     # open_angle = jnp.heaviside(180 - state['open_angle'], 0) * jnp.heaviside(state['open_angle'], 0)
# #     # distance = jnp.heaviside(state['distance'], 0)
# #     # turn_on = jnp.heaviside(180 + state['turn_on'], 1) * (1 - jnp.heaviside(-state['turn_on'] - 180, 0))
# #     # turn_off = jnp.heaviside(180 + state['turn_off'], 1) * (1 - jnp.heaviside(-state['turn_off'] - 180, 0))
    
# #     # return (1. - m1*m2*period*eccentricity*inclination*asc_node*arg_peri*open_angle*distance*turn_on*
# #     #         turn_off) * -jnp.inf
    
# #     array = jnp.array([0., -jnp.inf])
# #     eccentricity = jnp.heaviside(state[0], 1) * jnp.heaviside(1 - state[0], 0)
# #     inclination = jnp.heaviside(360 - state[1], 1) * (1 - jnp.heaviside(-state[1] - 360, 1))
# #     asc_node = jnp.heaviside(360 - state[2], 1) * (1 - jnp.heaviside(-state[2] - 360, 1))
# #     open_angle = jnp.heaviside(180 - state[3], 0) * jnp.heaviside(state[3], 0)
    
# #     # # print(eccentricity, inclination, asc_node, open_angle)
# #     # if not (1 - eccentricity*inclination*asc_node*open_angle):
# #     #     return 0. 
# #     # else:
# #     #     return -jnp.inf
# #     # a = (1. - eccentricity*inclination*asc_node*open_angle) * -jnp.inf
# #     num = 1 - [eccentricity*inclination*asc_node*open_angle][0]
# #     num = jnp.array(num, int)
# #     return array[num]
    
# #     # a = (1. - eccentricity*inclination*asc_node*open_angle) * -jnp.inf
# #     # return -np.min([np.nan_to_num(a), np.inf])
# # # @jit 
# # def log_likelihood(state, obs, obs_err):
    
# #     data_dict = apep.copy()
# #     data_dict['eccentricity'] = state[0]
# #     data_dict['inclination'] = state[1]
# #     data_dict['asc_node'] = state[2]
# #     data_dict['open_angle'] = state[3]
    
# #     particles, weights = dust_plume(data_dict)
# #     _, _, model = spiral_grid(particles, weights, data_dict)
# #     model = model.flatten()
# #     return -0.5 * jnp.sum((obs - model)**2 / obs_err**2)

# # @jit 
# # def log_prob(state, obs, obs_err):
# #     lp = log_prior(state)
# #     isfinite = jnp.array(jnp.isfinite(lp), int)
# #     return_arr = jnp.array([-jnp.inf, lp + log_likelihood(state, obs, obs_err)])
# #     return return_arr[isfinite]


# # nwalkers = 10

# # pos = np.array([apep['eccentricity'], apep['inclination'], apep['asc_node'], apep['open_angle']])
# # ndim = len(pos)
# # pos = pos * np.ones((nwalkers, ndim))
# # pos += 1e-1 * np.random.normal(0, 0.5, pos.shape)

# # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(obs, obs_err))
# # sampler.run_mcmc(pos, 1000, progress=True);


# # fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
# # samples = sampler.get_chain()
# # labels = ["e", "i", "an", "oa"]
# # for i in range(ndim):
# #     ax = axes[i]
# #     ax.plot(samples[:, :, i], "k", alpha=0.3)
# #     ax.set_xlim(0, len(samples))
# #     ax.set_ylabel(labels[i])
# #     ax.yaxis.set_label_coords(-0.1, 0.5)
    
    
# # flat_samples = sampler.get_chain(discard=300, flat=True)
# # import corner
# # import jax
# # labels = ['ecc', 'incl', 'asc_node', 'op_ang']
# # truths = np.array([apep['eccentricity'], apep['inclination'], apep['asc_node'], apep['open_angle']])
# # fig = corner.corner(flat_samples, labels=labels, truths=truths)





# # ### --- BLACKJAX --- ###
# # def log_prior(state):
# #     array = jnp.array([0., -jnp.inf])
# #     m1 = jnp.heaviside(state['m1'], 0) * jnp.heaviside(200 - state['m1'], 1)
# #     m2 = jnp.heaviside(state['m2'], 0) * jnp.heaviside(200 - state['m2'], 1)
# #     period = jnp.heaviside(state['period'], 0) * jnp.heaviside(1e3 - state['period'], 1)
# #     eccentricity = jnp.heaviside(state['eccentricity'], 1) * jnp.heaviside(1 - state['eccentricity'], 0)
# #     inclination = jnp.heaviside(360 - state['inclination'], 1) * (1 - jnp.heaviside(-state['inclination'] - 360, 1))
# #     asc_node = jnp.heaviside(360 - state['asc_node'], 1) * (1 - jnp.heaviside(-state['asc_node'] - 360, 1))
# #     arg_peri = jnp.heaviside(360 - state['arg_peri'], 1) * (1 - jnp.heaviside(-state['arg_peri'] - 360, 1))
# #     open_angle = jnp.heaviside(180 - state['open_angle'], 0) * jnp.heaviside(state['open_angle'], 0)
# #     distance = jnp.heaviside(state['distance'], 0)
# #     turn_on = jnp.heaviside(180 + state['turn_on'], 1) * (1 - jnp.heaviside(-state['turn_on'] - 180, 0))
# #     turn_off = jnp.heaviside(180 + state['turn_off'], 1) * (1 - jnp.heaviside(-state['turn_off'] - 180, 0))
    
# #     num = 1 - [m1*m2*period*eccentricity*inclination*asc_node*arg_peri*open_angle*distance*turn_on*turn_off][0]
# #     num = jnp.array(num, int)
    
# #     return array[num]

# # def log_likelihood(state, obs, obs_err):
# #     particles, weights = dust_plume(state)
# #     _, _, model = spiral_grid(particles, weights, state)
# #     model = model.flatten()
# #     return -0.5 * jnp.sum((obs - model)**2 / obs_err**2)

# # def log_prob(state, obs=obs, obs_err=obs_err):
# #     lp = log_prior(state)
# #     isfinite = jnp.array(jnp.isfinite(lp), int)
# #     return_arr = jnp.array([-jnp.inf, lp + log_likelihood(state, obs, obs_err)])
# #     return return_arr[isfinite]

# # import blackjax 
# # inverse_mass_matrix = jnp.ones(len(apep)) * 0.05
# # step_size = 1e-3
# # hmc = blackjax.nuts(log_prob, step_size, inverse_mass_matrix)

# # initial_position = apep
# # state = hmc.init(initial_position)
# # import jax
# # rng_key = jax.random.key(0)
# # step = jit(hmc.step)


# # def inference_loop(rng_key, kernel, initial_state, num_samples):

# #     @jax.jit
# #     def one_step(state, rng_key):
# #         state, _ = kernel(rng_key, state)
# #         return state, state

# #     keys = jax.random.split(rng_key, num_samples)
# #     _, states = jax.lax.scan(one_step, initial_state, keys)

# #     return states

# # states = inference_loop(rng_key, step, state, 1000)

# # mcmc_samples = states.position

# # samples = np.ones((len(list(mcmc_samples.keys())), len(mcmc_samples[list(mcmc_samples.keys())[0]])))
# # for i, key in enumerate(mcmc_samples.keys()):
# #     samples[i, :] = mcmc_samples[key]

# # import corner
# # corner.corner(samples)






# ### --- NUMPYRO --- ###
# import numpyro, chainconsumer, jax
# import numpyro.distributions as dists

# num_chains = 1

# def apep_model(Y, E):
#     m1 = numpyro.sample("m1", dists.Normal(apep['m1'], 5.))
#     m2 = numpyro.sample("m2", dists.Normal(apep['m2'], 5.))
#     eccentricity = numpyro.sample("eccentricity", dists.Normal(apep['eccentricity'], 0.05))
#     inclination = numpyro.sample("inclination", dists.Normal(apep['inclination'], 20.))
#     asc_node = numpyro.sample("asc_node", dists.Normal(apep['asc_node'], 20.))
#     arg_peri = numpyro.sample("arg_peri", dists.Normal(apep['arg_peri'], 20.))
#     open_angle = numpyro.sample("open_angle", dists.Normal(apep['open_angle'], 10.))
#     period = numpyro.sample("period", dists.Normal(apep['period'], 40.))
#     distance = numpyro.sample("distance", dists.Normal(apep['distance'], 500.))
#     windspeed1 = numpyro.sample("windspeed1", dists.Normal(apep['windspeed1'], 200.))
#     windspeed2 = numpyro.sample("windspeed2", dists.Normal(apep['windspeed2'], 200.))
#     turn_on = numpyro.sample("turn_on", dists.Normal(apep['turn_on'], 10.))
#     turn_off = numpyro.sample("turn_off", dists.Normal(apep['turn_off'], 10.))
#     orb_sd = numpyro.sample("orb_sd", dists.Exponential(1./10.))
#     orb_amp = numpyro.sample("orb_amp", dists.Exponential(1./0.1))
#     az_sd = numpyro.sample("az_sd", dists.Exponential(1./10.))
#     az_amp = numpyro.sample("az_amp", dists.Exponential(1./0.1))
#     phase = numpyro.sample("phase", dists.Uniform(0., 1.))
#     # sigma = numpyro.sample("sigma", dists.Uniform(0.01, 10.))
#     # histmax = numpyro.sample("histmax", dists.Uniform(0., 1.))
#     # open_angle = apep['open_angle']
#     # period = apep['period']
#     # distance = apep['distance']
#     # windspeed1 = apep['windspeed1']
#     # windspeed2 = apep['windspeed2']
#     # turn_on = apep['turn_on']
#     # turn_off = apep['turn_off']
#     # orb_sd = apep['orb_sd']
#     # orb_amp = apep['orb_amp']
#     # az_sd = apep['az_sd']
#     # az_amp = apep['az_amp']
#     # phase = apep['phase']
#     sigma = apep['sigma']
#     histmax = apep['histmax']
    
#     # constrain_fn
    
#     with numpyro.plate('data', 1):
#         params = {"m1":m1, "m2":m2,                # solar masses
#                 "eccentricity":eccentricity, 
#                 "inclination":inclination, "asc_node":asc_node, "arg_peri":arg_peri,           # degrees
#                 "open_angle":open_angle,       # degrees (full opening angle)
#                 "period":period, "distance":distance,        # pc
#                 "windspeed1":windspeed1, "windspeed2":windspeed2,      # km/s
#                 "turn_on":turn_on, "turn_off":turn_off,     # true anomaly (degrees)
#                 "orb_sd":orb_sd, "orb_amp":orb_amp, "az_sd":az_sd, "az_amp":az_amp, 
#                 "phase":phase, "sigma":sigma, "histmax":histmax}
#         samp_particles, samp_weights = dust_plume(params)
#         _, _, samp_H = spiral_grid(samp_particles, samp_weights, params)
#         samp_H = samp_H.flatten()
#         samp_H = jnp.nan_to_num(samp_H, 1e4)
#         numpyro.sample('y', dists.Normal(samp_H, E), obs=Y)



# init_params = apep.copy()
# init_params_arr = init_params.copy()
# for key in init_params.keys():
#     init_params_arr[key] = jnp.ones(num_chains) * init_params_arr[key]


# init_params = numpyro.infer.util.constrain_fn(apep_model, (obs, obs_err), {}, init_params)
# # sampler = numpyro.infer.MCMC(numpyro.infer.NUTS(apep_model, 
# #                                                 init_strategy=numpyro.infer.initialization.init_to_value(values=apep)),
# #                               num_chains=1,
# #                               num_samples=300,
# #                               num_warmup=20)
# sampler = numpyro.infer.MCMC(numpyro.infer.SA(apep_model, init_strategy=numpyro.infer.initialization.init_to_value(values=init_params)),
#                               num_chains=num_chains,
#                               num_samples=8000,
#                               num_warmup=1000)
# sampler.run(jax.random.PRNGKey(1), obs, obs_err*10, init_params=init_params_arr)

# results = sampler.get_samples()
# C = chainconsumer.ChainConsumer()
# C.add_chain(results, name='MCMC Results')
# C.plotter.plot(truth=apep)

# maxlike = apep.copy()
# for key in results.keys():
#     maxlike[key] = np.median(results[key])


# samp_particles, samp_weights = dust_plume(maxlike)
# X, Y, samp_H = spiral_grid(samp_particles, samp_weights, maxlike)
# plot_spiral(X, Y, samp_H)
    















### --- GUI Plot --- ###

import tkinter

import numpy as np

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

root = tkinter.Tk()
root.wm_title("Embedding in Tk")

fig, ax = plt.subplots()
particles, weights = dust_plume(apep)
X, Y, H = spiral_grid(particles, weights, apep)
plot_spiral(X, Y, H)
mesh = ax.pcolormesh(X, Y, H, cmap='hot')
ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()

# pack_toolbar=False will make it easier to use a layout manager later on.
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()

canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

button_quit = tkinter.Button(master=root, text="Quit", command=root.destroy)

starcopy = apep.copy()
def update_frequency(param, new_val):
    # retrieve frequency
    starcopy[param] = float(new_val)

    # update data
    particles, weights = dust_plume(starcopy)
    X, Y, H = spiral_grid(particles, weights, starcopy)
    mesh.update({'array':H.ravel()})
    
    new_coords = mesh._coordinates
    new_coords[:, :, 0] = X
    new_coords[:, :, 1] = Y
    mesh._coordinates = new_coords
    ax.set(xlim=(np.min(X), np.max(X)), ylim=(np.min(Y), np.max(Y)))

    # required to update canvas and attached toolbar!
    canvas.draw()


s1 = tkinter.Scale(root, from_=0, to=0.99, orient=tkinter.HORIZONTAL,
                              command=lambda v: update_frequency('eccentricity', v), label="Eccentricity", resolution=0.01)
s2 = tkinter.Scale(root, from_=-180, to=180, orient=tkinter.HORIZONTAL,
                              command=lambda v: update_frequency('inclination', v), label="Inclination", resolution=0.1)
s3 = tkinter.Scale(root, from_=0, to=180, orient=tkinter.HORIZONTAL,
                              command=lambda v: update_frequency('open_angle', v), label="Open Angle", resolution=0.1)
s4 = tkinter.Scale(root, from_=0., to=1.5, orient=tkinter.HORIZONTAL,
                              command=lambda v: update_frequency('phase', v), label="Phase", resolution=0.01)

ss = [s1, s2, s3, s4]
# Packing order is important. Widgets are processed sequentially and if there
# is no space left, because the window is too small, they are not displayed.
# The canvas is rather flexible in its size, so we pack it last which makes
# sure the UI controls are displayed as long as possible.
button_quit.pack(side=tkinter.BOTTOM)
for i, s in enumerate(ss[::-1]):
    s.pack(side=tkinter.BOTTOM, fill=tkinter.X)
toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

tkinter.mainloop()


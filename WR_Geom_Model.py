# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 08:36:43 2024

@author: ryanw
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, grad
import jax
import jax.lax as lax
import jax.scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import jax.scipy.signal as signal
from matplotlib import animation
import time
import emcee

import WR_binaries as wrb
jax.config.update("jax_enable_x64", True)

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
    turn_on = jnp.deg2rad(stardata['turn_on'])
    turn_off = jnp.deg2rad(stardata['turn_off'])
    turned_on = jnp.heaviside(transf_nu - turn_on, 0)
    turned_off = jnp.heaviside(turn_off - transf_nu, 0)
    direction = plume_direction[:, i] / jnp.linalg.norm(plume_direction[:, i])
    
    half_angle = jnp.deg2rad(stardata['open_angle']) / 2

    circle = jnp.array([jnp.ones(len(theta)) * jnp.cos(half_angle), 
                        jnp.sin(half_angle) * jnp.sin(theta), 
                        (1 - stardata['oblate']) * jnp.sin(half_angle) * jnp.cos(theta)])
    
    circle *= widths[i]
    angle_x = jnp.arctan2(direction[1], direction[0])
    circle = rotate_z(angle_x) @ circle
    
    # circle *= turned_on * turned_off
    weights = jnp.ones(len(theta)) * turned_on * turned_off
    
    # weights = jnp.ones(len(theta))
    # sigma = jnp.deg2rad(10)
    # mult = 0.1
    # weights *= 1 - (1 - turned_on - mult * jnp.exp(-0.5 * ((transf_nu - turn_on) / sigma)**2))
    # weights *= 1 - (1 - turned_off - mult * jnp.exp(-0.5 * ((transf_nu - turn_off) / sigma)**2))
    
    
    alpha = jnp.deg2rad(stardata['comp_incl'])
    beta = jnp.deg2rad(stardata['comp_az'])
    comp_halftheta = jnp.deg2rad(stardata['comp_open'] / 2)
    x = circle[0, :]
    y = circle[1, :]
    z = circle[2, :]
    r = jnp.sqrt(x**2 + y**2 + z**2)
    particles_alpha = jnp.arccos(z / r)
    particles_beta = jnp.sign(y) * jnp.arccos(x / jnp.sqrt(x**2 + y**2))
    
    ### to get angular separation of the points on the sphere, I used the cos(alpha) = ... formula from
    # https://www.atnf.csiro.au/people/Tobias.Westmeier/tools_spherical.php#:~:text=The%20angular%20separation%20of%20two%20points%20on%20a%20shpere&text=cos(%CE%B1)%3Dcos(%CF%911)cos(,%CF%861%E2%88%92%CF%862).
    term1 = jnp.cos(alpha) * jnp.cos(particles_alpha)
    term2 = jnp.sin(alpha) * jnp.sin(particles_alpha) * jnp.cos(beta - particles_beta)
    angular_dist = jnp.arccos(term1 + term2)
    
    photodis_prop = 1
    ## linear scaling for companion photodissociation
    # companion_dissociate = jnp.where(angular_dist < comp_halftheta,
    #                                  (1 - stardata['comp_reduction'] * jnp.ones(len(weights))), jnp.ones(len(weights)))
    ## gaussian scaling for companion photodissociation
    comp_gaussian = 1 - stardata['comp_reduction'] * jnp.exp(-0.5 * (angular_dist / comp_halftheta)**2)
    comp_gaussian = jnp.maximum(comp_gaussian, jnp.zeros(len(comp_gaussian))) # need weight value to be between 0 and 1
    companion_dissociate = jnp.where(angular_dist < photodis_prop * comp_halftheta,
                                      comp_gaussian, jnp.ones(len(weights)))
    
    in_comp_plume = jnp.where((photodis_prop * comp_halftheta < angular_dist) & (angular_dist < comp_halftheta),
                              jnp.ones(len(x)), jnp.zeros(len(x)))
    
    # now we need to generate angles around the plume edge that are inconsistent to the other rings so that it smooths out
    # i.e. instead of doing linspace(0, 2*pi, len(x)), just do a large number multiplied by our ring number and convert that to [0, 2pi]
    ring_theta = jnp.linspace(0, i * len(x), len(x))%(2*jnp.pi)
    
    ## The coordinate transformations below are from user DougLitke from
    ## https://math.stackexchange.com/questions/643130/circle-on-sphere?newreg=42e38786904e43a0a2805fa325e52b92
    new_x = r * (jnp.sin(comp_halftheta) * jnp.cos(alpha) * jnp.cos(beta) * jnp.cos(ring_theta) - jnp.sin(comp_halftheta) * jnp.sin(beta) * jnp.sin(ring_theta) + jnp.cos(comp_halftheta) * jnp.sin(alpha) * jnp.cos(beta))
    new_y = r * (jnp.sin(comp_halftheta) * jnp.cos(alpha) * jnp.sin(beta) * jnp.cos(ring_theta) + jnp.sin(comp_halftheta) * jnp.cos(beta) * jnp.sin(ring_theta) + jnp.cos(comp_halftheta) * jnp.sin(alpha) * jnp.sin(beta))
    new_z = r * (-jnp.sin(comp_halftheta) * jnp.sin(alpha) * jnp.cos(ring_theta) + jnp.cos(comp_halftheta) * jnp.cos(alpha))
    
    x = x + in_comp_plume * (-x + new_x)
    y = y + in_comp_plume * (-y + new_y)
    z = z + in_comp_plume * (-z + new_z)
    
    circle = jnp.array([x, y, z])
    
    weights *= (1 - in_comp_plume * (1 - stardata['comp_plume']))
    
    
    
    # now calculate the weights of each point according the their orbital variation
    prop_orb = 1 - (1 - stardata['orb_amp']) * jnp.exp(-0.5 * (((transf_nu*180/jnp.pi + 180) - stardata['orb_min']) / stardata['orb_sd'])**2) # weight proportion from orbital variation
    
    # now from azimuthal variation
    prop_az = 1 - (1 - stardata['az_amp']) * jnp.exp(-0.5 * ((theta * 180/jnp.pi - stardata['az_min']) / (stardata['az_sd']))**2)
    
    # we need our orbital proportion to be between 0 and 1
    prop_orb = jnp.min(jnp.array([prop_orb, 1]))
    prop_orb = jnp.max(jnp.array([prop_orb, 0]))
    # and the same for our azimuthal proportion
    prop_az = jnp.minimum(jnp.maximum(prop_az, jnp.zeros(len(prop_az))), jnp.ones(len(prop_az)))
    weights *= prop_orb * prop_az
    
    weights *= companion_dissociate
    
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
    M_kg = m1_kg + m2_kg                                # total mass in kg
    # M = m1 + m2                                         # total mass in solar masses
    mu = G * M_kg
    a = jnp.cbrt((period_s / (2 * jnp.pi))**2 * mu)/1000    # semi-major axis of the system (total separation)
    a1 = m2_kg / M_kg * a                                   # semi-major axis of first body (meters)
    a2 = a - a1                                             # semi-major axis of second body
    return a1, a2

def dust_plume_sub(theta, times, n_orbits, period_s, stardata):
    
    n_time = len(times)
    n_t = n_time / n_orbits
    ecc = stardata['eccentricity']
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
def dust_plume(stardata):
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
    particles, weights = dust_plume_sub(theta, times, n_orbits, period_s, stardata)
    return particles, weights

@jit
def dust_plume_2orb(stardata):
    '''
    Parameters
    ----------
    stardata : dict
    '''
    phase = stardata['phase']%1
    period_s = stardata['period'] * 365.25 * 24 * 60 * 60
    n_orbits = 2
    n_t = 1000       # circles per orbital period
    n_points = 400   # points per circle
    n_particles = n_points * n_t * n_orbits
    n_time = n_t * n_orbits
    theta = 2 * jnp.pi * jnp.linspace(0, 1, n_points)
    times = period_s * jnp.linspace(phase, n_orbits + phase, n_time)
    particles, weights = dust_plume_sub(theta, times, n_orbits, period_s, stardata)
    return particles, weights
    
    

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


# for i in range(10):
# t1 = time.time()
# particles, weights = dust_plume(wrb.apep)

# X, Y, H = spiral_grid(particles, weights, wrb.apep)
# print(time.time() - t1)
# plot_spiral(X, Y, H)


# spiral_gif(apep)




    


















# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 08:36:43 2024

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from matplotlib import animation
import time

def rotate_x(angle):
    arr = np.zeros((3, 3))
    arr[:, 0] = [1, 0, 0]
    arr[:, 1] = [0, np.cos(angle), np.sin(angle)]
    arr[:, 2] = [0, -np.sin(angle), np.cos(angle)]
    return arr
def rotate_y(angle):
    arr = np.zeros((3, 3))
    arr[:, 0] = [np.cos(angle), 0, -np.sin(angle)]
    arr[:, 1] = [0, 1, 0]
    arr[:, 2] = [np.sin(angle), 0, np.cos(angle)]
    return arr
def rotate_z(angle):
    arr = np.zeros((3, 3))
    arr[:, 0] = [np.cos(angle), np.sin(angle), 0]
    arr[:, 1] = [-np.sin(angle), np.cos(angle), 0]
    arr[:, 2] = [0, 0, 1]
    return arr

def kepler_solve(t, P, ecc):
    ''' Solver for Kepler's 2nd law giving the angle of an orbiter (rel. to origin) over time
    '''
    # follow the method in https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
    # to get true anomaly
    M = 2 * np.pi / P * t
    E = np.zeros(len(t))
    
    max_iter = 50
    tol = 1e-8
    
    for i in range(len(t)):
        E0 = M[i]
        
        # Newton's formula to solve for eccentric anomoly
        for j in range(max_iter):
            E1 = E0 - (E0 - ecc * np.sin(E0) - M[i]) / (1 - ecc * np.cos(E0))
            if abs(E1 - E0) < tol:
                break
            E0 = E1
        
        if j == max_iter:
            print('Did not converge')
        
        E[i] = E1
    
    # now output true anomaly (rad)
    return E, 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(E / 2), np.sqrt(1 - ecc) * np.cos(E / 2))


def dust_plume(a1, a2, windspeed1, windspeed2, period, ecc, incl, asc_node, arg_periastron, 
               turn_off, turn_on, cone_angle, distance, phase, n_orbits):
    '''
    Parameters
    ----------
    period : float
        seconds
    distance : float
        pc
    '''
    phase = phase%1
    n_t = 1000       # circles per orbital period
    n_points = 400   # points per circle
    n_particles = int(n_points * n_t * n_orbits)
    n_time = int(n_t * n_orbits)
    
    open_angle = np.deg2rad(cone_angle) / 2
    
    theta = 2 * np.pi * np.linspace(phase, n_orbits + phase, n_points)
    
    times = period * np.linspace(phase, n_orbits + phase, n_time)
    
    turn_on_rad = np.deg2rad(turn_on)
    turn_off_rad = np.deg2rad(turn_off)
    
    E, true_anomaly = kepler_solve(times, period, ecc)
    true_anomaly = true_anomaly
    
    r1 = a1 * (1 - ecc * np.cos(E))
    r2 = a2 * (1 - ecc * np.cos(E))
    ws_ratio = windspeed1 / windspeed2
    
    positions1 = np.zeros((3, n_time))
    positions1[0, :] = np.cos(true_anomaly)
    positions1[1, :] = np.sin(true_anomaly)
    positions2 = np.copy(positions1)
    positions1 *= -r1      # position in the orbital frame
    positions2 *=  r2     # position in the orbital frame
    
    widths = windspeed1 * period * (n_orbits - np.arange(n_time)/n_t)
    
    plume_direction = positions1 - positions2               # get the line of sight from first star to the second in the orbital frame
    # plume_direction /= np.linalg.norm(plume_direction)      # normalise it so that we only get the direction
    
    particles = np.zeros((3, n_particles))
    for i, nu in enumerate(true_anomaly):
        x = nu / (2 * np.pi)
        transf_nu = 2 * np.pi * (x + np.floor(0.5 - x))
        # transf_nu = nu - 2 * np.pi
        if transf_nu > turn_on_rad:
            if transf_nu < turn_off_rad:
                left = i * n_points
                right = (i + 1) * n_points
                # width = times[(n_time - 1) - i] * windspeed1
                # width = times[(n_time - 1) - i]%(2 * period) * windspeed1
                # width = (times[(n_time - 1) - i] / (n_orbits+1))%period * windspeed1
                
                
                # width = windspeed1 * period * (n_orbits - nu%(2 * np.pi))   # try this out!!
                
                direction = plume_direction[:, i] / np.linalg.norm(plume_direction[:, i])
                
                circle = np.array([np.ones(len(theta)) * np.cos(open_angle), np.sin(open_angle) * np.sin(theta), np.sin(open_angle) * np.cos(theta)])
                circle *= widths[i]
                angle_x = np.arctan2(direction[1], direction[0])
                circle = np.matmul(rotate_z(angle_x), circle)
                
                particles[:, left:right] = circle
    
    rotation = np.matmul(np.matmul(rotate_z(np.deg2rad(asc_node)), rotate_x(np.deg2rad(incl))), rotate_z(np.deg2rad(arg_periastron)))
    for i in range(n_particles):
        particles[:, i] = np.matmul(rotation, particles[:, i])
        
    return 60 * 60 * 180 / (2 * np.pi) * np.arctan(particles / (distance * 3.086e13))

def plot_spiral(particles):
    '''
    '''
    im_size = 256
    # im_res = 1
    _, n_points = particles.shape
    
    im = np.zeros((im_size, im_size))
    x = particles[0, :]
    y = particles[1, :]
    
    use_inds = np.where((x != 0) & (y != 0))
    x = x[use_inds]
    y = y[use_inds]

    H, xedges, yedges = np.histogram2d(x, y, bins=im_size)
    
    H = gaussian_filter(H, 1)
    
    fig, ax = plt.subplots()
    
    # ax.imshow(H, extent=[0, 1, 0, 1])
    ax.pcolormesh(xedges, yedges[::-1], H)
    ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')
    
def spiral_gif(a2, a1, windspeed1, windspeed2, period_s, eccentricity, inclination, 
                        asc_node, arg_periastron, turn_off, turn_on, cone_open_angle, distance):
    '''
    '''
    fig, ax = plt.subplots()
    
    
    im_size = 256
    im = np.zeros((im_size, im_size))
    n_orbits = 2 
    phase = 0 
    particles = dust_plume(a2, a1, windspeed1, windspeed2, period_s, eccentricity, inclination, 
                            asc_node, arg_periastron, turn_off, turn_on, cone_open_angle, distance, phase, n_orbits)
    x = particles[0, :]
    y = particles[1, :]
    use_inds = np.where((x != 0) & (y != 0))
    x = x[use_inds]
    y = y[use_inds]
    H, xbins, ybins = np.histogram2d(x, y, bins=im_size)
    H = gaussian_filter(H, 1)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    border = [[xmin, xmax], [ymin, ymax]]
    bins = [xbins, ybins]
    
    phase = 0.5 
    particles = dust_plume(a2, a1, windspeed1, windspeed2, period_s, eccentricity, inclination, 
                            asc_node, arg_periastron, turn_off, turn_on, cone_open_angle, distance, phase, n_orbits)
    x = particles[0, :]
    y = particles[1, :]
    use_inds = np.where((x != 0) & (y != 0))
    x = x[use_inds]
    y = y[use_inds]
    H, _, _ = np.histogram2d(x, y, bins=im_size)
    H = gaussian_filter(H, 1)
    vmin, vmax = np.min(H), np.max(H)
    
    every = 1
    length = 10
    # now calculate some parameters for the animation frames and timing
    nt = int(period_s / (60 * 60 * 24 * 365.25))    # roughly one year per frame
    # nt = 10
    frames = np.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
    fps = len(frames) // length  # fps for the final animation
    
    phases = np.linspace(0, 1, nt)
    ax.set(xlim=(min(xbins), max(xbins)), ylim=(min(ybins), max(ybins)), aspect='equal', 
           xlabel='Relative RA (")', ylabel='Relative Dec (")')
    def animate(i):
        if (i // every)%20 == 0:
            print(f"{i // every} / {len(frames)}")
        # print(i)
        particles = dust_plume(a2, a1, windspeed1, windspeed2, period_s, eccentricity, inclination, 
                               asc_node, arg_periastron, turn_off, turn_on, cone_open_angle, distance, phases[i] + 0.5, n_orbits)
        
        x = particles[0, :]
        y = particles[1, :]
        
        use_inds = np.where((x != 0) & (y != 0))
        x = x[use_inds]
        y = y[use_inds]

        H, xedges, yedges = np.histogram2d(x, y, bins=bins)
        H = gaussian_filter(H, 1)
        
        # ax.imshow(H, extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax, cmap='Greys')
        # ax.pcolormesh(xedges, yedges[::-1], H, vmax=vmax)
        ax.pcolormesh(xedges, yedges[::-1], H, vmax=vmax)
        
        return fig, 

    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
    ani.save(f"animation.gif", writer='pillow', fps=fps)
    
    
    

M_odot = 1.98e30
G = 6.67e-11
c = 299792458
yr2day = 365.25
kms2pcyr = 60*60*24*yr2day / (3.086e13) # km/s to pc/yr

m1 = 6                  # solar masses
m2 = 7                  # solar masses
eccentricity = 0.7
inclination = 25        # degrees
asc_node = -88          # degrees
arg_periastron = 0      # degrees
cone_open_angle = 125   # degrees (full opening angle)
period = 125            # years
period_s = period * yr2day * 24 * 60 * 60
distance = 2400         # pc
windspeed1 = 910       # km/s
windspeed2 = 2400       # km/s
turn_on = -114          # true anomaly (degrees)
turn_off = 150          # true anomaly (degrees)

m1, m2 = m1 * M_odot, m2 * M_odot
M = m1 + m2
mu = G * M
kms2masyr = np.arctan(kms2pcyr / distance) * 180/np.pi * 60 * 60 * 1000     # conversion from km/s to mas/yr at the system distance
ws1 = windspeed1 * kms2masyr
ws2 = windspeed2 * kms2masyr
a = np.cbrt((period_s / (2 * np.pi))**2 * mu)     # semi-major axis of the system (total separation)
a1 = m2 / M * a                                 # semi-major axis of first body
a2 = a - a1                                     # semi-major axis of second body

p1 = a1 * (1 - eccentricity**2)
p2 = a2 * (1 - eccentricity**2)

### plots orbits
# theta = np.linspace(0, 2 * np.pi, 100)
# r1 = p1 / (1 + eccentricity * np.cos(theta))
# r2 = p2 / (1 + eccentricity * np.cos(theta))

# x1, y1 = r1 * np.cos(theta), r1 * np.sin(theta)
# x2, y2 = -r2 * np.cos(theta), -r2 * np.sin(theta)

# fig, ax = plt.subplots()

# ax.plot(x1, y1)
# ax.plot(x2, y2)
# ax.set_aspect('equal')

n_orbits = 2 
phase = 0.5


particles = dust_plume(a2, a1, windspeed1, windspeed2, period_s, eccentricity, inclination, 
                        asc_node, arg_periastron, turn_off, turn_on, cone_open_angle, distance, phase, n_orbits)


plot_spiral(particles)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(particles[0, :], particles[1, :], particles[2, :])

# spiral_gif(a2, a1, windspeed1, windspeed2, period_s, eccentricity, inclination, 
#                         asc_node, arg_periastron, turn_off, turn_on, cone_open_angle, distance)

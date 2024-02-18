# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 08:36:43 2024

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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
               turn_off, turn_on, cone_angle, phase, n_orbits, n_t):
    '''
    Parameters
    ----------
    period : float
        seconds
    '''
    n_points = 500  # points per circle
    n_particles = int(n_points * n_t * n_orbits)
    n_time = int(n_t * n_orbits)
    
    open_angle = np.deg2rad(cone_angle) / 2
    
    theta = 2 * np.pi * np.linspace(phase, n_orbits + phase, n_points)
    
    times = period * np.linspace(phase, n_orbits + phase, n_time)
    
    turn_on_rad = np.deg2rad(turn_on) + np.pi
    turn_off_rad = np.deg2rad(turn_off) + np.pi
    
    E, true_anomaly = kepler_solve(times, period, ecc)
    true_anomaly = np.abs(true_anomaly)
    
    r1 = a1 * (1 - ecc * np.cos(E))
    r2 = a2 * (1 - ecc * np.cos(E))
    ws_ratio = windspeed1 / windspeed2
    
    positions1 = np.zeros((3, n_time))
    positions1[0, :] = np.cos(true_anomaly)
    positions1[1, :] = np.sin(true_anomaly)
    positions2 = np.copy(positions1)
    positions1 *= r1      # position in the orbital frame
    positions2 *= - r2     # position in the orbital frame
    
    
    plume_direction = positions1 - positions2               # get the line of sight from first star to the second in the orbital frame
    # plume_direction /= np.linalg.norm(plume_direction)      # normalise it so that we only get the direction
    
    particles = np.zeros((3, n_particles))
    # print(true_anomaly)
    for i, nu in enumerate(true_anomaly):
        if (2 * np.pi - turn_on_rad) > nu > turn_on_rad:
            if (2 * np.pi - turn_off_rad) < nu < turn_off_rad:
                left = i * n_points
                right = (i + 1) * n_points
                
                direction = plume_direction[:, i] / np.linalg.norm(plume_direction[:, i])
                shock_cone = np.ones((3, n_points))
                for j in range(3):
                    shock_cone[j, :] = direction[j]
                # circle = shock_cone + np.array([np.cos(theta), np.sin(theta), np.zeros(len(theta))]) * np.tan(cone_angle / 2)
                # circle = shock_cone + np.array([np.cos(theta), np.sin(theta), np.ones(len(theta))]) * np.tan(cone_angle / 2)
                # circle = shock_cone + np.array([np.ones(len(theta)), np.sin(theta), np.cos(theta)]) * np.tan(cone_angle / 2)
                
                # circle = shock_cone + np.array([np.ones(len(theta)), np.sin(cone_angle / 2) * np.sin(theta), np.sin(cone_angle / 2) * np.cos(theta)])
                
                circle = np.array([np.ones(len(theta)), np.sin(open_angle) * np.sin(theta), np.sin(open_angle) * np.cos(theta)])
                angle_x = np.arctan2(direction[1], direction[0])
                circle = np.matmul(rotate_z(angle_x), circle)
                # for j in range(3):
                #     circle[j, :] += plume_direction[j, i] * (1 - ws_ratio)
                
                particles[:, left:right] = times[(n_time - 1) - i] * windspeed1 * circle
    
    rotation = np.matmul(np.matmul(rotate_z(-np.deg2rad(asc_node)), rotate_x(-np.deg2rad(incl))), rotate_z(-np.deg2rad(arg_periastron)))
    for i in range(n_particles):
        particles[:, i] = np.matmul(rotation, particles[:, i])
        
    return particles / np.max(particles)

def plot_spiral(particles):
    im_size = 256
    # im_res = 1
    _, n_points = particles.shape
    
    im = np.zeros((im_size, im_size))
    x = particles[0, :]
    y = particles[1, :]
    
    use_inds = np.where((x != 0) & (y != 0))
    x = x[use_inds]
    y = y[use_inds]
    
    # xs = np.linspace(min(x), max(x), len(x))
    # ys = np.linspace(min(y), max(y), len(y))
    # for i in range(n_points):
        # if particles[0, i] == min(x):
        #     imx = 0 
        # elif particles[0, i] == max(x):
        #     imx = im_size - 1
        # else:
        #     imx = np.argwhere((xs <= particles[0, i]) & (xs + 1 > particles[0, i])).flatten()[0]
        
        # if particles[1, i] == min(y):
        #     imy = 0 
        # elif particles[1, i] == max(y):
        #     imy = im_size - 1
        # else:
        #     imy = np.argwhere((ys <= particles[1, i]) & (ys + 1 > particles[1, i])).flatten()[0]
        
        # print(imx, imy)
        # im[imx, imy] += 1
    H, _, _ = np.histogram2d(x, y, bins=im_size)
    
    H = gaussian_filter(H, 1)
    
    fig, ax = plt.subplots()
    
    ax.imshow(H, extent=[0, 1, 0, 1])

M_odot = 1.98e30
G = 6.67e-11
c = 299792458
yr2day = 365.25
kms2pcyr = 60*60*24*yr2day / (3.086e13) # km/s to pc/yr

m1 = 6                 # solar masses
m2 = 7                 # solar masses
eccentricity = 0.7
inclination = 25       # degrees
asc_node = -88          # degrees
arg_periastron = 0      # degrees
cone_open_angle = 125   # degrees (full opening angle)
period = 125            # years
period_s = period * yr2day * 24 * 60 * 60
distance = 2400         # pc
windspeed1 = 910       # km/s
windspeed2 = 1       # km/s
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
phase = 0.8
n_t = 200

particles = dust_plume(a2, a1, windspeed1, windspeed2, period_s, eccentricity, inclination, 
                       asc_node, arg_periastron, turn_off, turn_on, cone_open_angle, phase, n_orbits, n_t)


plot_spiral(particles)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(particles[0, :], particles[1, :], particles[2, :])

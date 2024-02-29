# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 08:36:43 2024

@author: ryanw
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import jax.lax as lax
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from matplotlib import animation
import time

@jit
def rotate_x(angle):
    arr = jnp.array([[1, 0, 0],
                     [0, jnp.cos(angle), jnp.sin(angle)],
                     [0, -jnp.sin(angle), jnp.cos(angle)]])
    return arr
@jit
def rotate_y(angle):
    arr = jnp.array([[jnp.cos(angle), 0, -jnp.sin(angle)],
                     [0, 1, 0],
                     [jnp.sin(angle), 0, jnp.cos(angle)]])
    return arr
@jit
def rotate_z(angle):
    arr = jnp.array([[jnp.cos(angle), jnp.sin(angle), 0],
                     [-jnp.sin(angle), jnp.cos(angle), 0],
                     [0, 0, 1]])
    return arr


@jit
def kepler_solve_sub_sub(i, E0_ecc_mi):
    E0, ecc, mi = E0_ecc_mi
    return (E0 - (E0 - ecc * jnp.sin(E0) - mi) / (1 - ecc * jnp.cos(E0)), ecc, mi)


@jit
def kepler_solve_sub(i, ecc, tol, M):
    E0 = M[i]
    # Newton's formula to solve for eccentric anomaly
    E0 = lax.fori_loop(0, 20, kepler_solve_sub_sub, (E0, ecc, M[i]))[0]
    return E0


@jit
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


@jit 
def dust_plume_sub(i_nu, turn_on_rad, turn_off_rad, theta, open_angle, plume_direction, widths, n_points):
    i, nu = i_nu
    x = nu / (2 * jnp.pi)
    transf_nu = 2 * jnp.pi * (x + jnp.floor(0.5 - x))
    turned_on = jnp.heaviside(transf_nu - turn_on_rad, 0)
    turned_off = jnp.heaviside(turn_off_rad - transf_nu, 0)
    direction = plume_direction[:, i] / jnp.linalg.norm(plume_direction[:, i])
    # print(direction)
    circle = jnp.array([jnp.ones(len(theta)) * jnp.cos(open_angle), 
                        jnp.sin(open_angle) * jnp.sin(theta), 
                        jnp.sin(open_angle) * jnp.cos(theta)])
    circle *= widths[i]
    angle_x = jnp.arctan2(direction[1], direction[0])
    circle = rotate_z(angle_x) @ circle
    
    circle *= turned_on * turned_off
    return circle


# @jit
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
    n_particles = jnp.round(n_points * n_t * n_orbits)
    n_time = jnp.round(n_t * n_orbits)
    
    open_angle = jnp.deg2rad(cone_angle) / 2
    
    theta = 2 * jnp.pi * jnp.linspace(0, 1, n_points)
    
    times = period * jnp.linspace(phase, n_orbits + phase, n_time)
    
    turn_on_rad = jnp.deg2rad(turn_on)
    turn_off_rad = jnp.deg2rad(turn_off)
    
    E, true_anomaly = kepler_solve(times, period, ecc)
    
    r1 = a1 * (1 - ecc * jnp.cos(E)) * 1e-3     # radius in km 
    r2 = a2 * (1 - ecc * jnp.cos(E)) * 1e-3
    ws_ratio = windspeed1 / windspeed2
    
    
    positions1 = jnp.array([jnp.cos(true_anomaly), 
                            jnp.sin(true_anomaly), 
                            jnp.zeros(n_time)])
    positions2 = jnp.copy(positions1)
    positions1 *= -r1      # position in the orbital frame
    positions2 *=  r2     # position in the orbital frame
    
    widths = windspeed1 * period * (n_orbits - jnp.arange(n_time) / n_t)
    
    plume_direction = positions1 - positions2               # get the line of sight from first star to the second in the orbital frame
    
        
    particles = vmap(lambda i_nu: dust_plume_sub(i_nu, turn_on_rad, turn_off_rad, theta, open_angle, 
                                                  plume_direction, widths, n_points))((jnp.arange(n_time), true_anomaly))
    
    particles = jnp.array([jnp.ravel(particles[:, 0, :]),
                           jnp.ravel(particles[:, 1, :]),
                           jnp.ravel(particles[:, 2, :])])

    particles = rotate_z(jnp.deg2rad(-asc_node)) @ (
            rotate_x(jnp.deg2rad(-incl)) @ (
            rotate_z(jnp.deg2rad(-arg_periastron)) @ particles))

    return 60 * 60 * 180 / jnp.pi * jnp.arctan(particles / (distance * 3.086e13))

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
    
    # ii = np.arange(int(0.5*len(x)), len(x))
    # x = x[ii]
    # y = y[ii]
    

    H, xedges, yedges = np.histogram2d(y, x, bins=im_size)
    
    # H = np.maximum(H, 4)
    H = gaussian_filter(H, 2)
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    
    fig, ax = plt.subplots()
    
    # ax.pcolormesh(X, Y, H)
    import matplotlib.colors as cols
    # ax.pcolormesh(X, Y, H, norm=cols.LogNorm(vmin=1, vmax=H.max()))
    ax.pcolormesh(X, Y, H, norm=cols.PowerNorm(gamma=1/2))
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
        ax.pcolormesh(xedges[::-1], yedges[::-1], H, vmax=vmax)
        
        return fig, 

    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
    ani.save(f"animation.gif", writer='pillow', fps=fps)
    
    
    

M_odot = 1.98e30
G = 6.67e-11
c = 299792458
yr2day = 365.25
kms2pcyr = 60*60*24*yr2day / (3.086e13) # km/s to pc/yr

# below are rough params for Apep 
m1 = 15                  # solar masses
m2 = 10                  # solar masses
eccentricity = 0.7
inclination = 25        # degrees
asc_node = -88          # degrees
arg_periastron = 0      # degrees
cone_open_angle = 125   # degrees (full opening angle)
period = 125            # years
period_s = period * yr2day * 24 * 60 * 60
distance = 2400         # pc
windspeed1 = 700       # km/s
windspeed2 = 2400       # km/s
turn_on = -114          # true anomaly (degrees)
turn_off = 150          # true anomaly (degrees)

# # below are rough params for WR 112
# m1 = 15                  # solar masses
# m2 = 10                  # solar masses
# eccentricity = 0.
# inclination = 100        # degrees
# asc_node = 75          # degrees
# arg_periastron = 170      # degrees
# cone_open_angle = 110   # degrees (full opening angle)
# period = 19            # years
# period_s = period * yr2day * 24 * 60 * 60
# distance = 2400         # pc
# windspeed1 = 700       # km/s
# windspeed2 = 2400       # km/s
# turn_on = -180          # true anomaly (degrees)
# turn_off = 180          # true anomaly (degrees)

# # below are rough params for WR 140
# m1 = 8.4                  # solar masses
# m2 = 20                  # solar masses
# eccentricity = 0.9
# inclination = 119        # degrees
# asc_node = 349          # degrees
# arg_periastron = 42.3      # degrees
# cone_open_angle = 80   # degrees (full opening angle)
# period = 7.9            # years
# period_s = period * yr2day * 24 * 60 * 60
# distance = 5600         # pc
# windspeed1 = 2600       # km/s
# windspeed2 = 2400       # km/s
# turn_on = -135          # true anomaly (degrees)
# turn_off = 135          # true anomaly (degrees)


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

t1 = time.time()
particles = dust_plume(a2, a1, windspeed1, windspeed2, period_s, eccentricity, inclination, 
                        asc_node, arg_periastron, turn_off, turn_on, cone_open_angle, distance, phase, n_orbits)
print(time.time() - t1)

plot_spiral(particles)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(particles[0, :], particles[1, :], particles[2, :])

# spiral_gif(a2, a1, windspeed1, windspeed2, period_s, eccentricity, inclination, 
#                         asc_node, arg_periastron, turn_off, turn_on, cone_open_angle, distance)

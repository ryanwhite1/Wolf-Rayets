# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:53:10 2024

@author: ryanw
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, grad
import jax
import jax.lax as lax
import jax.scipy.stats as stats
from jax.interpreters import ad
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import jax.scipy.signal as signal
from matplotlib import animation
import time
import emcee
import pickle
import jaxopt

import WR_Geom_Model as gm
import WR_binaries as wrb

# we need 64 bit math for the semi-major axis calculations
jax.config.update("jax_enable_x64", True)

# define constants
M_odot = 1.98e30                        # solar mass in kg
G = 6.67e-11                            # grav constant in SI
c = 299792458                           # speed of light m/s
yr2day = 365.25                         # num days in a year
yr2s = yr2day * 24*60*60                # num seconds in a year
kms2pcyr = 60*60*24*yr2day / (3.086e13) # km/s to pc/yr
AU2km = 1.496e8                         # km in an AU
h = 6.626e-34
kb = 1.38e-23
b = 2.89777e-3

def weins_temp(wavelength):
    return b / wavelength
def blackbody(x, T):
    return (2 * h * c**2 / (x**5)) * 1 / (np.exp(h * c / (x * kb * T)) - 1)

# with open('particles.pickle', 'rb') as input_file:
#     shell_particles = pickle.load(input_file)

# with open('weights.pickle', 'rb') as input_file:
#     shell_weights = pickle.load(input_file)

# shell_len = len(shell_weights)
    
apep = wrb.apep.copy()
# apep['histmax'] = 0.002
shells = 3


particles, weights = gm.gui_funcs[shells - 1](apep)

shell_len = int(len(weights) / shells)

particles = np.array(particles)
weights = np.array(weights)

temperatures = np.array([500, 200, 80])               # estimated temperature of each shell (first, second, third)
jwst_temps = np.array([376.33, 193.1847, 113.638])    # wiens displacement on 7.7um, 15um, and 25.5um
jwst_lambdas = np.array([7.7e-6, 15e-6, 25.5e-6])

blackbodies = blackbody(jwst_lambdas[2], temperatures)
blackbodies /= max(blackbodies)

# particles = np.empty((3, 0))
# weights = np.empty((0))

fig, ax = plt.subplots()
ax.plot(weights)

for i in range(shells):
    # new_particles = shell_particles.copy() * (i + 1)
    # particles = np.append(particles, new_particles, axis=1)
    
    # new_weights = shell_weights.copy() * blackbodies[i]
    # weights = np.append(weights, new_weights)
    
    weights[i * shell_len:(i+1) * shell_len] *= blackbodies[shells - i - 1]
    print(i * shell_len,(i+1) * shell_len)

weights = weights**0.05

ax.plot(weights)

X, Y, H = gm.smooth_histogram2d(particles, weights, apep)
gm.plot_spiral(X, Y, H)
    


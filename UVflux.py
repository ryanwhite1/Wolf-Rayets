# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:30:59 2024

@author: ryanw
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

h = 6.626e-34
c = 299792458
k = 1.38e-23
T = 30000
R_odot = 6.696e8 
R_star = 30 * R_odot 
AU_to_m = 150e9
pc_to_m = 3.084e16
distance = 1700 * AU_to_m

c1 = 2 * np.pi * h * c**2
c2 = h * c / (k * T)

# planck_log = lambda x: (c1 / 10**(x*5)) * (1 / (np.exp(c2 / 10**x) - 1))
planck = lambda x: (c1 / x**5) * (1 / (np.exp(c2 / x) - 1))

# integral = quad(planck, 0, 3e-7)
# # integral2 = quad(planck, 0, 3e-6)

# val = integral[0]

# luminosity = val * 4 * np.pi * R_star**2

# flux = luminosity / (4 * np.pi * distance**2)



# model_spectrum = np.loadtxt("ob-i_29-30_sed.txt")
model_spectrum = np.loadtxt("ob-i_28-30_sed.txt")

wavelengths = 10**model_spectrum[:, 0]

cutoff = wavelengths <= 2400

wavelengths = wavelengths[cutoff]

fluxes = 10**model_spectrum[:, 1]

fluxes = fluxes[cutoff] * ((10 * pc_to_m) / distance)**2 # calibrate to distance

fluxes = fluxes / 1e7 * 1e4 # calibrate to W/m^2/A from erg/cm^2/s/A

# now manually calibrate the brightness down a bit to compensate for the model star being significantly brighter

# fluxes *= 10**(5.9 - 6.07)

def trap(x1, x2, y1, y2):
    wid = x2 - x1
    return 0.5 * wid * (y1 + y2)

UVFlux = 0

for i in range(len(fluxes) - 1):
    UVFlux += trap(wavelengths[i], wavelengths[i + 1], fluxes[i], fluxes[i+1])
    
fig, ax = plt.subplots()

ax.plot(wavelengths, fluxes)

ax.set(xlabel='wavelength (A)', ylabel='specific flux (W/m^2/A) @ 1700 AU')

blackbody = planck(wavelengths * 1e-10)
blackbody /= 4 * np.pi * distance**2 
blackbody *= 1e-10 * 4 * np.pi * R_star**2

ax.plot(wavelengths, blackbody)





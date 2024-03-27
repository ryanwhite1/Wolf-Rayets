# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:04:50 2024

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

### --- GUI Plot --- ###

import tkinter

import numpy as np

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

starcopy = wrb.apep.copy()
starcopy['n_orbits'] = 1

root = tkinter.Tk()
root.wm_title("Embedding in Tk")

fig, ax = plt.subplots()
particles, weights = gm.dust_plume(starcopy)
X, Y, H = gm.spiral_grid(particles, weights, starcopy)
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


def update_frequency(param, new_val):
    # retrieve frequency
    starcopy[param] = float(new_val)
    
    particles, weights = gm.gui_funcs[int(starcopy['n_orbits']) - 1](starcopy)
    
    X, Y, H = gm.spiral_grid(particles, weights, starcopy)
    mesh.update({'array':H.ravel()})
    
    new_coords = mesh._coordinates
    new_coords[:, :, 0] = X
    new_coords[:, :, 1] = Y
    mesh._coordinates = new_coords
    ax.set(xlim=(np.min(X), np.max(X)), ylim=(np.min(Y), np.max(Y)))

    # required to update canvas and attached toolbar!
    canvas.draw()


ecc = tkinter.Scale(root, from_=0, to=0.99, orient=tkinter.HORIZONTAL, 
                    command=lambda v: update_frequency('eccentricity', v), label="Eccentricity", resolution=0.01)
ecc.set(starcopy['eccentricity'])
inc = tkinter.Scale(root, from_=0, to=180, orient=tkinter.HORIZONTAL, 
                    command=lambda v: update_frequency('inclination', v), label="Inclination", resolution=0.1)
inc.set(starcopy['inclination'])
asc_node = tkinter.Scale(root, from_=0, to=360, orient=tkinter.HORIZONTAL,
                         command=lambda v: update_frequency('asc_node', v), label="Ascending Node", resolution=0.1)
asc_node.set(starcopy['asc_node'])
arg_peri = tkinter.Scale(root, from_=0, to=180, orient=tkinter.HORIZONTAL,
                         command=lambda v: update_frequency('arg_peri', v), label="Argument of Periastron", resolution=0.1)
arg_peri.set(starcopy['arg_peri'])
opang = tkinter.Scale(root, from_=0, to=180, orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('open_angle', v), label="Open Angle", resolution=0.1)
opang.set(starcopy['open_angle'])
m1 = tkinter.Scale(root, from_=5, to=50, orient=tkinter.HORIZONTAL,
                   command=lambda v: update_frequency('m1', v), label="WR Mass", resolution=0.1)
m1.set(starcopy['m1'])
m2 = tkinter.Scale(root, from_=5, to=50, orient=tkinter.HORIZONTAL, 
                   command=lambda v: update_frequency('m2', v), label="Companion Mass", resolution=0.1)
m2.set(starcopy['m2'])
phase = tkinter.Scale(root, from_=0.01, to=1.5, orient=tkinter.HORIZONTAL, 
                      command=lambda v: update_frequency('phase', v), label="Phase", resolution=0.01)
phase.set(starcopy['phase'])
n_orb = tkinter.Scale(root, from_=1, to=20, orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('n_orbits', v), label="Shells")
n_orb.set(starcopy['n_orbits'])
turnon = tkinter.Scale(root, from_=-180, to=0, orient=tkinter.HORIZONTAL,
                       command=lambda v: update_frequency('turn_on', v), label="Turn On Angle", resolution=0.1)
turnon.set(starcopy['turn_on'])
turnoff = tkinter.Scale(root, from_=0, to=180, orient=tkinter.HORIZONTAL,
                        command=lambda v: update_frequency('turn_off', v), label="Turn Off Angle", resolution=0.1)
turnoff.set(starcopy['turn_off'])
distance = tkinter.Scale(root, from_=1e3, to=1e4, orient=tkinter.HORIZONTAL,
                         command=lambda v: update_frequency('distance', v), label="Distance (pc)", resolution=1)
distance.set(starcopy['distance'])
ws1 = tkinter.Scale(root, from_=0, to=5e3, orient=tkinter.HORIZONTAL,
                    command=lambda v: update_frequency('windspeed1', v), label="WR Windspeed (km/s)", resolution=1)
ws1.set(starcopy['windspeed1'])
ws2 = tkinter.Scale(root, from_=0, to=5e3, orient=tkinter.HORIZONTAL,
                    command=lambda v: update_frequency('windspeed2', v), label="Companion Windspeed (km/s)", resolution=1)
ws2.set(starcopy['windspeed2'])
period = tkinter.Scale(root, from_=0, to=200, orient=tkinter.HORIZONTAL,
                       command=lambda v: update_frequency('period', v), label="Period (yr)", resolution=0.1)
period.set(starcopy['period'])
osd = tkinter.Scale(root, from_=0, to=180, orient=tkinter.HORIZONTAL,
                    command=lambda v: update_frequency('orb_sd', v), label="Orb. Var. SD (deg)", resolution=0.1)
osd.set(starcopy['orb_sd'])
oamp = tkinter.Scale(root, from_=-1, to=1, orient=tkinter.HORIZONTAL,
                     command=lambda v: update_frequency('orb_amp', v), label="Orb. Var. Amp", resolution=0.01)
oamp.set(starcopy['orb_amp'])
orbmin = tkinter.Scale(root, from_=0, to=360, orient=tkinter.HORIZONTAL,
                     command=lambda v: update_frequency('orb_min', v), label="Orb. Minimum (deg)", resolution=0.1)
orbmin.set(starcopy['orb_min'])
azsd = tkinter.Scale(root, from_=0, to=180, orient=tkinter.HORIZONTAL,
                     command=lambda v: update_frequency('az_sd', v), label="Az. Var. SD (deg)", resolution=0.1)
azsd.set(starcopy['az_sd'])
azamp = tkinter.Scale(root, from_=-1, to=1, orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('az_amp', v), label="Az. Var. Amp", resolution=0.01)
azamp.set(starcopy['az_amp'])
azmin = tkinter.Scale(root, from_=0, to=360, orient=tkinter.HORIZONTAL,
                     command=lambda v: update_frequency('az_min', v), label="Az. Minimum (deg)", resolution=0.1)
azmin.set(starcopy['az_min'])
compincl = tkinter.Scale(root, from_=0, to=180, orient=tkinter.HORIZONTAL,
                     command=lambda v: update_frequency('comp_incl', v), label="Companion Incl. (deg)", resolution=0.1)
compincl.set(starcopy['comp_incl'])
compaz = tkinter.Scale(root, from_=0, to=360, orient=tkinter.HORIZONTAL,
                     command=lambda v: update_frequency('comp_az', v), label="Companion Azimuth. (deg)", resolution=0.1)
compaz.set(starcopy['comp_az'])
compopen = tkinter.Scale(root, from_=0, to=180, orient=tkinter.HORIZONTAL,
                     command=lambda v: update_frequency('comp_open', v), label="Companion Open Angle (deg)", resolution=0.1)
compopen.set(starcopy['comp_open'])
compreduc = tkinter.Scale(root, from_=0, to=2, orient=tkinter.HORIZONTAL,
                     command=lambda v: update_frequency('comp_reduction', v), label="Companion Photodissociation", resolution=0.01)
compreduc.set(starcopy['comp_reduction'])
compplume = tkinter.Scale(root, from_=0, to=2, orient=tkinter.HORIZONTAL,
                     command=lambda v: update_frequency('comp_plume', v), label="Companion Plume", resolution=0.01)
compplume.set(starcopy['comp_reduction'])
histmax = tkinter.Scale(root, from_=1, to=0, orient=tkinter.HORIZONTAL,
                        command=lambda v: update_frequency('histmax', v), label="Max Brightness", resolution=0.01)
histmax.set(starcopy['histmax'])
sigma = tkinter.Scale(root, from_=0.01, to=10, orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('sigma', v), label="Gaussian Blur", resolution=0.01)
sigma.set(starcopy['sigma'])
oblate = tkinter.Scale(root, from_=0., to=1, orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('oblate', v), label="Plume Oblateness", resolution=0.01)
oblate.set(starcopy['oblate'])

sliders = [ecc, inc, asc_node, arg_peri, phase, opang, m1, m2, 
           turnon, turnoff, distance, n_orb, ws1, ws2, 
           osd, orbmin, oamp, sigma, period, oblate,
           azsd, azmin, azamp, histmax, compopen, compplume,
           compreduc, compincl, compaz]

num_in_row = 6
toolbar.grid(row=0, columnspan=num_in_row)
canvas.get_tk_widget().grid(row=1, column=0, columnspan=num_in_row)
for i, slider in enumerate(sliders):
    row = int(np.floor(i / num_in_row)) + 2
    col = i%(num_in_row)
    slider.grid(row=row, column=col)

button_quit.grid(row=row+1, column=1)



tkinter.mainloop()
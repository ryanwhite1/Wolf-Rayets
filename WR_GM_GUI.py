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

apep = wrb.apep.copy()

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
particles, weights = gm.dust_plume(apep)
X, Y, H = gm.spiral_grid(particles, weights, apep)
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
    particles, weights = gm.dust_plume(starcopy)
    X, Y, H = gm.spiral_grid(particles, weights, starcopy)
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
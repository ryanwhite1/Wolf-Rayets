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
from astropy.io import fits
from glob import glob
import os

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
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

starcopy = wrb.apep_aniso2.copy()
starcopy['n_orbits'] = 1

# n = 256     # standard
# n = 600     # VISIR
n = 898     # JWST
@jit
def smooth_histogram2d(particles, weights, stardata):
    im_size = n
    
    x = particles[0, :]
    y = particles[1, :]
    
    xbound, ybound = jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))
    bound = jnp.max(jnp.array([xbound, ybound])) * (1. + 2. / im_size)
    
    xedges, yedges = jnp.linspace(-bound, bound, im_size+1), jnp.linspace(-bound, bound, im_size+1)
    return gm.smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
@jit
def smooth_histogram2d_w_bins(particles, weights, stardata, xbins, ybins):
    im_size = n
    return gm.smooth_histogram2d_base(particles, weights, stardata, xbins, ybins, im_size)


def standard_sim_reference():
    particles, weights = gm.dust_plume(starcopy)
    X, Y, H_original = smooth_histogram2d(particles, weights, starcopy)
    H_original = gm.add_stars(X[0, :], Y[:, 0], H_original, starcopy)
    
    return X, Y, H_original

def Apep_VISIR_reference():
    pscale = 1000 * 23/512 # mas/pixel, (Yinuo's email said 45mas/px, but I think the FOV is 23x23 arcsec for a 512x512 image?)
    
    directory = "Data\\VLT"
    fnames = glob(directory + "\\*.fits")
    
    vlt_data = fits.open(fnames[-1])    # for the 2024 epoch
    
    data = vlt_data[0].data
    length = data.shape[0]
    
    X = jnp.linspace(-1., 1., length) * pscale * length/2 / 1000
    Y = X.copy()
    
    xs, ys = jnp.meshgrid(X, Y)
    
    data = jnp.array(data)
    # data = data - jnp.median(data)
    data = data - jnp.percentile(data, 84)
    data = data/jnp.max(data)
    data = jnp.maximum(data, 0)
    data = jnp.abs(data)**0.5
    
    
    return xs, ys, data

def Apep_JWST_reference():
    
    directory = "Data\\JWST\\MAST_2024-07-29T2157\\JWST"
    fname = glob(directory+"\\jw05842-o001_t001_miri_f2550w\\*_i2d.fits")[0]
    
    jwst_center_x = 565
    jwst_center_y = 755
    
    jwst_data = fits.open(fname)    # for the 2024 epoch
    
    data = jwst_data[1].data.T[:, ::-1]
    pscale = np.sqrt(jwst_data[1].header['PIXAR_A2']) * 1000
    im_size = data.shape[0] - jwst_center_y
    data = data[(jwst_center_y - im_size):, (jwst_center_x - im_size):(jwst_center_x + im_size)]
    length = data.shape[0]
    
    X = jnp.linspace(-1., 1., length) * pscale * length/2 / 1000
    Y = X.copy()
    
    xs, ys = jnp.meshgrid(X, Y)
    
    data = jnp.array(data)
    # data = data - jnp.median(data)
    data = data - jnp.percentile(data, 60)
    data = data/jnp.max(data)
    data = jnp.maximum(data, 0)
    data = jnp.abs(data)**0.5 
    
    return xs, ys, data
    

# X_ref, Y_ref, H_ref = Apep_VISIR_reference()
# fig, ax = plt.subplots()
# ax.plot(np.arange(H_ref.shape[0]), H_ref[:, 300])

# fig, ax = plt.subplots()
# ax.imshow(H_ref)
# ax.invert_yaxis()

# X_ref, Y_ref, H_ref = Apep_JWST_reference()
# fig, ax = plt.subplots()
# ax.plot(np.arange(H_ref.shape[0]), H_ref[:, 300])

# fig, ax = plt.subplots()
# ax.imshow(H_ref)
# ax.invert_yaxis()


root = tkinter.Tk()
root.wm_title("Embedding in Tk")

titles = ['Model', 'Reference', 'Difference']
w = 1/3.08
fig, axes = plt.subplots(figsize=(12, 4), ncols=3, gridspec_kw={'wspace':0, 'width_ratios':[w, w, 1-2*w]})

# X_ref, Y_ref, H_ref = standard_sim_reference()
# X_ref, Y_ref, H_ref = Apep_VISIR_reference()
X_ref, Y_ref, H_ref = Apep_JWST_reference()
reference_mesh = axes[1].pcolormesh(X_ref, Y_ref, H_ref, cmap='hot')
maxside2 = np.max(np.abs(np.array([X_ref, Y_ref])))
axes[1].set(xlim=(-maxside2, maxside2), ylim=(-maxside2, maxside2))


X, Y, H_original = standard_sim_reference()
mesh = axes[0].pcolormesh(X, Y, H_original, cmap='hot')
axes[0].set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")', title=titles[0])

# if n == 898:    # i.e. if we're using the jwst image
    

H_ref_ravel = H_ref.ravel()
norm = colors.Normalize(vmin=-1., vmax=1.)
diff_mesh = axes[2].pcolormesh(X_ref, Y_ref, H_original - H_ref, cmap='seismic', norm=norm)
for i in range(1, 3):
    axes[i].set(aspect='equal', xlabel='Relative RA (")', title=titles[i])
    axes[i].tick_params(axis='y',
                        which='both',
                        left=False,
                        labelleft=False)
the_divider = make_axes_locatable(axes[2])
color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(diff_mesh, cax=color_axis)

for i in range(2):
    axes[i].set_facecolor('k')

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()

# pack_toolbar=False will make it easier to use a layout manager later on.
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()

canvas.mpl_connect(
    "key_press_event", lambda event: print(f"you pressed {event.key}"))
canvas.mpl_connect("key_press_event", key_press_handler)

button_quit = tkinter.Button(master=root, text="Quit", command=root.destroy)


def update_frequency(param, new_val, X=X, Y=Y):
    starcopy[param] = float(new_val)
    
    particles, weights = gm.gui_funcs[int(starcopy['n_orbits']) - 1](starcopy)
    
    X_new, Y_new, H = smooth_histogram2d(particles, weights, starcopy)
    H = gm.add_stars(X_new[0, :], Y_new[:, 0], H, starcopy)
    new_H = H.ravel()
    mesh.update({'array':new_H})
    
    _, _, H_diff = smooth_histogram2d_w_bins(particles, weights, starcopy, X_ref[0, :], Y_ref[:, 0])
    H_diff = gm.add_stars(X[0, :], Y[:, 0], H_diff, starcopy)
    H_diff = H_diff.ravel()
    
    diff_mesh.update({'array':H_diff - H_ref_ravel})
    
    new_coords = mesh._coordinates
    new_coords[:, :, 0] = X_new
    new_coords[:, :, 1] = Y_new
    mesh._coordinates = new_coords
    
    maxside1 = np.max(np.abs(np.array([X_new, Y_new])))
    axes[0].set(xlim=(-maxside1, maxside1), ylim=(-maxside1, maxside1))
    diff_maxside = np.max([maxside1, np.max(np.abs(np.array([X, Y])))])
    axes[2].set(xlim=(-diff_maxside, diff_maxside), ylim=(-diff_maxside, diff_maxside))

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
period = tkinter.Scale(root, from_=0, to=300, orient=tkinter.HORIZONTAL,
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
compplume.set(starcopy['comp_plume'])
histmax = tkinter.Scale(root, from_=1, to=0, orient=tkinter.HORIZONTAL,
                        command=lambda v: update_frequency('histmax', v), label="Max Brightness", resolution=0.01)
histmax.set(starcopy['histmax'])
sigma = tkinter.Scale(root, from_=0.01, to=10, orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('sigma', v), label="Gaussian Blur", resolution=0.01)
sigma.set(starcopy['sigma'])
oblate = tkinter.Scale(root, from_=0., to=1, orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('oblate', v), label="Plume Oblateness", resolution=0.01)
oblate.set(starcopy['oblate'])
nuc_dist = tkinter.Scale(root, from_=0.1, to=1e3, orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('nuc_dist', v), label="Nuc. Dist", resolution=0.01)
nuc_dist.set(starcopy['nuc_dist'])
opt_thin_dist = tkinter.Scale(root, from_=0.2, to=1e3, orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('opt_thin_dist', v), label="Opt. Thin Dist.", resolution=0.01)
opt_thin_dist.set(starcopy['opt_thin_dist'])
acc_max = tkinter.Scale(root, from_=0.1, to=4e3, orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('acc_max', v), label="Max Accel.", resolution=0.01)
acc_max.set(starcopy['acc_max'])
lum_power = tkinter.Scale(root, from_=0.001, to=2., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('lum_power', v), label="Lum. Power", resolution=0.01)
lum_power.set(starcopy['lum_power'])
# lat_v_var = tkinter.Scale(root, from_=-1., to=10., orient=tkinter.HORIZONTAL,
#                       command=lambda v: update_frequency('lat_v_var', v), label="Latitude V Var", resolution=0.01)
# lat_v_var.set(starcopy['lat_v_var'])


spin_inc = tkinter.Scale(root, from_=0.001, to=90., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('spin_inc', v), label="Spin Inc", resolution=0.01)
spin_inc.set(starcopy['spin_inc'])
spin_Omega = tkinter.Scale(root, from_=0.001, to=360., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('spin_Omega', v), label="Spin Omega", resolution=0.01)
spin_Omega.set(starcopy['spin_Omega'])
spin_oa_mult = tkinter.Scale(root, from_=-1., to=1., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('spin_oa_mult', v), label="Spin OA Mult", resolution=0.01)
spin_oa_mult.set(starcopy['spin_oa_mult'])
spin_oa_sd = tkinter.Scale(root, from_=0.001, to=90., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('spin_oa_sd', v), label="Spin OA SD", resolution=0.01)
spin_oa_sd.set(starcopy['spin_oa_sd'])
spin_vel_mult = tkinter.Scale(root, from_=-1., to=6., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('spin_vel_mult', v), label="Spin Vel Mult", resolution=0.01)
spin_vel_mult.set(starcopy['spin_vel_mult'])
spin_vel_sd = tkinter.Scale(root, from_=0.001, to=90., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('spin_vel_sd', v), label="Spin Vel SD", resolution=0.01)
spin_vel_sd.set(starcopy['spin_vel_sd'])



star1amp = tkinter.Scale(root, from_=0.001, to=10., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('star1amp', v), label="Star1Amp", resolution=0.001)
star1amp.set(starcopy['star1amp'])
star2amp = tkinter.Scale(root, from_=0.001, to=10., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('star2amp', v), label="Star2Amp", resolution=0.001)
star2amp.set(starcopy['star2amp'])
star3amp = tkinter.Scale(root, from_=0.001, to=10., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('star3amp', v), label="Star3Amp", resolution=0.001)
star3amp.set(starcopy['star3amp'])
star1sd = tkinter.Scale(root, from_=-3., to=1., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('star1sd', v), label="Star1SD", resolution=0.001)
star1sd.set(starcopy['star1sd'])
star2sd = tkinter.Scale(root, from_=-3., to=1., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('star2sd', v), label="Star2SD", resolution=0.001)
star2sd.set(starcopy['star2sd'])
star3sd = tkinter.Scale(root, from_=-3., to=1., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('star3sd', v), label="Star3SD", resolution=0.001)
star3sd.set(starcopy['star3sd'])
star3dist = tkinter.Scale(root, from_=1., to=20000., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('star3dist', v), label="Star3Dist (AU)", resolution=1.)
star3dist.set(starcopy['star3dist'])


gradual_turn = tkinter.Scale(root, from_=0.01, to=180., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('gradual_turn', v), label="Gradual Turn", resolution=0.1)
gradual_turn.set(starcopy['gradual_turn'])


comp_plume_sd = tkinter.Scale(root, from_=0.01, to=180., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('comp_plume_sd', v), label="comp_plume_sd", resolution=0.1)
comp_plume_sd.set(starcopy['comp_plume_sd'])
comp_plume_max = tkinter.Scale(root, from_=0., to=540., orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('comp_plume_max', v), label="comp_plume_max", resolution=0.1)
comp_plume_max.set(starcopy['comp_plume_max'])


accel_rate = tkinter.Scale(root, from_=-5, to=1, orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('accel_rate', v), label="Log10(Accel Rate)", resolution=0.01)
accel_rate.set(starcopy['accel_rate'])
term_windspeed = tkinter.Scale(root, from_=0., to=5e3, orient=tkinter.HORIZONTAL,
                      command=lambda v: update_frequency('term_windspeed', v), label="Terminal Windspeed", resolution=5.)
term_windspeed.set(starcopy['term_windspeed'])




sliders = [ecc, inc, asc_node, arg_peri, phase, period, m1, m2,  
            distance, ws1, ws2, turnon, turnoff, opang, oblate, n_orb,
            osd, orbmin, oamp, azsd, azmin, azamp, sigma, histmax,
            compopen, compplume, compreduc, compincl, compaz, nuc_dist, opt_thin_dist, acc_max,
            lum_power, spin_inc, spin_Omega, spin_oa_mult, spin_oa_sd, spin_vel_mult, spin_vel_sd,
            star1amp, star1sd, star2amp, star2sd, star3amp, star3sd, star3dist, gradual_turn, 
            comp_plume_sd, comp_plume_max, accel_rate, term_windspeed]

num_in_row = 8
toolbar.grid(row=0, columnspan=num_in_row)
canvas.get_tk_widget().grid(row=1, column=0, columnspan=num_in_row)
for i, slider in enumerate(sliders):
    row = int(np.floor(i / num_in_row)) + 2
    col = i%(num_in_row)
    slider.grid(row=row, column=col)

button_quit.grid(row=row+1, column=num_in_row//2)



tkinter.mainloop()
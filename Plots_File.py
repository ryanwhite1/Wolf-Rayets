# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 07:57:21 2024

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

# set LaTeX font for our figures
plt.rcParams.update({"text.usetex": True})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# n = 256     # standard
n = 600     # VISIR
# n = 898     # JWST
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

def apep_plot(filename, custom_params={}):
    star = wrb.apep.copy()
    
    for param in custom_params:
        star[param] = custom_params[param]
    
    particles, weights = gm.dust_plume(star)
    X, Y, H = smooth_histogram2d(particles, weights, star)
    H = gm.add_stars(X[0, :], Y[:, 0], H, star)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')
    
    fig.savefig(f'Images/{filename}.png', dpi=400, bbox_inches='tight')
    fig.savefig(f'Images/{filename}.pdf', dpi=400, bbox_inches='tight')

def apep_cone_plot():
    def turning_point(data):
        ''' Finds the indices of the turning points when there are exactly two turning points in a 1d array. '''
        indices = np.zeros(2)
        deriv = np.diff(data)
        sign = np.sign(data[0])
        j = 0
        for i in range(len(deriv)):
            if np.sign(deriv[i]) != sign:
                indices[j] = i 
                j += 1 
                sign = np.sign(deriv[i])
        return indices.astype(int)
            
    star = wrb.apep.copy()
    star['histmax'] = 0.5
    
    particles, weights = gm.dust_plume(star)
    X, Y, H = smooth_histogram2d(particles, weights, star)
    H = gm.add_stars(X[0, :], Y[:, 0], H, star)

    # now display a circle around the cavity from the ternary star
    u = np.linspace(0, 2 * np.pi, 100)
    open_ang = np.deg2rad(star['comp_open']) / 2
    incl = np.deg2rad(star['comp_incl'])
    az = np.deg2rad(star['comp_az'])
    # formula from https://stackoverflow.com/questions/42068073/python-plotting-points-and-circles-on-a-sphere
    x = np.sin(open_ang) * np.cos(incl) * np.cos(az) * np.cos(u) + np.cos(open_ang) * np.sin(incl) * np.cos(az) - np.sin(open_ang) * np.sin(az) * np.sin(u)
    y = np.sin(open_ang) * np.cos(incl) * np.sin(az) * np.cos(u) + np.cos(open_ang) * np.sin(incl) * np.sin(az) + np.sin(open_ang) * np.cos(az) * np.sin(u)
    z = -np.sin(open_ang) * np.sin(incl) * np.cos(u) + np.cos(open_ang) * np.cos(incl)

    cone_circ = np.array([x, y, z])
    
    # get the distance to the edge (bottom) of the cone
    distance = star['windspeed1'] * star['period'] * star['phase'] * gm.yr2s
    cone_circ *= distance
    
    cone_circ, _ = gm.transform_orbits(cone_circ, np.zeros(cone_circ.shape), star)
    
    turn_x = turning_point(cone_circ[0, :])     # get the turning point indices in each of the x and y directions
    turn_y = turning_point(cone_circ[1, :])
    
    y_turn_1 = cone_circ[1, turn_y[0]]  # y-values of each turning point for the y array
    y_turn_2 = cone_circ[1, turn_y[1]] 
    
    
    point_1, point_2 = np.zeros(2), np.zeros(2)
    arg_min = np.argmin([cone_circ[0, turn_x[0]], cone_circ[0, turn_x[1]]])
    other_arg = int(not arg_min)
    point_1[0] = cone_circ[0, turn_x[arg_min]]
    point_1[1] = cone_circ[1, turn_x[arg_min]]
    point_2[0] = cone_circ[0, turn_x[other_arg]]
    point_2[1] = cone_circ[1, turn_x[other_arg]]
        
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')
    ax.plot(cone_circ[0, :], cone_circ[1, :], c='w', rasterized=True)
    ax.plot([0, np.mean(cone_circ[0, :])], [0, np.mean(cone_circ[1, :])], ls='--', c='w', rasterized=True)
    ax.plot([0, point_1[0]], [0, point_1[1]], c='w', rasterized=True)
    ax.plot([0, point_2[0]], [0, point_2[1]], c='w', rasterized=True)
    
    fig.savefig('Images/Apep_Cone.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Apep_Cone.pdf', dpi=400, bbox_inches='tight')
    
    
    fig, axes = plt.subplots(figsize=(9, 3), ncols=3, sharey=True, gridspec_kw={'wspace':0})
    
        
    axes[1].pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    axes[1].plot(cone_circ[0, :], cone_circ[1, :], c='w', rasterized=True)
    axes[1].plot([0, np.mean(cone_circ[0, :])], [0, np.mean(cone_circ[1, :])], ls='--', c='w', rasterized=True)
    axes[1].plot([0, point_1[0]], [0, point_1[1]], c='w', rasterized=True)
    axes[1].plot([0, point_2[0]], [0, point_2[1]], c='w', rasterized=True)
    
    axes[2].pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    
    star['comp_reduction'] = 0
    
    particles, weights = gm.dust_plume(star)
    X, Y, H = smooth_histogram2d(particles, weights, star)
    H = gm.add_stars(X[0, :], Y[:, 0], H, star)
    
    axes[0].pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    
    import matplotlib

    cmap = matplotlib.cm.get_cmap('hot')
    
    rgba = cmap(0.)
    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_facecolor(rgba)
        for direction in ['top', 'right', 'bottom', 'left']:
            ax.spines[direction].set_visible(False)
        xlim = np.array(ax.get_xlim())
        ylim = np.array(ax.get_ylim())
        # ax.set(xlim=1.1*xlim, ylim=1.1*ylim)
        for x in xlim:
            ax.axvline(x, c='w')
        for y in ylim:
            ax.axhline(y, c='w')
        
        # yval = 0.8 * ylim[1] #if i < 2 else 0.8 * ylim[0]
        # AX.text(0.9 * xlim[0], yval, order[i], c='w', fontsize='14')
    
    fig.savefig('Images/Apep_Cone_horiz.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Apep_Cone_horiz.pdf', dpi=400, bbox_inches='tight')

def Apep_VISIR_reference(year):
    from glob import glob
    from astropy.io import fits
    pscale = 1000 * 23/512 # mas/pixel, (Yinuo's email said 45mas/px, but I think the FOV is 23x23 arcsec for a 512x512 image?)
    
    years = {2016:0, 2017:1, 2018:2, 2024:3}
    directory = "Data\\VLT"
    fnames = glob(directory + "\\*.fits")
    
    vlt_data = fits.open(fnames[years[year]])    # for the 2024 epoch
    
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

def Apep_JWST_reference(wavelength):
    from glob import glob
    from astropy.io import fits
    directory = "Data\\JWST\\MAST_2024-07-29T2157\\JWST"
    fname = glob(directory+f"\\jw05842-o001_t001_miri_f{wavelength}w\\*_i2d.fits")[0]
    
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

def Apep_VISIR_mosaic():
    years = [2016, 2017, 2018, 2024]
    year_pos = {2016:[0, 0], 2017:[0, 1], 2018:[1, 0], 2024:[1, 1]}
    
    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2, sharex=True, sharey=True, gridspec_kw={'hspace':0.04, 'wspace':0.04})
    
    for i, year in enumerate(years):
        x, y, H = Apep_VISIR_reference(year)
        
        row, col = year_pos[year]
        axes[row, col].pcolormesh(x, y, H, cmap='hot', rasterized=True)
        axes[row, col].text(-6, 6, f'{year}', c='w', fontsize=14)
        
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.set(xlim=(-8, 8), ylim=(-8, 8))
            if i == 1:
                ax.set(xlabel='Relative RA (")')
            if j == 0:
                ax.set(ylabel='Relative Dec (")')
    
    fig.savefig('Images/Apep_VISIR_Mosaic.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Apep_VISIR_Mosaic.pdf', dpi=400, bbox_inches='tight')    
    
def Apep_VISIR_expansion():
    from glob import glob
    from astropy.io import fits
    pscale = 1000 * 23/512 # mas/pixel, (Yinuo's email said 45mas/px, but I think the FOV is 23x23 arcsec for a 512x512 image?)
    
    years = {2016:0, 2017:1, 2018:2, 2024:3}
    directory = "Data\\VLT"
    fnames = glob(directory + "\\*.fits")
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for year in list(years.keys()):
        vlt_data = fits.open(fnames[years[year]])    # for the 2024 epoch
        
        data = vlt_data[0].data
        length = data.shape[0]
        
        X = jnp.linspace(-1., 1., length) * pscale * length/2 / 1000
        Y = X.copy()
        
        lower = 140
        upper = 240
        
        data = data[600//2, lower:upper]
        data /= max(data)
        
        ax.plot(X[lower:upper], data, label=f'{year}')
    ax.legend()
    ax.set(ylabel='Relative Flux', xlabel='Relative RA (")')
    
    fig.savefig('Images/Apep_VISIR_Expansion.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Apep_VISIR_Expansion.pdf', dpi=400, bbox_inches='tight')
    
def Apep_JWST_mosaic():
    wavelengths = [770, 1500, 2550]
    # year_pos = {2016:[0, 0], 2017:[0, 1], 2018:[1, 0], 2024:[1, 1]}
    
    fig, axes = plt.subplots(figsize=(9, 3), ncols=3, sharex=True, sharey=True, gridspec_kw={'wspace':0.0})
    
    
    # from matplotlib import gridspec
    # fig = plt.figure(figsize=(9, 6.66))
    
    # gs = gridspec.GridSpec(nrows=12, ncols=12, wspace=0)
    
    # axtm = fig.add_subplot(gs[0:6, 3:9])
    # axbl = fig.add_subplot(gs[6:, 0:7])
    # axbr = fig.add_subplot(gs[6:, 6:])
    
    # axes = [axtm, axbl, axbr]
    
    for i, wavelength in enumerate(wavelengths):
        x, y, H = Apep_JWST_reference(wavelength)
        
        # row, col = year_pos[year]
        axes[i].pcolormesh(x, y, H, cmap='hot', rasterized=True)
        axes[i].text(-40, 40, f'F{wavelength}W', c='w', fontsize=14)
        axes[i].set(aspect='equal', xlabel='Relative RA (")')
        
        if i == 0:
            axes[i].set(ylabel='Relative Dec (")')
        
        
    # for i, row in enumerate(axes):
    #     for j, ax in enumerate(row):
    #         ax.set(xlim=(-8, 8), ylim=(-8, 8))
    #         if i == 1:
    #             ax.set(xlabel='Relative RA (")')
    #         if j == 0:
    #             ax.set(ylabel='Relative Dec (")')
    
    # fig.tight_layout()
    
    fig.savefig('Images/Apep_JWST_Mosaic.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Apep_JWST_Mosaic.pdf', dpi=400, bbox_inches='tight')  
    

def Apep_image_fit():
    from matplotlib.figure import Figure
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    @jit
    def smooth_histogram2d_898(particles, weights, stardata):
        im_size = 898
        
        x = particles[0, :]
        y = particles[1, :]
        
        xbound, ybound = jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))
        bound = jnp.max(jnp.array([xbound, ybound])) * (1. + 2. / im_size)
        
        xedges, yedges = jnp.linspace(-bound, bound, im_size+1), jnp.linspace(-bound, bound, im_size+1)
        return gm.smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
    @jit
    def smooth_histogram2d_w_bins_898(particles, weights, stardata, xbins, ybins):
        im_size = 898
        return gm.smooth_histogram2d_base(particles, weights, stardata, xbins, ybins, im_size)
    
    from matplotlib import gridspec
    
    # fig, axes = plt.subplots(figsize=(16, 6), ncols=3, nrows=2, gridspec_kw={'wspace':0, 'width_ratios':[w, w, 1-2*w]})
    # fig, axes = plt.subplots(figsize=(9, 6), ncols=4, nrows=2)
    
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=4, width_ratios=[1, 1, 1, 0.05], wspace=0.1, hspace=0.15)
    
    axes = []
    for i in range(2):
        subaxes = []
        for j in range(3):
            subaxes.append(fig.add_subplot(gs[i, j]))
        axes.append(subaxes)
    cbar_ax = fig.add_subplot(gs[:, -1])
    axes = np.array(axes)
            
    
    
    # titles = [['Model', '2016', '2017'], ['2018', '2024', 'JWST']]
    # w = 1/3.08
    

    X_jwst, Y_jwst, H_jwst = Apep_JWST_reference(2550)
    axes[1, 2].pcolormesh(X_jwst, Y_jwst, H_jwst, cmap='hot', rasterized=True)
    maxside_jwst = np.max(np.abs(np.array([X_jwst, Y_jwst])))
    axes[1, 2].set(xlim=(-maxside_jwst, maxside_jwst), ylim=(-maxside_jwst, maxside_jwst))

    
    starcopy = wrb.apep.copy()
    starcopy['histmax'] = 0.5
    particles, weights = gm.dust_plume(starcopy)
    X, Y, H_original = smooth_histogram2d(particles, weights, starcopy)
    axes[0, 0].pcolormesh(X, Y, H_original, cmap='hot', rasterized=True)
    axes[0, 0].set(aspect='equal', ylabel='Relative Dec (")', xlim=(-8, 8), ylim=(-8, 8))
    axes[0, 0].set_facecolor('k')
    axes[0, 0].text(-7, 6, 'Model', c='w')
    
    starcopy['histmax'] = 1.

    starcopy_3shell = starcopy.copy()
    starcopy_3shell['histmax'] = 0.15
    particles, weights = gm.gui_funcs[2](starcopy_3shell)
    X_3shell, Y_3shell, H_3shell = smooth_histogram2d_w_bins_898(particles, weights, starcopy_3shell, X_jwst[0, :], Y_jwst[:, 0])
    # H_3shell = gm.add_stars(X_3shell[0, :], Y_3shell[:, 0], H_3shell, starcopy_3shell)
    # jwst_mesh = jwst_mesh.ravel()
    norm = colors.Normalize(vmin=-1., vmax=1.)
    jwst_diff_mesh = axes[1, 2].pcolormesh(X_jwst, Y_jwst, H_3shell - H_jwst, cmap='seismic', norm=norm, rasterized=True)
    axes[1, 2].text(-45, 40, 'JWST', c='k')
    # the_divider = make_axes_locatable(axes[1, 2])
    # color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(jwst_diff_mesh, cax=cbar_ax, label='Difference')


    for j in [0, 1]:
        for i in range(0, 3):
            if j != 0:
                axes[j, i].set(aspect='equal', xlabel='Relative RA (")')
                if i == 0:
                    axes[j, i].set(ylabel='Relative Dec (")')
            else:
                axes[j, i].set(aspect='equal')
                
    year_inds = {2016:[0, 1], 2017:[0, 2], 2018:[1, 0], 2024:[1, 1], 'jwst':[1, 2]}

    for i, year in enumerate([2016, 2017, 2018, 2024]):
        a, b = year_inds[year]
        
        X_ref, Y_ref, H_ref = Apep_VISIR_reference(year)
        
        year_starcopy = starcopy.copy()
        year_starcopy['phase'] += (year - 2024) / year_starcopy['period']
        particles, weights = gm.dust_plume(year_starcopy)
        X_year, Y_year, H_year = smooth_histogram2d_w_bins(particles, weights, year_starcopy, X_ref[0, :], Y_ref[:, 0])
        # H_year = gm.add_stars(X_ref[0, :], Y_ref[:, 0], H_year, starcopy)
        
        axes[a, b].pcolormesh(X_ref, Y_ref, H_year - H_ref, cmap='seismic', norm=norm, rasterized=True)
        axes[a, b].set(xlim=(-8, 8), ylim=(-8, 8))
        axes[a, b].text(-7, 6, f'{year}', c='k')
        
    fig.savefig('Images/Apep_Fit.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Apep_Fit.pdf', dpi=400, bbox_inches='tight')
    
def apep_tertiary_movement():
    from matplotlib.figure import Figure
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    @jit
    def smooth_histogram2d_898(particles, weights, stardata):
        im_size = 898
        
        x = particles[0, :]
        y = particles[1, :]
        
        xbound, ybound = jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))
        bound = jnp.max(jnp.array([xbound, ybound])) * (1. + 2. / im_size)
        
        xedges, yedges = jnp.linspace(-bound, bound, im_size+1), jnp.linspace(-bound, bound, im_size+1)
        return gm.smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
    @jit
    def smooth_histogram2d_w_bins_898(particles, weights, stardata, xbins, ybins):
        im_size = 898
        return gm.smooth_histogram2d_base(particles, weights, stardata, xbins, ybins, im_size)
    
    
    fig, axes = plt.subplots(figsize=(9, 4), ncols=3)
            
    incls = [0, 5, 10]
    

    X_jwst, Y_jwst, H_jwst = Apep_JWST_reference(2550)

    norm = colors.Normalize(vmin=-1., vmax=1.)
    starcopy = wrb.apep.copy()
    starcopy['histmax'] = 1.
    particles, weights = gm.dust_plume(starcopy)
    year = 2024
    
    X_ref, Y_ref, H_ref = Apep_VISIR_reference(year)
    
    year_starcopy = starcopy.copy()
    year_starcopy['phase'] += (year - 2024) / year_starcopy['period']
    particles, weights = gm.dust_plume(year_starcopy)
    X_year, Y_year, H_year = smooth_histogram2d_w_bins(particles, weights, year_starcopy, X_ref[0, :], Y_ref[:, 0])
    
    axes[0].pcolormesh(X_ref, Y_ref, H_year - H_ref, cmap='seismic', norm=norm, rasterized=True)
    axes[0].set(xlim=(-4.5, 1.5), ylim=(0.5, 6.5), ylabel='Relative Dec (")')
    axes[0].text(-4, 5.85, fr'$\beta_{{\rm tert}} = {starcopy["comp_incl"]:.0f}$')
    
    for i, delta_inc in enumerate(incls[1:]):
        starcopy_3shell = starcopy.copy()
        starcopy_3shell['histmax'] = 0.25
        starcopy_3shell['comp_incl'] += delta_inc
        particles, weights = gm.gui_funcs[2](starcopy_3shell)
        X_3shell, Y_3shell, H_3shell = smooth_histogram2d_w_bins_898(particles, weights, starcopy_3shell, X_jwst[0, :], Y_jwst[:, 0])
        
        axes[i + 1].pcolormesh(X_jwst, Y_jwst, H_3shell - H_jwst, cmap='seismic', norm=norm, rasterized=True)
    
    axes[1].set(xlim=(-14, -4), ylim=(10, 20))
    axes[1].text(-13, 19, fr'$\beta_{{\rm tert}} = {starcopy["comp_incl"] + incls[1]:.0f}$')
    axes[2].set(xlim=(-31, -5), ylim=(11.5, 37.5))
    axes[2].text(-29, 35, fr'$\beta_{{\rm tert}} = {starcopy["comp_incl"] + incls[2]:.0f}$')
    
    for ax in axes:
        ax.set(aspect='equal', xlabel='Relative RA (")')
        
    fig.savefig('Images/Apep_Tertiary_Movement.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Apep_Tertiary_Movement.pdf', dpi=400, bbox_inches='tight')
    
def Apep_flipbook(pages=93):
    import matplotlib.colors as colors
    norm = colors.Normalize(vmin=0., vmax=1.)
    apep = wrb.apep.copy()
    
    apep['histmax'] = 0.5
    
    particles, weights = gm.gui_funcs[1](apep)
    X, Y, H = smooth_histogram2d(particles, weights, apep)
    
    xbins = 1.1 * X[0, :]
    ybins = 1.1 * Y[:, 0]
    
    phases = np.linspace(0, 1, pages//2)
    
    for i in range(pages//2):
        apep['phase'] = phases[i]
        particles, weights = gm.gui_funcs[1](apep)
        
        X, Y, H = smooth_histogram2d_w_bins(particles, weights, apep, xbins, ybins)
        
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.pcolormesh(X, Y, H, cmap='binary', norm=norm, rasterized=True)
        ax.set(aspect='equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        fig.savefig(f'Images/flipbook/image_{i+1}.png', dpi=400, bbox_inches='tight')
        
        plt.close('all')

def visir_gif():
    from glob import glob
    from astropy.io import fits
    pscale = 1000 * 23/512 # mas/pixel, (Yinuo's email said 45mas/px, but I think the FOV is 23x23 arcsec for a 512x512 image?)
    
    years = {2016:0, 2017:1, 2018:2, 2024:3}
    directory = "Data\\VLT"
    fnames = glob(directory + "\\*.fits")
    
    year_data = {}
    
    years_list = [2016, 2017, 2018, 2024]
    
    for year in years_list:
        vlt_data = fits.open(fnames[years[year]])    # for the 2024 epoch
        
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
        
        year_data[year] = [xs, ys, data]
        
    every = 1
    length = 2
    # now calculate some parameters for the animation frames and timing
    # nt = int(stardata['period'])    # roughly one year per frame
    nt = 4
    # nt = 10
    frames = jnp.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
    fps = len(frames) // length  # fps for the final animation
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.set_facecolor('k')
    ax.set_axis_off()
    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    
    def animate(i):
        print(i)
        ax.cla()
        xs, ys, data = year_data[years_list[i]]
        ax.pcolormesh(xs, ys, data, cmap='hot')
        ax.set(aspect='equal', xlim=(-8, 8), ylim=(-8, 8))
        ax.text(5, 6.5, f"{years_list[i]}", c='w', fontsize=20)
        return fig, 
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
    # writer = animation.FFMpegWriter(fps=fps)
    ani.save("Images/Apep_VISIR_gif.gif", writer='ffmpeg', fps=fps)

def Apep_gif():
    
    N = 200
    phases = np.linspace(0.5, 1.5, N)
    star = wrb.apep.copy()
    star['histmax'] = 0.7
    
    vmin, vmax = 0., 0.7
    
    particles, weights = gm.gui_funcs[1](star)
    X_orig, Y_orig, H = smooth_histogram2d(particles, weights, star)
    H = gm.add_stars(X_orig[0, :], Y_orig[:, 0], H, star)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(X_orig, Y_orig, H, cmap='hot', vmin=vmin, vmax=vmax, rasterized=True)
    
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    
    ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')
    
    every = 1
    length = 10
    # now calculate some parameters for the animation frames and timing
    # nt = int(stardata['period'])    # roughly one year per frame
    # nt = 10
    frames = jnp.arange(0, N, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
    fps = len(frames) // length  # fps for the final animation
    
    import matplotlib

    cmap = matplotlib.cm.get_cmap('hot')
    
    rgba = cmap(0.)
    
    def animate(i):
        if i%(N // 10) == 0:
            print(i/N * 100, "%", sep='')
        ax.clear()
        ax.set_facecolor(rgba)
            
        star['phase'] = phases[i]
        particles, weights = gm.gui_funcs[1](star)
        weights = np.array(weights)
        weights[:len(weights)//2] /= 2
        X, Y, H = smooth_histogram2d_w_bins(particles, weights, star, X_orig[0, :], Y_orig[:, 0])
        H = gm.add_stars(X[0, :], Y[:, 0], H, star)
        
        # mesh.set_array(H)
        # mesh._coordinates = np.array([X, Y])
        ax.pcolormesh(X, Y, H, cmap='hot', vmin=vmin, vmax=vmax, rasterized=True)
        
        ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")', title=rf"$\phi = {phases[i]%1:.2f}$")
        # ax.set(xlim=xlim, ylim=ylim)
        return fig, 
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
    # writer = animation.FFMpegWriter(fps=fps)
    ani.save("Images/Apep_evolution.gif", writer='ffmpeg', fps=fps)
    
    
    
    
def WR48a_plot():
    star = wrb.WR48a.copy()
    
    particles, weights = gm.gui_funcs[1](star)
    X, Y, H = gm.smooth_histogram2d(particles, weights, star)
    H = gm.add_stars(X[0, :], Y[:, 0], H, star)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")', ylim=(-4, 4))
    
    fig.savefig(f'Images/WR48a_geometry.png', dpi=400, bbox_inches='tight')
    fig.savefig(f'Images/WR48a_geometry.pdf', dpi=400, bbox_inches='tight')
    

def smooth_hist_demo():
    im_size = 16
    
    x = np.array([-1.1, 0, 0.5, 0.54, -0.536, -0.6])
    y = np.array([0, 0, 0.67, -0.698, -0.6, 0.7])
    
    particles = np.array([x, y])
    weights = np.ones(len(x))
    
    xbound, ybound = jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))
    bound = jnp.max(jnp.array([xbound, ybound])) * (1. + 2. / im_size)
    
    stardata = wrb.test_system.copy()
    stardata['sigma'] = 0.1
    
    xedges, yedges = jnp.linspace(-bound, bound, im_size+1), jnp.linspace(-bound, bound, im_size+1)
    X, Y, H = gm.smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
    
    fig, ax = plt.subplots()
    ax.set_facecolor('k')
    ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    ax.scatter(x, y, rasterized=True)
    for i in range(len(xedges)):
        ax.axhline(xedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
        ax.axvline(yedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
    ax.set(aspect='equal', xlabel=r'$x$', ylabel=r'$y$')
    
    fig.savefig('Images/Smooth_Hist_Demo.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Smooth_Hist_Demo.pdf', dpi=400, bbox_inches='tight')
    
    
    H, _, _, _ = plt.hist2d(x, y, bins=[xedges, yedges])
    
    fig, ax = plt.subplots()
    ax.set_facecolor('k')
    ax.pcolormesh(X, Y, H.T, cmap='hot', rasterized=True)
    ax.scatter(x, y, rasterized=True)
    for i in range(len(xedges)):
        ax.axhline(xedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
        ax.axvline(yedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
    ax.set(aspect='equal', xlabel=r'$x$', ylabel=r'$y$')
    
    fig.savefig('Images/Normal_Hist.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Normal_Hist.pdf', dpi=400, bbox_inches='tight')
    
def smooth_hist_gif():
    im_size = 10
    
    x = np.array([-0.8])
    y = np.array([0])
    
    particles = np.array([x, y])
    weights = np.ones(len(x))
    
    xbound = ybound = 1
    bound = jnp.max(jnp.array([xbound, ybound])) * (1. + 2. / im_size)
    
    stardata = wrb.test_system.copy()
    stardata['sigma'] = 0.1
    
    xedges, yedges = jnp.linspace(-bound, bound, im_size+1), jnp.linspace(-bound, bound, im_size+1)
    X, Y, H = gm.smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
    
    fig, ax = plt.subplots()
    ax.set_facecolor('k')
    mesh = ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    scatter = ax.scatter(x, y, rasterized=True)
    ax.set(aspect='equal', xlabel=r'$x$', ylabel=r'$y$')
    for i in range(len(xedges)):
        ax.axhline(xedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
        ax.axvline(yedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
    
    
    every = 1
    length = 10
    # now calculate some parameters for the animation frames and timing
    # nt = int(stardata['period'])    # roughly one year per frame
    nt = 100
    # nt = 10
    frames = jnp.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
    fps = len(frames) // length  # fps for the final animation
    
    xs = np.linspace(x[0], 1, nt)
    ys = xs**2
    
    Hs_smooth = []
    for i in range(nt):
        current_xs = np.array([xs[i], 0])
        current_ys = np.array([ys[i], 0])
        particles = np.array([current_xs, current_ys])
        weights = np.array([1, 0])
        X, Y, H = gm.smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
        Hs_smooth.append(H)
    
    def animate(i):
        if i%(nt // 10) == 0:
            print(i/nt * 100, "%", sep='')
        
        mesh.set_array(Hs_smooth[i])
        scatter.set_offsets(np.c_[xs[i], ys[i]])
        return fig, 

    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
    ani.save(f"Images/Smooth_Hist_Gif.gif", writer='pillow', fps=fps)
    
    # now for the normal histogramming
    fig, ax = plt.subplots()
    ax.set_facecolor('k')
    current_xs = np.array([xs[0], 0])
    current_ys = np.array([ys[0], 0])
    weights = np.array([1, 0])
    H, X, Y = np.histogram2d(current_xs, current_ys, bins=X[0], weights=weights)
    mesh = ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    scatter = ax.scatter(x, y, rasterized=True)
    ax.set(aspect='equal', xlabel=r'$x$', ylabel=r'$y$')
    for i in range(len(xedges)):
        ax.axhline(xedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
        ax.axvline(yedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
    Hs_normal = []
    for i in range(nt):
        current_xs = np.array([xs[i], 0])
        current_ys = np.array([ys[i], 0])
        H, _, _ = np.histogram2d(current_xs, current_ys, bins=X, weights=weights)
        Hs_normal.append(H.T)
    
    def animate(i):
        if i%(nt // 10) == 0:
            print(i/nt * 100, "%", sep='')
        
        mesh.set_array(Hs_normal[i])
        scatter.set_offsets(np.c_[xs[i], ys[i]])
        return fig, 

    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
    ani.save(f"Images/Normal_Hist_Gif.gif", writer='pillow', fps=fps)
    
    # now for both simultaneously
    
    fig, axes = plt.subplots(figsize=(14, 7), ncols=2, gridspec_kw={'wspace':0})
    for ax in axes:
        ax.set_facecolor('k')
        ax.set(aspect='equal', xlabel=r'$x$', ylabel=r'$y$')
        for i in range(len(xedges)):
            ax.axhline(xedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
            ax.axvline(yedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
    axes[0].axvline(xedges[-1], c='w', lw=2, rasterized=True)
    axes[1].axvline(xedges[0], c='w', lw=2, rasterized=True)
    axes[0].set(title='Normal Histogram')
    axes[1].set(ylabel=None, yticklabels=[], title='Smooth Method')
    axes[1].tick_params(left=False)
        
    mesh_hist = axes[0].pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    scatter_hist = axes[0].scatter(x, y, rasterized=True)
    
    mesh_smooth = axes[1].pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    scatter_smooth = axes[1].scatter(x, y, rasterized=True)
    
    
    def animate(i):
        if i%(nt // 10) == 0:
            print(i/nt * 100, "%", sep='')
        
        mesh_hist.set_array(Hs_normal[i])
        scatter_hist.set_offsets(np.c_[xs[i], ys[i]])
        mesh_smooth.set_array(Hs_smooth[i])
        scatter_smooth.set_offsets(np.c_[xs[i], ys[i]])
        
        return fig, 

    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
    ani.save(f"Images/Hist_Comparison_Gif.gif", writer='pillow', fps=fps)
    
    

    

def variation_gaussian():
    '''Plots the Gaussian used for the azimuthal and orbital variation equations.'''
    
    gaussian = lambda A, theta, minimum, sigma: np.maximum(1 - (1 - A) * np.exp(-0.5 * ((theta - minimum) / sigma)**2), 0)
    
    As = [0.5, 0.1, 0, -1]
    sigmas = [10, 25, 50, 50]
    
    minimum_az = 0
    minimum_orb = 0
    
    n = 500
    thetas = np.linspace(-180, 180, n)
    
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax2 = ax.twiny()
    
    for i in range(len(As)):
        values = gaussian(As[i], thetas, minimum_az, sigmas[i])
        
        ax.plot(thetas, values, label=f'$A={As[i]}$, $\sigma={sigmas[i]}^\circ$')
        ax2.plot(thetas, values)
    
    ax.legend(frameon=False)
    
    ax.set(xlabel='Particle Angle from Leading Edge', ylabel='Relative Dust Production Strength')
    ax2.set(xlabel="True Anomaly from Periastron")
    
    fig.savefig('Images/Variation_Gaussian.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Variation_Gaussian.pdf', dpi=400, bbox_inches='tight')
    
    
    
    
def effects_compare():
    '''Compares the effects of orbital + azimuthal + gradual dust variation on a single plot.'''
    # fig, axes = plt.subplots(nrows=2, ncol=3, gridspec_kw={'hspace':0, 'wspace':0})
    from matplotlib import gridspec
    
    # fig, ax = plt.subplots(figsize = (8,6))
    fig = plt.figure(figsize=(8, 6.66))
    
    gs = gridspec.GridSpec(nrows=5, ncols=6, hspace=0, wspace=0)
    
    axtl = fig.add_subplot(gs[0:3, 0:3])
    axtr = fig.add_subplot(gs[0:3, 3:6])
    axbl = fig.add_subplot(gs[3:, 0:2])
    axbm = fig.add_subplot(gs[3:, 2:4])
    axbr = fig.add_subplot(gs[3:, 4:])
    
    axes = [axtl, axtr, axbl, axbm, axbr]
    order = ['Basic', 'Full Variation', 'Azimuthal Variation', 'Orbital Modulation', 'Gradual Turn On/Off']
    
    
    
    test = wrb.WR48a.copy()
    
    
    test_ = test.copy()
    test_['orb_sd'] = 0
    test_['az_sd'] = 0
    test_['gradual_turn'] = 0
    particles, weights = gm.dust_plume(test_)
    X, Y, H = gm.smooth_histogram2d(particles, weights, test_)
    axtl.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    
    test_ = test.copy()
    particles, weights = gm.dust_plume(test_)
    X, Y, H = gm.smooth_histogram2d(particles, weights, test_)
    axtr.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    
    test_ = test.copy()
    test_['orb_sd'] = 0
    test_['gradual_turn'] = 0
    particles, weights = gm.dust_plume(test_)
    X, Y, H = gm.smooth_histogram2d(particles, weights, test_)
    axbl.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    
    test_ = test.copy()
    test_['az_sd'] = 0
    test_['gradual_turn'] = 0
    particles, weights = gm.dust_plume(test_)
    X, Y, H = gm.smooth_histogram2d(particles, weights, test_)
    axbm.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    
    test_ = test.copy()
    test_['orb_sd'] = 0
    test_['az_sd'] = 0
    particles, weights = gm.dust_plume(test_)
    X, Y, H = gm.smooth_histogram2d(particles, weights, test_)
    axbr.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    
    import matplotlib

    cmap = matplotlib.cm.get_cmap('hot')
    
    rgba = cmap(0.)
    
    for i, AX in enumerate(axes):
        AX.get_xaxis().set_visible(False)
        AX.get_yaxis().set_visible(False)
        
        AX.set_facecolor(rgba)
        
        for direction in ['top', 'right', 'bottom', 'left']:
            AX.spines[direction].set_visible(False)
            
        xlim = np.array(AX.get_xlim())
        ylim = np.array(AX.get_ylim())
        AX.set(xlim=1.1*xlim, ylim=1.1*ylim)
        for x in xlim:
            AX.axvline(1.1 * x, c='w')
        for y in ylim:
            AX.axhline(1.1 * y, c='w')
        
        yval = 0.8 * ylim[1] #if i < 2 else 0.8 * ylim[0]
        AX.text(0.9 * xlim[0], yval, order[i], c='w', fontsize='14')
            
        
    
    fig.tight_layout()
    
    fig.savefig('Images/Variation_Effects.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Variation_Effects.pdf', dpi=400, bbox_inches='tight')

def anisotropy_compare():
    
    fig, axes = plt.subplots(figsize=(6, 3.5), ncols=2, gridspec_kw={'hspace':0, 'wspace':0})
    
    test = wrb.WR104.copy()
    test_aniso = test.copy()
    test_aniso['spin_inc'] = 24
    test_aniso['spin_Omega'] = 16
    test_aniso['aniso_vel_mult'] = -5.45
    
    
    particles, weights = gm.dust_plume(test)
    X, Y, H = smooth_histogram2d(particles, weights, test)
    xbins = 1.1 * X[0, :]
    ybins = 1.1 * Y[:, 0]
    
    particles_aniso, weights_aniso = gm.dust_plume(test_aniso)
    X_aniso, Y_aniso, H_aniso = smooth_histogram2d(particles_aniso, weights_aniso, test_aniso)
    xbins_aniso = 1.1 * X_aniso[0, :]
    ybins_aniso = 1.1 * Y_aniso[:, 0]
    
    
    
    X, Y, H = smooth_histogram2d_w_bins(particles, weights, test, xbins_aniso, ybins_aniso)
    axes[0].pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    
    
    
    particles, weights = gm.dust_plume(test)
    X, Y, H = smooth_histogram2d_w_bins(particles_aniso, weights_aniso, test_aniso, xbins_aniso, ybins_aniso)
    axes[1].pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    
    order = ['Original', 'Anisotropic']
    
    import matplotlib

    cmap = matplotlib.cm.get_cmap('hot')
    
    rgba = cmap(0.)
    
    for i, AX in enumerate(axes):
        AX.set(aspect='equal')
        AX.set_facecolor(rgba)
        AX.get_xaxis().set_visible(False)
        AX.get_yaxis().set_visible(False)
        
        xlim = np.array(AX.get_xlim())
        ylim = np.array(AX.get_ylim())
        AX.set(xlim=1.1*xlim, ylim=1.1*ylim)
        
        yval = 0.8 * ylim[0] #if i < 2 else 0.8 * ylim[0]
        AX.text(0, yval, order[i], c='w', fontsize='14')
    
    
    fig.savefig('Images/Anisotropy_Effects.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Anisotropy_Effects.pdf', dpi=400, bbox_inches='tight')

def smooth_hist_gradient():
    im_size = 15
    
    xbound = 10
    ybound = 10
    bound = jnp.max(jnp.array([xbound, ybound])) * (1. + 2. / im_size)
    
    xedges, yedges = jnp.linspace(-bound, bound, im_size+1), jnp.linspace(-bound, bound, im_size+1)
    
    stardata = wrb.test_system.copy()
    stardata['sigma'] = 0.1
    stardata['histmax'] = 2
    
    n = 1000
    x = np.linspace(-5, 5, n)
    y = np.ones(n) * 0 
    
    weights = jnp.array([1])
    
    def bin_value(X):
        new_parts = jnp.array([[-5, X], [-5, 0.]])
        X, Y, H = gm.smooth_histogram2d_base(new_parts, weights, stardata, xedges, yedges, im_size)
        
        return H[im_size//2, im_size//2].astype(float)
    
    gradient = vmap(grad(bin_value, allow_int=True))
    val = vmap(bin_value)
    
    values = np.zeros(n, dtype=float)
    grads = np.zeros(n, dtype=float)
    L = xedges[1] - xedges[0]
    xs = np.linspace(-1.5 * L, 1.5 * L, n)
    
    values = val(xs)
    grads = gradient(xs)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(xs / L, values / max(values), label='Bin Value')
    ax.plot(xs / L, grads / max(values), label='Bin Value Gradient')
    ax.legend()
    ax.set(xlabel='Particle Distance from Bin Center ($L$ distances)', ylabel='Bin/Gradient Value')
    
    fig.savefig('Images/Smooth_Hist_Gradient.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Smooth_Hist_Gradient.pdf', dpi=400, bbox_inches='tight')
    
    
    XX = 0. 
    new_parts = jnp.array([[-5, XX], [-5, 0.]])
    X, Y, H = gm.smooth_histogram2d_base(new_parts, weights, stardata, xedges, yedges, im_size)
    fig, ax = plt.subplots()
    ax.set_facecolor('k')
    ax.pcolormesh(X, Y, H, cmap='hot', rasterized=True)
    ax.scatter(XX, 0., rasterized=True)
    for i in range(len(xedges)):
        ax.axhline(xedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
        ax.axvline(yedges[i], c='tab:grey', lw=0.5, ls='--', rasterized=True)
    ax.set(aspect='equal', xlabel=r'$x$', ylabel=r'$y$')
    
def WR140_lightcurve():
    phases, fluxes = gm.generate_lightcurve(wrb.WR140, n=100, shells=2)
    
    fig, ax = plt.subplots(figsize=(4, 6.75))
    
    fluxes /= max(fluxes)
    
    phases_orig = phases.copy()
    
    phases = np.concatenate((phases - 1, phases))
    phases = np.concatenate((phases, 1 + phases_orig))
    
    fluxes = np.tile(fluxes, 3)
    
    ax.plot(phases, fluxes)
    ax.set(xlabel='Phase', ylabel='Flux', yscale='log', xlim=(-0.1, 1.1))
    
    fig.savefig('Images/WR140_Light_Curve.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/WR140_Light_Curve.pdf', dpi=400, bbox_inches='tight')
    
def WR48a_lightcurve():
    ''' Thanks to https://stackoverflow.com/questions/21920233/matplotlib-log-scale-tick-label-number-formatting for the y axis tick labels
    '''
    from matplotlib.ticker import FuncFormatter
    phases, fluxes = gm.generate_lightcurve(wrb.WR48a, n=100, shells=5)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    fluxes /= max(fluxes)
    
    phases_orig = phases.copy()
    
    phases = np.concatenate((phases - 1, phases))
    phases = np.concatenate((phases, 1 + phases_orig))
    
    fluxes = np.tile(fluxes, 3)
    
    # ax.scatter(phases, fluxes)
    # ax.set(xlabel='Phase', ylabel='Flux', yscale='log', xlim=(-0.1, 1.2))
    # for x in [0.1, 1.1]:
    #     ax.axvline(x, ls='--', c='tab:red')
    # ax.grid(True)
    
    # peri_year = 1970
    peri_year = 2004 - wrb.WR48a['phase'] * wrb.WR48a['period']
    
    left_p = -0.05
    right_p = 1.05
    
    
    scattersize = 8
    ax.plot(phases, fluxes)
    # ax.scatter(phases, fluxes, s=scattersize)
    ax.set(xlabel='Phase', ylabel='Flux', yscale='log')
    # for x in [0.1, 1.1]:
    #     ax.axvline(x, ls='--', c='tab:red')
    ax.grid(True)
    ax.set(xlim=(left_p, right_p))
    ax.minorticks_on()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    
    ax2 = ax.twiny()
    ax2.minorticks_on()
    
    # ax2.scatter(phases * wrb.WR48a['period'] + peri_year, fluxes, s=scattersize)
    ax2.plot(phases * wrb.WR48a['period'] + peri_year, fluxes)
    ax2.set(yscale='log', xlim=(left_p * wrb.WR48a['period'] + peri_year, right_p * wrb.WR48a['period'] + peri_year))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax2.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    
    fig.savefig('Images/WR48a_Light_Curve.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/WR48a_Light_Curve.pdf', dpi=400, bbox_inches='tight')
    
    
def apep_orbit():
    apep = wrb.apep.copy()
    pc_to_AU = 206265
    km_to_AU = 6.68459e-9
    
    n = 10000
    
    angular_sep = 47 # mas
    angular_sep_unc = 6
    
    inclinations = np.random.normal(24, 3, size=n)
    
    angular_sep_ests = np.random.normal(angular_sep, angular_sep_unc, size=n)
    angular_sep_ests /= np.sin(np.pi/2 - np.deg2rad(inclinations))
    
    distance = 2400 # pc
    distance_unc = 200
    
    dist_ests = np.random.normal(distance, distance_unc, size=n)
    
    abs_dists = np.tan(np.deg2rad(angular_sep_ests / (1e3 * 60 * 60))) * dist_ests * pc_to_AU
    
    abs_dist = np.mean(abs_dists)
    abs_dist_unc = np.std(abs_dists)
    
    print(abs_dist, abs_dist_unc)
    
    phase2016 = apep['phase'] - (2024 - 2016) / apep['period']
    phase_unc = 0.02
    
    # phase2016 = 0.5
    
    phases = np.random.normal(phase2016, phase_unc, size=n)
    
    eccentricities = np.random.normal(apep['eccentricity'], 0.04, size=n)
    eccentricities = np.minimum(eccentricities, 0.98)
    
    eccentric_anoms = gm.kepler(phases * 2 * np.pi, eccentricities)
    
    eccentric_anom = np.mean(eccentric_anoms)
    eccentric_anom_err = np.std(eccentric_anoms)
    
    true_anoms = gm.true_from_eccentric_anomaly(eccentric_anoms, eccentricities)
    
    true_anom = np.mean(true_anoms)
    true_anom_unc = np.std(true_anoms)
    
    periods = np.random.normal(apep['period'], 10, size=n)
    periods_s = periods * 365 * 24 * 60 * 60
    
    
    
    M = 20
    m1 = M / 4
    m2 = 3 * M / 4
    
    a1, a2 = gm.calculate_semi_major(periods_s, m1, m2)
    r1 = a1 * (1. - eccentricities * jnp.cos(eccentric_anoms)) * 1e-3     # radius in km 
    r2 = a2 * (1. - eccentricities * jnp.cos(eccentric_anoms)) * 1e-3
    
    print(np.mean(r1), np.mean(r2))
    
    separation = (r1 + r2) * km_to_AU
    separation = np.mean(separation)
    
    print(separation)
    
    
    a = (abs_dists / (1 - eccentricities**2)) * (1 + eccentricities * np.cos(true_anoms))
    
    print(np.mean(a))
    
    mass = ((a * gm.AU2km * 1e3)**3 / (gm.G * (periods_s / (2 * np.pi))**2 )) / gm.M_odot
    
    print(np.mean(mass))
    
    

    

    
def main():
    # apep_plot('Apep_Plot')
    # apep_plot('Apep_Plot_No_Photodiss', custom_params={'comp_reduction':0})
    # apep_cone_plot()
    
    # Apep_VISIR_mosaic()
    # Apep_VISIR_expansion()
    # visir_gif()
    # apep_orbit()
    # Apep_gif()
    
    # Apep_JWST_mosaic()
    # Apep_image_fit()
    # apep_tertiary_movement()
    
    Apep_flipbook(pages=93)
    
    # smooth_hist_demo()
    # smooth_hist_gif()
    # smooth_hist_gradient()
    
    # variation_gaussian()
    
    # anisotropy_compare()
    
    # WR140_lightcurve()
    
    # WR48a_lightcurve()
    # WR48a_plot()
    



if __name__ == "__main__":
    main()
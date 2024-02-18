# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:07:02 2024

@author: ryanw (translated from Yinuo Han's code!)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def rectify(theta):
    x = theta / (2 * np.pi)
    return 2 * np.pi * (x + np.floor(0.5 - x))

def kepler_solve(t, P, ecc):
    maxj = 50; # Max number of iterations
    tol = 1e-8; # Convergence tolerance
    
    M = 2 * np.pi / P * t
    E = np.zeros(len(t))
    
    for i in range(len(t)):
        E0 = M[i]
        
        # Newton's formula to solve for eccentric anomoly
        for j in range(maxj):
            E1 = E0 - (E0 - ecc * np.sin(E0) - M[i]) / (1 - ecc * np.cos(E0))
            if abs(E1 - E0) < tol:
                break
            E0 = E1
        
        if j == maxj:
            print('Did not converge')
        
        E[i] = E1
    
    # Compute 2-dimensional spiral angles & radii
    # return 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E / 2))
    return 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(E/2), np.sqrt(1 - ecc) * np.cos(E/2))

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


def spiral(skeleton, gif, Title, dim, pix, windspeed, period, inclination, big_omega, turn_off, eccentricity, omega_lock, 
           little_omega, periastron, cone_angle, offset, n_circ, theta_lim):
    # Plotting parameters
    n_p = 200;                      # Number of points per circle
    n_c = 300;                     # Number of circles per period
    
    im_siz = dim;        #360       # Image size (pixels) (square)
    im_res = pix;                   # Image resolution (mas/pix)
    
    # Physics parameters
    w     = windspeed / 365.25;        # windspeed [mas/year]->[mas/day]
    P     = period * 365.25;           # period [year]->[day]
    omega = np.deg2rad(little_omega);  # omega
    inc   = np.deg2rad(inclination);   # inclination  
    Ohm   = np.deg2rad(big_omega);     # Omega 
    ecc   = eccentricity;           # Eccentricity
    pa    = periastron * 365.25;       # Periastron date (Julian days)
    cone  = np.deg2rad(cone_angle);    # cone angle (deg)
    offst = offset * 365.25;           # time offset [year]->[day]
    #offst = t_obs - pa;            # time offset
    rnuc = turn_off;
    lim = theta_lim / 180 * np.pi;         # limit for theta to produce dust
    
    # Don't use in general
    if omega_lock:
        adjust = kepler_solve(pa-245, P, ecc);
        omega = rectify(omega + adjust);
        
    n_points = n_circ * n_c
    points = np.arange(0, n_points)
    # Time vector
    t = points / n_c * P + offst 
    
    # Angles from Keplers laws as a function of time
    theta = kepler_solve(t, P, ecc)
    
    # Radius of each dust circle
    r2 = w * (n_circ * P - points / n_c * P)
    # First ring in time has biggest radius
    
    # 3D spiral plume
    # NOTE: Spiral is always generated to align with first point
    # on the x-axis (theta = 0). The result is then rotated by 
    # the specified angles using direction cosine matrices 
    # (note: order of rotations is important & must be preserved
    # - omega, inc, Omega)
    
    # Generate the coordinates of the initial unit circle
    # which is copied around the spiral
    chi = np.arange(0, n_p - 1) / n_p * np.pi * 2; # Angle
    ones2 = np.ones(len(chi)) # Just ones
    
    # The circle is parallel to the y-z plane
    circ = np.array([np.cos(cone/2) * ones2, # x - becomes North
            np.sin(cone/2) * np.cos(chi), # y - becomes East
            np.sin(cone/2) * np.sin(chi)]) # z - becomes (-) line of sight
    
    # With anisotropic winds, can try and ellipse
    # elongation_factor = 1;
    # circ = [(cos(cone/2)*ones2); # x
    #         (sin(cone/2)*cos(chi)); # y
    #         (sin(cone/2)*sin(chi)) * elongation_factor]; # z
    
    # Initialise full array to store coordinates of all points
    circfull = np.zeros((3, (n_p + 1) * len(theta)))
    # gen = range(0, n_p) # indices of one circle
    
    # Calculate coordinates of each circle on the spiral
    for j in range(len(theta)):
        if r2[j] >= rnuc :
            if rectify(theta[j]) >= lim[0]:
                if rectify(theta[j]) <= lim[1]:
                    # circj = rotate_z(theta[j]) * circ * r2[j]
                    # circfull[:, (j-1)*n_p + gen] = circj
                    
                    circj = np.zeros((3, len(chi)))
                    for ii in range(len(chi)):
                        circj[:, ii] = np.matmul(rotate_z(theta[j]), circ[:, ii] * r2[j])
                    left_ind = j *n_p
                    circfull[:, left_ind:left_ind+n_p-1] = circj
    
    # TEST orbit
    # Input
    # im_res = 1
    # e = ecc
    # a = 100
    # c = e * a;
    # b = sqrt(a^2 - c^2);
    # 
    # circfull = zeros(3, (n_p+1)*length(theta));
    # for theta = 1:1:360
    #     circfull(1,floor(theta)) = a*cosd(theta);
    #     circfull(2,floor(theta)) = b*sind(theta);
    # end
    # 
    # circfull(1,361) = c;
    # circfull(2,361) = 0;
    # 
    # for i = 362:1:362+a-1
    #     circfull(1,floor(i)) = i-362;
    #     circfull(2,floor(i)) = 0;
    # end
    
    # Rotate points by specifed rotations -------------------
    # circfull = rotate_z(Ohm) * (rotate_x(inc) * (rotate_z(omega) * circfull));
    #circfull = circfull' * rotate_z(omega) * rotate_x(inc) * rotate_z(Ohm) ;
    #circfull = circfull';
    
    circfull = np.matmul(np.matmul(np.matmul(rotate_z(Ohm), rotate_x(inc)), rotate_z(omega)), circfull)
    
    # Variable density across spiral
    use_density = 1; # Varies across angle within each circle
    use_d2 = 1; # Varies across spiral so circles differ
    
    comment = "";
    
    if use_density:
        # Goes sinusoidally between 1 and 3, reaching 3 at {densest_angle}
        #densest_angle = 230;
        #density = cos(chi - densest_angle/180*pi) + 1; 
        
        # Gaussian
        densest_angle = 180;
        # densest angle = 230 -30 +20
        # densest_angle = 180 fixed, used in paper
        sd = 80;
        # sd = 60 -10 +30;
        # sd = 80 -20 +20;
        dist_chi = min([min(abs(chi / np.pi * 180 - densest_angle)), 
                        min(abs(chi / np.pi * 180 + 360 - densest_angle)), 
                        min(abs(chi / np.pi * 180 - 360 - densest_angle))])
        density = np.exp(-((dist_chi / 180 * np.pi) / (sd / 180 * np.pi))**2)
        
        comment = comment + " " + str(densest_angle) + "/" + str(sd);
    
    if use_d2:
        # Turn off completely at some stage
        #d2 = (theta < -120/180*pi) + (theta > 120/180*pi);
        
        # Inverse Gaussian
        #least_dense = 0; # perisatron
        #d2 = 1 - exp(-((theta - least_dense/180*pi) / (50/180*pi)).^2);
        
        # Double Gaussian
        dense1 = -135
        dense2 = 135
        sd = 40
        # sd = 40 -10 +30
        
        dist_theta1 = min([min(abs(theta / np.pi * 180 - dense1)),
                           min(abs(theta / np.pi * 180 + 360 - dense1)),
                           min(abs(theta / np.pi * 180 - 360 - dense1))]);
        dist_theta2 = min([min(abs(theta / np.pi * 180 - dense2)),
                           min(abs(theta / np.pi * 180 + 360 - dense2)),
                           min(abs(theta / np.pi * 180 - 360 - dense2))])
        
        d2 = np.exp(-((dist_theta1 / 180 * np.pi) / (sd / 180 * np.pi))**2) + np.exp(-((dist_theta2 / 180 * np.pi) / (sd / 180 * np.pi))**2)
    
        comment = comment + " " + str(dense1) + "/" + str(dense2) + "/" + str(sd);
        
        # Plotting
    #     distances = true_to_radius(theta/pi*180, 1, eccentricity);
    #     phases = mod(t/365.25.25, period)/period;
    #     
    #     d2_plot = d2;
    #     d2_plot(theta/pi*180 < -135) = 0;
    #     d2_plot(theta/pi*180 > 135) = 0;
    #     
    #     figure
    #     hold on
    #     #plot(theta)
    #     plot([phases 1+phases], [distances distances], '.')
    #     plot([phases 1+phases], [d2_plot d2_plot], '.')
    #     
    #     figure
    #     hold off
    #     plot(distances, d2_plot)
    #     figure
    
    # Generate image
    n_total = n_p * len(theta)
    im = np.zeros((im_siz, im_siz))
    
    # TEST line
    # circfull = zeros(3, (n_p+1)*length(theta));
    # for i = 1:100
    #     circfull(1,i) = i;
    #     circfull(2,i) = i/2;
    # end
    
    # Project 3D spiral to pixel values by looping over all points
    for i in range(n_total):
        
        # View along z axis
        imy = np.fix(circfull[1, i] / im_res + im_siz / 2);
        imx = im_siz - np.fix(circfull[2, i] / im_res + im_siz / 2);
        imy, imx = int(imy), int(imx)
        
        #imy = fix(circfull(1, i)/im_res + im_siz/2);
        #imx = fix(circfull(2, i)/im_res + im_siz/2);
        
        # Image y = up = North = coordinate x
        # Image -x = left = East = coordinate y
        
        # Add density
        if (imx > 0) and (imx < im_siz) and (imy > 0) and (imy < im_siz):
            if use_density and ~use_d2:
                # print((i-1)%n_p + 1)
                # dens = density[(i-1)%n_p + 1]
                dens = density
                im[imy, imx] = im[imy, imx] + dens
            elif use_d2 and ~use_density:
                dens2 = d2(np.floor((i - 1) / n_p) + 1)
                im[imy, imx] = im[imy, imx] + dens2
            elif use_d2 and use_density:
                # dens3 = density[(i-1)%n_p + 1] * d2(np.floor((i - 1) / n_p) + 1)
                dens3 = density * d2(np.floor((i - 1) / n_p) + 1)
                im[imy, imx] = im[imy, imx] + dens3
            else:
                im[imy, imx] = im[imy, imx] + 1
        # Coordinate 1 = image y
        # Coordinate 2 = image x
    
    # Get rid of centre
    #im(im_siz/2, im_siz/2) = 0;
    im[im_siz // 2, im_siz // 2] = 1/8 * (sum(sum(im[im_siz//2-1:im_siz//2+1, im_siz//2-1:im_siz//2+1])) - im[im_siz//2, im_siz//2])
    
    # Normalise
    if np.max(im) > 0 and ~gif:
        im = im / np.max(im)
    
    
    # ---------- PLOTTING ----------
    scale = im_res / 1e3;
    dim = im_siz / 2;
    x = np.linspace(-dim*scale, dim*scale, int(dim)*2);
    
    if ~gif:
        
        # DATA
        fig, ax = plt.subplots()
        # ax.imshow(x, x, skeleton)
        # ax.imshow(skeleton, extent=[min(x), max(x), min(x), max(x)])
        ax.set_xlabel("Relative RA ('')")
        ax.set_ylabel("Relative Dec ('')")
        # set(gca,'YDir','normal')
        
        # MODEL OUTLINE
        # hold on
        high_pass_model = gaussian_filter(im - gaussian_filter(im, 2), 1);
        model_threshold = 0.0005;
        high_pass_model[high_pass_model > model_threshold] = 1;
        high_pass_model[high_pass_model <= model_threshold] = 0;
        high_pass_model = gaussian_filter(high_pass_model, 1)
        high_pass_model = high_pass_model / np.max(high_pass_model)
        
        # alpha = (1 - (high_pass_model < 0.85)) * 0.8;
        alpha = 1
        # h2 = ax.imshow(x, x, high_pass_model, alpha=alpha)
        ax.imshow(high_pass_model, extent=[min(x), max(x), min(x), max(x)], alpha=alpha)
        ax.set_xlabel("Relative RA ('')")
        ax.set_ylabel("Relative Dec ('')")
        
        # #colormap gray
        # set(gca,'YDir','normal')

        ax.set_title(Title)
    
    # # MODEL
    # if gif:
    #     fig, ax = plt.subplots()
    #     #h = imagesc(x,x,-min(im,0.1));
    #     #imagesc(-min(im,0.1));
    #     imagesc(-min(im,10));
    #     xlabel("Relative RA ('')")
    #     ylabel("Relative Dec ('')")
    #     colormap gray
    #     axis image
    #     set(gca,'YDir','normal')
    #     title(Title)
    return im, comment
    



# Generate at a specific phase
obs_dates = [""];
outnames = [""];
phases = [0.47];
dims = [256];
pixs = [10];

# Params for high
#dims = [512, 512, 512, 512, 512, 512];
#pixs = [10/4, 10/4, 9.52/4, 9.52/4, 10/4, 9.52];

for group in range(len(obs_dates)):
    # ---------- Model ----------
    # Fixed
    omega_lock = False
    period = 125
    eccentricity = 0.7
    little_omega = 0
    periastron = 1946
    big_omega = -88
    inclination = 25
    distance = 2.4

    # Masses
    mwc = 10 # ± 0.5 Msum for the WC7
    mo =  13 # ± 1.3 Msun for the O5
    M = mwc + mo # Enclosed mass
    a = (M*period**2)**(1/3); # semimajor axis in AU

    # Constants
    Msun = 1.98847e30; # kg
    G = 6.67408e-11; # SI units
    AU = 1.495978707e11; # m

    # Vis-Viva
    d = a * (1 - eccentricity) # AU
    v = np.sqrt(G * M * Msun * (2 / (d * AU) - 1 / (a * AU))) * 1e-3; # km/s

    # Adjustable
    windspeed = 3400 * 0.210805 / distance; # km/s to mas/year
    # windspeed = 2460 -130 +130

    cone_angle = 60*2;
    # cone angle = (42 -5 +5) * 2

    theta_lim = np.array([-114, 150])
    # Low = -(135 -5 +10), High = 135 -5 +15

    # On and off
    turn_off = 0;
    n_circ = 1;
    
    # Use below when obs_date given but not phase
    #obs_date = obs_dates(group);
    
    # Use below when phase given but not obs_date
    phase = phases[group];
    obs_date = periastron + period * (1 + phase);
    
    offset = obs_date - n_circ*period - periastron;
    # The programs implies: {n_circ} periods ago, the position angle of the
    # binary corresponds to an offset from periastron of {offset} years

    # Calculate phase
    #phase = (obs_date - periastron)/period;
    #phase = round(phase - floor(phase), 3);

    # Generate Spiral
    dim = dims[group]
    pix = pixs[group]
    make_gif = 0;
    Title = "";
    [im, comment] = spiral([], make_gif, Title, dim, pix, windspeed, period, inclination, big_omega, turn_off,
                           eccentricity, omega_lock, little_omega, periastron, cone_angle, offset, n_circ, theta_lim)

    # Write to fits
    print(phase)
    #fitswrite(im, outdir + "VIS4inc90_" + outnames(group) + ".fits")
    #fitswrite(im, "/Users/yinuo/Desktop/Test1.fits")
    
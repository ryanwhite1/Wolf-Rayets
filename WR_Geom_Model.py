# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 08:36:43 2024

@author: ryanw
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, grad
import jax
import jax.lax as lax
import jax.scipy.stats as stats
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import jax.scipy.signal as signal
from matplotlib import animation
import time
import emcee
# import jaxoplanet
import jaxopt

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



def rotate_x(angle):
    ''' Rotation matrix about the x-axis
    '''
    arr = jnp.array([[1, 0, 0],
                     [0, jnp.cos(angle), -jnp.sin(angle)],
                     [0, jnp.sin(angle), jnp.cos(angle)]])
    return arr

def rotate_y(angle):
    ''' Rotation matrix about the y-axis
    '''
    arr = jnp.array([[jnp.cos(angle), 0, jnp.sin(angle)],
                     [0, 1, 0],
                     [-jnp.sin(angle), 0, jnp.cos(angle)]])
    return arr

def rotate_z(angle):
    ''' Rotation matrix about the z-axis
    '''
    arr = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0],
                     [jnp.sin(angle), jnp.cos(angle), 0],
                     [0, 0, 1]])
    return arr
def euler_angles(coords, Omega, i, w):
    ''' This function rotates coordinates in 3D space using the Z-X-Z Euler Angle rotation https://en.wikipedia.org/wiki/Euler_angles
    This combination of rotations allows us to rotate completely in 3D space given 3 angles. 
    To do the correct rotation w.r.t the orbital elements, we need to rotate by the negative of each angle. Note the signs
    of the angles of the matrix representation in https://en.wikipedia.org/wiki/Orbital_elements#Euler_angle_transformations
    
    Parameters
    ----------
    coords : j/np.array
        3xN coordinates of particles, i.e. N particles in 3D space
    Omega : float
        Longitude of ascending node
    i : float
        Inclination
    w : float
        Argument of periapsis, (i.e. little omega)
    
    Returns
    -------
    j/np.array
        Rotated 3xN coordinate array
    '''
    return rotate_z(jnp.deg2rad(-Omega)) @ (
            rotate_x(jnp.deg2rad(-i)) @ (
            rotate_z(jnp.deg2rad(-w)) @ coords))

def inv_rotate_x(angle):
    return rotate_x(angle).T
def inv_rotate_y(angle):
    return rotate_y(angle).T
def inv_rotate_z(angle):
    return rotate_z(angle).T


### the following kepler solver functions are from https://jax.exoplanet.codes/en/latest/tutorials/core-from-scratch/#core-from-scratch

def kepler_starter(mean_anom, ecc):
    ome = 1 - ecc
    M2 = jnp.square(mean_anom)
    alpha = 3 * jnp.pi / (jnp.pi - 6 / jnp.pi)
    alpha += 1.6 / (jnp.pi - 6 / jnp.pi) * (jnp.pi - mean_anom) / (1 + ecc)
    d = 3 * ome + alpha * ecc
    alphad = alpha * d
    r = (3 * alphad * (d - ome) + M2) * mean_anom
    q = 2 * alphad * ome - M2
    q2 = jnp.square(q)
    w = jnp.square(jnp.cbrt(jnp.abs(r) + jnp.sqrt(q2 * q + r * r)))
    return (2 * r * w / (jnp.square(w) + w * q + q2) + mean_anom) / d
def kepler_refiner(mean_anom, ecc, ecc_anom):
    ome = 1 - ecc
    sE = ecc_anom - jnp.sin(ecc_anom)
    cE = 1 - jnp.cos(ecc_anom)

    f_0 = ecc * sE + ecc_anom * ome - mean_anom
    f_1 = ecc * cE + ome
    f_2 = ecc * (ecc_anom - sE)
    f_3 = 1 - f_1
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    d_42 = d_4 * d_4
    dE = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24)

    return ecc_anom + dE
@jnp.vectorize
def kepler_solver_impl(mean_anom, ecc):
    mean_anom = mean_anom % (2 * jnp.pi)

    # We restrict to the range [0, pi)
    high = mean_anom > jnp.pi
    mean_anom = jnp.where(high, 2 * jnp.pi - mean_anom, mean_anom)

    # Solve
    ecc_anom = kepler_starter(mean_anom, ecc)
    ecc_anom = kepler_refiner(mean_anom, ecc, ecc_anom)

    # Re-wrap back into the full range
    ecc_anom = jnp.where(high, 2 * jnp.pi - ecc_anom, ecc_anom)

    return ecc_anom
def kepler(mean_anom, ecc):
    ''' Kepler solver implemented in jaxoplanet. 
    https://jax.exoplanet.codes/en/latest/tutorials/core-from-scratch/
    Parameters
    ----------
    mean_anom : jnp.array
        Our mean anomalies that we want to solve for the eccentric and true anomaly
    ecc : jnp.array
        Array of 1 element, the eccentricity of the orbit
    Returns
    -------
    E : jnp.array
        The eccentric anomaly for each of the input mean anomalies
    nu : jnp.array
        The true anomaly for each of the input mean anomalies
    '''
    E = kepler_solver_impl(mean_anom, ecc)
    return E, 2 * jnp.arctan2(jnp.sqrt(1 + ecc) * jnp.sin(E / 2), jnp.sqrt(1 - ecc) * jnp.cos(E / 2))



def nonlinear_accel(x, t, rt, amax):
    xuse = x[0]
    sqrtx = jnp.sqrt(1 - rt/xuse) * jnp.heaviside(xuse - rt, 0)
    t_est = (xuse * sqrtx + rt * jnp.arctanh(sqrtx)) * AU2km / jnp.sqrt(2 * amax/yr2s * rt * AU2km)
    return jnp.abs(t_est - t)

def spin_orbit_mult(true_anom, direction, stardata):
    # derotate = rotate_x(jnp.deg2rad(stardata['spin_inc'])) @ (rotate_z(jnp.deg2rad(stardata['spin_Omega'])) @ direction)
    # inclination = jnp.arccos(derotate[2] / jnp.linalg.norm(derotate)) - jnp.pi / 2
    # open_angle_mult = 1 - stardata['spin_oa_mult'] * jnp.sin(inclination)**2
    # vel_mult = 1 + stardata['spin_vel_mult'] * jnp.sin(inclination)**2
    # return jnp.max(jnp.array([open_angle_mult, 0])), vel_mult
    
    inclination = jnp.pi / 2 + jnp.deg2rad(stardata['spin_inc']) * jnp.sin(true_anom - jnp.deg2rad(stardata['spin_Omega']))
    
    inclination = jnp.rad2deg(inclination)
    
    dist = jnp.min(jnp.abs(jnp.array([inclination - 180, inclination])))
    # dist = inclination - 90
    
    
    # gaussians for the open-angle/velocity-latitude curve
    spin_oa_sd = jnp.max(jnp.array([stardata['spin_oa_sd'], 0.01]))
    spin_vel_sd = jnp.max(jnp.array([stardata['spin_vel_sd'], 0.01]))
    open_angle_mult = 1 - stardata['spin_oa_mult'] * jnp.exp(- (dist / spin_oa_sd)**2)
    open_angle_mult = jnp.max(jnp.array([open_angle_mult, 0.001]))
    vel_mult = 1 + stardata['spin_vel_mult'] * jnp.exp(- (dist / spin_vel_sd)**2)
    
    
    # # test with a power law
    # x = jnp.abs(dist / 90 - 1)
    # spin_oa_sd = jnp.max(jnp.array([stardata['spin_oa_sd'], 0.001]))
    # spin_vel_sd = jnp.max(jnp.array([stardata['spin_vel_sd'], 0.001]))
    # open_angle_mult = 1 - stardata['spin_oa_mult'] * x**(1 / spin_oa_sd)
    # open_angle_mult = jnp.max(jnp.array([open_angle_mult, 0.001]))
    # vel_mult = 1 + stardata['spin_vel_mult'] * x**(1 / spin_vel_sd)

    
    # open_angle_mult = 1 
    # vel_mult = 1
    return open_angle_mult, vel_mult
    

def dust_circle(i_nu, stardata, theta, plume_direction, widths):
    ''' Creates a single ring of particles (a dust ring) in our dust plume. Applies weighting criteria as a proxy of 
    dust brightness or absence. 
    Parameters
    ----------
    i, nu : list of [int, float]
        i is the current ring number in our plume (e.g. the 1st generated ring will be i=0, the 10th generated ring will be i=9, etc)
        nu is the true anomaly value in radians
    stardata : dict
        Our dictionary of system parameters
    theta : j/np.array 
        1D array of length N (where N is the number of particles in one ring) that describe the angular positions of each particle
        w.r.t the center of the ring
    plume_direction : j/np.array
        3xNr array of delta positions, where Nr is the total number of rings in our model. 
        Array is calculated earlier on as pos1 - pos2, where pos1 is the main WR star position and pos2 is the binary position. 
        With this we isolate the [:, i]th element which encodes the 3D direction of our dust plume in cartesian coordinates [x, y, z]
    widths : j/np.array
        1D array of length Nr that describes how wide each ring should be. We access the ith element for this single ring. 
    
    Returns
    -------
    circle : Nx4 j/np.array
        First 3 axes correspond to the cartesian coordinates of each of the N particles in this ring. Units are in km from central binary barycenter. 
        Fourth axis corresponds to the weights of each particle for the histogram/imaging step. 
    '''
    i, nu = i_nu    # get our ring number, i, and the true anomaly, nu
    # x = nu / (2 * jnp.pi)       # convert true anomaly to proportion from 0 to 1
    # # now we need to ensure that the rings *smoothly* wrap around as the stars orbit each other
    # # we do this by shifting our true anomaly values from the range [0, 2pi] to [-pi, pi]
    # # if we don't do this, there's a discontinuity in the dust production at nu = 0
    # transf_nu = 2 * jnp.pi * (x + jnp.floor(0.5 - x))   # this is this transformed true anomaly 
    transf_nu = (nu - jnp.pi)%(2 * jnp.pi) - jnp.pi
    turn_on = jnp.deg2rad(stardata['turn_on'])          # convert our turn on true anomaly from degrees to radians
    turn_off = jnp.deg2rad(stardata['turn_off'])        # convert our turn off true anomaly from degrees to radians
    turned_on = jnp.heaviside(transf_nu - turn_on, 0)   # determine if our current true anomaly is greater than our turn on true anomaly (i.e. is dust production turned on?)
    # we can only visible dust if the ring is far enough away (past the nucleation distance), so we're not visibly turned on unless our ring is wider than this
    turned_on *= jnp.heaviside(widths[i] - stardata['nuc_dist'] * AU2km, 1)   # nucleation distance (no dust if less than nucleation dist), converted from AU to km
    turned_off = jnp.heaviside(turn_off - transf_nu, 0) # determine if our current true anomaly is less than our turn off true anomaly (i.e. is dust production still turned on?)
    
    direction = plume_direction[:, i] / jnp.linalg.norm(plume_direction[:, i])  # normalize our plume direction vector
    
    # oa_mult, v_mult = spin_orbit_mult(nu, direction, stardata)  # get the open angle and velocity multipliers for our current ring/true anomaly based on any wind anisotropy
    v_mult = oa_mult = 1
    # for the circle construction, we only use the half open angle
    half_angle = jnp.deg2rad(stardata['open_angle'] * oa_mult) / 2  # calculate the half open angle after multiplying by our open angle factor
    half_angle = jnp.min(jnp.array([half_angle, jnp.pi / 2]))

    # we also need to effectively dither our particle angular coordinate to reduce the effect of using a finite number of rings/particles on our final image
    shifted_theta = (theta + i)%(2*jnp.pi)   # since theta is in radians, we can just add our (integer) ring number which will somewhat randomly shift the data
    # now we construct our circle *along the x axis* (i.e. all circle points have the same x value, and only look like a circle when looked at in the y-z plane)
    # the stars are orbiting in the xy plane here, so z points out of the orbital plane
    # the below circle are the particle coordinates in cartesian coordinates, but not in meaningful units (yet)
    circle = jnp.array([jnp.ones(len(theta)) * jnp.cos(half_angle), 
                        jnp.sin(half_angle) * jnp.sin(shifted_theta), 
                        jnp.sin(half_angle) * jnp.cos(shifted_theta)])
    # circle = jnp.array([jnp.ones(len(theta)) * jnp.cos(half_angle), 
    #                     jnp.sin(half_angle) * jnp.sin(shifted_theta), 
    #                     (1 - stardata['oblate']) * jnp.sin(half_angle) * jnp.cos(shifted_theta)])
    
    
    ### below attempts to model latitude varying windspeed -- don't see this significantly in apep
    ### if you think about it, the CW shock occurs more or less around the equatorial winds so it shouldnt have a huge effect
    # latitude_speed_var = jnp.array([jnp.ones(len(theta)), 
    #                     jnp.ones(len(theta)), 
    #                     jnp.ones(len(theta)) * (1. + stardata['lat_v_var'] * jnp.cos(theta)**2)])
    # circle *= latitude_speed_var
    
    
    # circle *= widths[i]           # this is the width the circle should have assuming no velocity affecting effects
    spiral_time = widths[i] / stardata['windspeed1']    # our widths are calculated by w=v*t, so we can get the 'time' of the current ring by rearranging
    
    circle *= widths[i] * v_mult    # our circle should have the original width multiplied by our anisotropy multiplier
    
    
    # ------------------------------------------------------------------
    ### --- More work needed for dust circle acceleration --- ###
    # ### Below few lines handle acceleration of dust from radiation pressure -- only relevant when phase is tiny
    # # https://physics.stackexchange.com/questions/15587/how-to-get-distance-when-acceleration-is-not-constant
    # valid_dists = jnp.heaviside(stardata['opt_thin_dist'] - stardata['nuc_dist'], 1)
    # t_noaccel = stardata['nuc_dist'] * AU2km / stardata['windspeed1']
    # t_linear = 2 * jnp.sqrt(valid_dists * (stardata['opt_thin_dist'] - stardata['nuc_dist']) * AU2km / (2 * stardata['acc_max']/yr2s))
    # accel_lin = jnp.heaviside(spiral_time - t_noaccel, 0)
    # dist_accel_lin = accel_lin * 0.5 * stardata['acc_max']/yr2s * jnp.min(jnp.array([spiral_time, t_linear]))**2
    # # accel_r2 = jnp.heaviside(spiral_time - t_linear, 0)
    
    
    # ### much more work needed for nonlinear acceleration
    # # minim = minimize(nonlinear_accel, jnp.array([10*stardata['opt_thin_dist']]), args=(spiral_time - t_linear, stardata['opt_thin_dist'], stardata['acc_max']), method='BFGS', tol=1e-6)
    # # # print(minim.x)
    # # dist_accel_r2 = minim.x
    # # dist_accel_r2 *= accel_r2
    # # circle += valid_dists * (dist_accel_lin + dist_accel_r2)
    
    # # solver = jaxopt.GradientDescent(fun=nonlinear_accel, maxiter=200)
    # # res = solver.run(jnp.array([100*stardata['opt_thin_dist']]), t=spiral_time - t_linear, rt=stardata['opt_thin_dist'], amax=stardata['acc_max'])
    # # dist_accel_r2 = res.params[0]
    # # dist_accel_r2 *= accel_r2
    # dist_accel_r2 = 0
    # circle += valid_dists * (dist_accel_lin + dist_accel_r2)
    
    # ------------------------------------------------------------------
    
    ### now rotate the circle to account for the star orbit direction
    # remembering that the stars orbit in the x-y plane
    angle_x = jnp.arctan2(direction[1], direction[0]) + jnp.pi
    circle = rotate_z(angle_x) @ circle         # want to rotate the circle about the z axis
    
    weights = jnp.ones(len(theta)) * turned_on * turned_off
    
    # ------------------------------------------------------------------
    ## below accounts for the dust production not turning on/off instantaneously (probably negligible effect for most systems)
    # weights = jnp.ones(len(theta))
    sigma = jnp.deg2rad(stardata['gradual_turn'])
    sigma = jnp.max(jnp.array([sigma, 0.01]))
    # mult = 0.1
    # weights *= 1 - (1 - turned_on - mult * jnp.exp(-0.5 * ((transf_nu - turn_on) / sigma)**2))
    # weights *= 1 - (1 - turned_off - mult * jnp.exp(-0.5 * ((transf_nu - turn_off) / sigma)**2))
    
    residual_on = (1 - turned_on) * jnp.exp(-0.5 * ((transf_nu - turn_on) / sigma)**2)
    residual_off = (1 - turned_off) * jnp.exp(-0.5 * ((transf_nu - turn_off) / sigma)**2)
    residual = jnp.min(jnp.array([residual_on + residual_off, 1]))
    weights = weights + residual
    # ------------------------------------------------------------------
    
    ### Now we need to take into account the photodissociation effect from a ternary companion (specifically for Apep)
    # start by getting the inclination and azimuth of the companion
    alpha = jnp.deg2rad(stardata['comp_incl'])  
    beta = jnp.deg2rad(stardata['comp_az'])
    comp_halftheta = jnp.deg2rad(stardata['comp_open'] / 2) # as before, we use the half open angle in calculations
    x = circle[0, :]
    y = circle[1, :]
    z = circle[2, :]
    r = jnp.sqrt(x**2 + y**2 + z**2)
    particles_alpha = jnp.arccos(z / r)     # get the polar angle of the particles
    particles_beta = jnp.sign(y) * jnp.arccos(x / jnp.sqrt(x**2 + y**2))    # get the azimuthal angle of the particles
    
    ### to get angular separation of the points on the sphere, I used the cos(alpha) = ... formula from
    # https://www.atnf.csiro.au/people/Tobias.Westmeier/tools_spherical.php#:~:text=The%20angular%20separation%20of%20two%20points%20on%20a%20shpere&text=cos(%CE%B1)%3Dcos(%CF%911)cos(,%CF%861%E2%88%92%CF%862).
    term1 = jnp.cos(alpha) * jnp.cos(particles_alpha)
    term2 = jnp.sin(alpha) * jnp.sin(particles_alpha) * jnp.cos(beta - particles_beta)
    angular_dist = jnp.arccos(term1 + term2)
    
    photodis_prop = 1   # how much of the plume is photodissociated by the companion. set to < 1 if you want a another plume generated
    
    ## linear scaling for companion photodissociation
    # companion_dissociate = jnp.where(angular_dist < comp_halftheta,
    #                                   (1 - stardata['comp_reduction'] * jnp.ones(len(weights))), jnp.ones(len(weights)))
    # companion_dissociate = jnp.maximum(jnp.zeros(len(companion_dissociate)), companion_dissociate)
    
    ## gaussian scaling for companion photodissociation
    comp_gaussian = 1 - stardata['comp_reduction'] * jnp.exp(-(angular_dist / comp_halftheta)**2)
    comp_gaussian = jnp.maximum(comp_gaussian, jnp.zeros(len(comp_gaussian))) # need weight value to be between 0 and 1
    companion_dissociate = jnp.where(angular_dist < photodis_prop * comp_halftheta,
                                      comp_gaussian, jnp.ones(len(weights)))
    
    weights *= companion_dissociate         # this is us 'destroying' the particles
    
    
    # # ------------------------------------------------------------------
    # ## below code calculates another plume from the wind-companion interaction
    # ## currently is commented out to save on computation
    
    # # below populates companion plume with points taken from a narrow region around the ring edge
    # # in_comp_plume = jnp.where((photodis_prop * comp_halftheta < angular_dist) & (angular_dist < comp_halftheta),
    # #                           jnp.ones(len(x)), jnp.zeros(len(x)))
    
    # # below populates companion plume with points from the entire photodissociation region (also means that we can't have a semi-photodissociated region!!)
    # in_comp_plume = jnp.where(angular_dist < comp_halftheta, jnp.ones(len(x)), jnp.zeros(len(x)))
    # plume_weight = jnp.ones(len(x))
    
    # # now we need to generate angles around the plume edge that are inconsistent to the other rings so that it smooths out
    # # i.e. instead of doing linspace(0, 2*pi, len(x)), just do a large number multiplied by our ring number and convert that to [0, 2pi]
    # # ring_theta = jnp.linspace(0, i * len(x), len(x))%(2*jnp.pi)
    
    # # or instead use the below to put plume along the direction where there was already dust
    # az_circle = rotate_x(alpha) @ (rotate_z(beta) @ circle)
    # ring_theta = 3*jnp.pi/2 + jnp.sign(az_circle[1, :]) * jnp.arccos(az_circle[0, :] / jnp.sqrt(az_circle[0, :]**2 + az_circle[1, :]**2))
    
    # # or instead use the below to put the plume centered on a point with a gaussian fall off of angle
    # # ring_theta = jnp.linspace(1e-4, 1., len(x)) * 
    # ring_theta = jnp.linspace(0, i * len(x), len(x))%(2*jnp.pi)
    # comp_plume_max = stardata['comp_plume_max'] % 360.
    # val_comp_plume_sd = jnp.max(jnp.array([stardata['comp_plume_sd'], 0.01]))   # need to set a minimum azimuthal variation to avoid nans in the gradient
    # plume_particle_distance = jnp.minimum(abs(ring_theta * 180/jnp.pi - comp_plume_max), 
    #                                       abs(ring_theta * 180/jnp.pi - comp_plume_max + 360))
    # comp_plume_weights = jnp.exp(-0.5 * (plume_particle_distance / val_comp_plume_sd)**2)
    
    # # The coordinate transformations below are from user DougLitke from
    # # https://math.stackexchange.com/questions/643130/circle-on-sphere?newreg=42e38786904e43a0a2805fa325e52b92
    # new_x = r * (jnp.sin(comp_halftheta) * jnp.cos(alpha) * jnp.cos(beta) * jnp.cos(ring_theta) - jnp.sin(comp_halftheta) * jnp.sin(beta) * jnp.sin(ring_theta) + jnp.cos(comp_halftheta) * jnp.sin(alpha) * jnp.cos(beta))
    # new_y = r * (jnp.sin(comp_halftheta) * jnp.cos(alpha) * jnp.sin(beta) * jnp.cos(ring_theta) + jnp.sin(comp_halftheta) * jnp.cos(beta) * jnp.sin(ring_theta) + jnp.cos(comp_halftheta) * jnp.sin(alpha) * jnp.sin(beta))
    # new_z = r * (-jnp.sin(comp_halftheta) * jnp.sin(alpha) * jnp.cos(ring_theta) + jnp.cos(comp_halftheta) * jnp.cos(alpha))
    # x = x + in_comp_plume * (-x + new_x)
    # y = y + in_comp_plume * (-y + new_y)
    # z = z + in_comp_plume * (-z + new_z)
    
    # circle = jnp.array([x, y, z])
    
    # # weights *= (1 - in_comp_plume * (1 - stardata['comp_plume']))
    # weights *= (1 - in_comp_plume * (1 - stardata['comp_plume'] * comp_plume_weights))
    
    # ------------------------------------------------------------------
    
    # now calculate the weights of each point according the their orbital variation
    val_orb_sd = jnp.max(jnp.array([stardata['orb_sd'], 0.01]))     # need to set a minimum orbital variation to avoid nans in the gradient
    # we decide the weight multiplier accounting for orbital variation with a gaussian of the form
    # w_orb = 1 - (1 - A) * exp(((nu - min) / sd)^2)
    # that is, we take a weight of 1 (i.e. no change) as the baseline. Then we subtract off a maximum of (1 - A)*Gauss from this,
    # where A is the 'minimum' weighting value with our orbital variation accounted for, and Gauss is our gaussian function weighting 
    # which puts the minimum value at some true anomaly value and with a user defined standard deviation in this
    prop_orb = 1 - (1 - stardata['orb_amp']) * jnp.exp(-0.5 * (((transf_nu*180/jnp.pi + 180) - stardata['orb_min']) / val_orb_sd)**2) # weight proportion from orbital variation
    
    # now from azimuthal variation
    # this is analogous to the math for orbital variation, but instead of weighting entire rings based on the position in the orbit, 
    # we weight particles in the ring based on azimuthal variation in dust production
    val_az_sd = jnp.max(jnp.array([stardata['az_sd'], 0.01]))   # need to set a minimum azimuthal variation to avoid nans in the gradient
    prop_az = 1 - (1 - stardata['az_amp']) * jnp.exp(-0.5 * ((shifted_theta * 180/jnp.pi - stardata['az_min']) / val_az_sd)**2)
    
    # we need our orbital weighting proportion to be between 0 and 1
    prop_orb = jnp.min(jnp.array([prop_orb, 1]))
    prop_orb = jnp.max(jnp.array([prop_orb, 0]))
    # and the same for our azimuthal proportion
    prop_az = jnp.minimum(jnp.maximum(prop_az, jnp.zeros(len(prop_az))), jnp.ones(len(prop_az)))
    weights *= prop_orb * prop_az       # now scale the particle weights by our orbital/azimuthal variations
    
    
    # now set up our particles in the needed array format
    circle = jnp.array([circle[0, :], 
                        circle[1, :], 
                        circle[2, :],
                        weights])
    
    return circle

def calculate_semi_major(period_s, m1, m2):
    '''
    '''
    m1_kg = m1 * M_odot                                 # mass of stars in kg
    m2_kg = m2 * M_odot
    M_kg = m1_kg + m2_kg                                # total mass in kg
    # M = m1 + m2                                         # total mass in solar masses
    mu = G * M_kg
    a = jnp.cbrt((period_s / (2 * jnp.pi))**2 * mu)/1000    # semi-major axis of the system (total separation)
    a1 = m2_kg / M_kg * a                                   # semi-major axis of first body (meters)
    a2 = a - a1                                             # semi-major axis of second body
    return a1, a2



def dust_plume_sub(theta, times, n_orbits, period_s, stardata):
    
    n_time = len(times)
    n_t = int(n_time / n_orbits)
    ecc = stardata['eccentricity']
    # E, true_anomaly = kepler_solve(times, period_s, ecc)
    
    # E, true_anomaly = kepler(2 * jnp.pi * times / period_s, jnp.array([ecc]))
    
    # a1, a2 = calculate_semi_major(period_s, stardata['m1'], stardata['m2'])
    # r1 = a1 * (1 - ecc * jnp.cos(E)) * 1e-3     # radius in km 
    # r2 = a2 * (1 - ecc * jnp.cos(E)) * 1e-3
    # # ws_ratio = stardata['windspeed1'] / stardata['windspeed2']
    
    # positions1 = jnp.array([jnp.cos(true_anomaly), 
    #                         jnp.sin(true_anomaly), 
    #                         jnp.zeros(n_time)])
    # positions2 = jnp.copy(positions1)
    # positions1 *= r1      # position in the orbital frame
    # positions2 *= -r2     # position in the orbital frame
    
    # widths = stardata['windspeed1'] * period_s * (n_orbits - jnp.arange(n_time) / n_t)
    
    # plume_direction = positions1 - positions2               # get the line of sight from first star to the second in the orbital frame
    
        
    # particles = vmap(lambda i_nu: dust_circle(i_nu, stardata, theta, plume_direction, widths))((jnp.arange(n_time), true_anomaly))
    
    
    
    
    
    # ecc = 0. 
    ecc_factor = jnp.sqrt((1 - ecc) / (1 + ecc))
    
    # turn_on_true_anom = jnp.deg2rad(stardata['turn_on']) + jnp.pi 
    turn_on_true_anom = jnp.max(jnp.array([-180., stardata['turn_on'] - stardata['gradual_turn']]))
    turn_on_true_anom = (jnp.deg2rad(turn_on_true_anom))%(2. * jnp.pi) 
    # turn_on_ecc_anom = 2. * jnp.arctan(ecc_factor * jnp.tan(turn_on_true_anom / 2.))
    turn_on_ecc_anom = 2. * jnp.atan2(jnp.tan(turn_on_true_anom / 2.), 1./ecc_factor)
    turn_on_mean_anom = turn_on_ecc_anom - ecc * jnp.sin(turn_on_ecc_anom)
    
    # turn_on_mean_anom = jnp.atan2(-jnp.sqrt(1 - ecc**2) * jnp.sin(turn_on_true_anom), -ecc - jnp.cos(turn_on_true_anom)) + jnp.pi - ecc * (jnp.sqrt(1 - ecc**2) * jnp.sin(turn_on_true_anom)) / (1 + ecc * jnp.cos(turn_on_true_anom))
    
    # turn_off_true_anom = jnp.deg2rad(stardata['turn_off']) + jnp.pi 
    turn_off_true_anom = jnp.min(jnp.array([180., stardata['turn_off'] + stardata['gradual_turn']]))
    turn_off_true_anom = (jnp.deg2rad(turn_off_true_anom))%(2. * jnp.pi) 
    # turn_off_ecc_anom = 2. * jnp.arctan(ecc_factor * jnp.tan(turn_off_true_anom / 2.))
    turn_off_ecc_anom = 2. * jnp.atan2(jnp.tan(turn_off_true_anom / 2.), 1./ecc_factor)
    turn_off_mean_anom = turn_off_ecc_anom - ecc * jnp.sin(turn_off_ecc_anom)
    
    # turn_off_mean_anom = jnp.atan2(-jnp.sqrt(1 - ecc**2) * jnp.sin(turn_off_true_anom), -ecc - jnp.cos(turn_off_true_anom)) + jnp.pi - ecc * (jnp.sqrt(1 - ecc**2) * jnp.sin(turn_off_true_anom)) / (1 + ecc * jnp.cos(turn_off_true_anom))
    
    # print(turn_on_mean_anom)
    # print(turn_off_mean_anom)
    # mean_anomalies = jnp.linspace(turn_on_mean_anom, turn_off_mean_anom + 2 * jnp.pi, len(times))%(2 * jnp.pi)
    mean_anomalies = jnp.linspace(turn_on_mean_anom, turn_off_mean_anom, len(times))%(2 * jnp.pi)
    # print(mean_anomalies)
    
    E, true_anomaly = kepler(mean_anomalies, jnp.array([ecc]))
    
    a1, a2 = calculate_semi_major(period_s, stardata['m1'], stardata['m2'])
    r1 = a1 * (1 - ecc * jnp.cos(E)) * 1e-3     # radius in km 
    r2 = a2 * (1 - ecc * jnp.cos(E)) * 1e-3
    # ws_ratio = stardata['windspeed1'] / stardata['windspeed2']
    
    positions1 = jnp.array([jnp.cos(true_anomaly), 
                            jnp.sin(true_anomaly), 
                            jnp.zeros(n_time)])
    positions2 = jnp.copy(positions1)
    positions1 *= r1      # position in the orbital frame
    positions2 *= -r2     # position in the orbital frame
    
    # print(turn_on_mean_anom, turn_off_mean_anom)
    
    widths = stardata['windspeed1'] * period_s * (n_orbits - (jnp.linspace(turn_on_mean_anom, turn_off_mean_anom, len(times)) + jnp.pi) / (2 * jnp.pi))
    
    plume_direction = positions1 - positions2               # get the line of sight from first star to the second in the orbital frame
    
        
    particles = vmap(lambda i_nu: dust_circle(i_nu, stardata, theta, plume_direction, widths))((jnp.arange(n_time), true_anomaly))
    
    
    
    
    
    
    
    
    
    # mean_anomaly = jnp.deg2rad(jnp.linspace(stardata['turn_on'], stardata['turn_off'], n_t))# + jnp.pi
    # min_anomaly = jnp.deg2rad(stardata['turn_on']) + jnp.pi 
    # # mean_anomaly = min_anomaly + (times - )
    
    # max_anomaly = max(times)%period_s
    # # print(mean_anomaly)
    # E, true_anomaly = kepler(mean_anomaly, jnp.array([ecc]))
    # # E, true_anomaly = kepler(2 * jnp.pi * times[:n_t] / period_s, jnp.array([ecc]))
    # # E = jnp.linspace(jnp.deg2rad(stardata['turn_on']), jnp.deg2rad(stardata['turn_off']), n_t)
    # # true_anomaly = jnp.arccos((jnp.cos(E) - stardata['eccentricity']) / (1. - stardata['eccentricity'] * jnp.cos(E)))
    # # true_anomaly = jnp.repeat(true_anomaly, n_orbits)
    # # x = true_anomaly / (2 * jnp.pi)       # convert true anomaly to proportion from 0 to 1
    # # transf_nu = 2 * jnp.pi * (x + jnp.floor(0.5 - x))   # this is this transformed true anomaly 
    
    # # transf_turn_on = jnp.deg2rad(stardata['turn_on'])%(2 * jnp.pi) - jnp.pi
    # # transf_turn_off = jnp.deg2rad(stardata['turn_off'])%(2 * jnp.pi) - jnp.pi
    # # transf_turn_on = -jnp.deg2rad(stardata['turn_on']) + jnp.pi
    # # transf_turn_off = -jnp.deg2rad(stardata['turn_off']) + jnp.pi
    
    # # anom_min = jnp.min(jnp.array([transf_turn_on, transf_turn_off]))%(2 * jnp.pi) - jnp.pi
    # # anom_max = jnp.max(jnp.array([transf_turn_on, transf_turn_off]))%(2 * jnp.pi) - jnp.pi
    
    # # # transf_nu = (true_anomaly - jnp.pi)%(2 * jnp.pi) - jnp.pi
    # # # ring_anomalies = jnp.linspace(stardata['turn_on'], stardata['turn_off'], n_t)
    # # ring_anomalies = jnp.linspace(anom_min, anom_max, n_t)
    # # ring_anomalies = jnp.repeat(ring_anomalies, n_orbits)
    
    # # ring_anomalies = jnp.deg2rad(ring_anomalies) + jnp.pi
    
    # a1, a2 = calculate_semi_major(period_s, stardata['m1'], stardata['m2'])
    # r1 = a1 * (1 - ecc * jnp.cos(E)) * 1e-3     # radius in km 
    # r2 = a2 * (1 - ecc * jnp.cos(E)) * 1e-3
    # # ws_ratio = stardata['windspeed1'] / stardata['windspeed2']
    
    # positions1 = jnp.array([jnp.cos(true_anomaly), 
    #                         jnp.sin(true_anomaly), 
    #                         jnp.zeros(n_time)])
    # positions2 = jnp.copy(positions1)
    # positions1 *= r1      # position in the orbital frame
    # positions2 *= -r2     # position in the orbital frame
    
    # # widths = stardata['windspeed1'] * period_s * (n_orbits - jnp.arange(n_time) / n_t)
    # widths = stardata['windspeed1'] * period_s * (n_orbits - E / (2 * jnp.pi))
    
    # plume_direction = positions1 - positions2               # get the line of sight from first star to the second in the orbital frame
    
        
    # particles = vmap(lambda i_nu: dust_circle(i_nu, stardata, theta, plume_direction, widths))((jnp.arange(n_time), true_anomaly))







    weights = particles[:, 3, :].flatten()
    particles = particles[:, :3, :]
    
    
    particles = jnp.array([jnp.ravel(particles[:, 0, :]),
                           jnp.ravel(particles[:, 1, :]),
                           jnp.ravel(particles[:, 2, :])])
    

    ### the shock originates from the second star, not the WR, so we need to add its position to the spiral
    shock_start = positions2
    shock_start = jnp.repeat(shock_start, len(theta), axis=-1)
    particles += shock_start

    particles = euler_angles(particles, stardata['asc_node'], stardata['inclination'], stardata['arg_peri'])

    return 60 * 60 * 180 / jnp.pi * jnp.arctan(particles / (stardata['distance'] * 3.086e13)), weights

# @jit
def dust_plume(stardata):
    '''
    Parameters
    ----------
    stardata : dict
    '''
    phase = stardata['phase']%1
    
    period_s = stardata['period'] * 365.25 * 24 * 60 * 60
    
    n_orbits = 1
    n_t = 1000       # circles per orbital period
    n_points = 500   # points per circle
    n_particles = n_points * n_t * n_orbits
    n_time = n_t * n_orbits
    theta = 2 * jnp.pi * jnp.linspace(0, 1, n_points)
    times = period_s * jnp.linspace(phase, n_orbits + phase, n_time)
    particles, weights = dust_plume_sub(theta, times, n_orbits, period_s, stardata)
    return particles, weights
  
    
gui_funcs = [lambda stardata, i=i: dust_plume_GUI_sub(stardata, i) for i in range(1, 20)]
gui_funcs = [jit(gui_funcs[i]) for i in range(len(gui_funcs))]
def dust_plume_GUI_sub(stardata, n_orb):
    phase = stardata['phase']%1
    
    period_s = stardata['period'] * 365.25 * 24 * 60 * 60
    
    n_orbits = n_orb
    n_t = 1000       # circles per orbital period
    n_points = 400   # points per circle
    n_particles = n_points * n_t * n_orbits
    n_time = n_t * n_orbits
    theta = 2 * jnp.pi * jnp.linspace(0, 1, n_points)
    times = period_s * jnp.linspace(phase, n_orbits + phase, n_time)
    particles, weights = dust_plume_sub(theta, times, n_orbits, period_s, stardata)
    return particles, weights


def smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size):
    '''
    '''
    x = particles[0, :]
    y = particles[1, :]
    
    side_width = xedges[1] - xedges[0]
    
    xpos = x - jnp.min(xedges)
    ypos = y - jnp.min(yedges)
    
    x_indices = jnp.floor(xpos / side_width).astype(int)
    y_indices = jnp.floor(ypos / side_width).astype(int)
    
    alphas = xpos%side_width
    betas = ypos%side_width
    
    a_s = jnp.minimum(alphas, side_width - alphas) + side_width / 2
    b_s = jnp.minimum(betas, side_width - betas) + side_width / 2
    
    one_minus_a_indices = x_indices + jnp.where(alphas > side_width / 2, 1, -1)
    one_minus_b_indices = y_indices + jnp.where(betas > side_width / 2, 1, -1)
    
    one_minus_a_indices = one_minus_a_indices.astype(int)
    one_minus_b_indices = one_minus_b_indices.astype(int)
    
    # now check the indices that are out of bounds
    x_edge_check = jnp.heaviside(one_minus_a_indices, 1) * jnp.heaviside(im_size - one_minus_a_indices, 0)
    y_edge_check = jnp.heaviside(one_minus_b_indices, 1) * jnp.heaviside(im_size - one_minus_b_indices, 0)
    x_main_check = jnp.heaviside(x_indices, 1) * jnp.heaviside(im_size - x_indices, 0)
    y_main_check = jnp.heaviside(y_indices, 1) * jnp.heaviside(im_size - y_indices, 0)

    x_edge_check = x_edge_check.astype(int)
    x_main_check = x_main_check.astype(int)
    y_edge_check = y_edge_check.astype(int)
    y_main_check = y_main_check.astype(int)
    
    main_quadrant = a_s * b_s * weights * x_main_check * y_main_check
    horizontal_quadrant = (side_width - a_s) * b_s * weights * x_edge_check * y_main_check
    vertical_quadrant = a_s * (side_width - b_s) * weights * y_edge_check * x_main_check
    corner_quadrant = (side_width - a_s) * (side_width - b_s) * weights * x_edge_check * y_edge_check

    # The below few lines rely fundamentally on the following line sourced from https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html :
    # Unlike NumPy in-place operations such as x[idx] += y, if multiple indices refer to the same location, all updates will be applied (NumPy would only apply the last update, rather than applying all updates.)
    
    H = jnp.zeros((im_size, im_size))
    H = H.at[x_indices, y_indices].add(main_quadrant)
    H = H.at[one_minus_a_indices, y_indices].add(horizontal_quadrant)
    H = H.at[x_indices, one_minus_b_indices].add(vertical_quadrant)
    H = H.at[one_minus_a_indices, one_minus_b_indices].add(corner_quadrant)

    X, Y = jnp.meshgrid(xedges, yedges)
    H = H.T
    
    shape = 30 // 2  # choose just large enough grid for our gaussian
    gx, gy = jnp.meshgrid(jnp.arange(-shape, shape+1, 1), jnp.arange(-shape, shape+1, 1))
    gxy = jnp.exp(- (gx*gx + gy*gy) / (2 * stardata['sigma']**2))
    gxy /= gxy.sum()
    
    H = signal.convolve(H, gxy, mode='same', method='fft')
    
    H /= jnp.max(H)
    H = H**stardata['lum_power']
    H /= jnp.max(H)
    
    H = jnp.minimum(H, jnp.ones((im_size, im_size)) * stardata['histmax'])
    H /= jnp.max(H)
    
    return X, Y, H
n = 256
@jit
def smooth_histogram2d(particles, weights, stardata):
    im_size = n
    
    x = particles[0, :]
    y = particles[1, :]
    
    xbound, ybound = jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))
    bound = jnp.max(jnp.array([xbound, ybound])) * (1. + 2. / im_size)
    
    xedges, yedges = jnp.linspace(-bound, bound, im_size+1), jnp.linspace(-bound, bound, im_size+1)
    return smooth_histogram2d_base(particles, weights, stardata, xedges, yedges, im_size)
@jit
def smooth_histogram2d_w_bins(particles, weights, stardata, xbins, ybins):
    im_size = n
    return smooth_histogram2d_base(particles, weights, stardata, xbins, ybins, im_size)




@jit
def spiral_grid(particles, weights, stardata):
    ''' Takes in the particle positions and weights and calculates the 2D histogram, ignoring those points at (0,0,0), and
        applying a Gaussian blur.
    Parameters
    ----------
    particles : ndarray (Ndim, Nparticles)
        Particle positions in cartesian coordinates
    weights : array (Nparticles)
        Weight of each particle in the histogram (for orbital/azimuthal variations)
    sigma : 
    '''
    im_size = 256
    
    x = particles[0, :]
    y = particles[1, :]
    
    weights = jnp.where((x != 0) & (y != 0), weights, 0)
    
    H, xedges, yedges = jnp.histogram2d(x, y, bins=im_size, weights=weights)
    X, Y = jnp.meshgrid(xedges, yedges)
    H = H.T
    H /= jnp.max(H)
    
    H = jnp.minimum(H, jnp.ones((im_size, im_size)) * stardata['histmax'])
    
    shape = 30 // 2  # choose just large enough grid for our gaussian
    gx, gy = jnp.meshgrid(jnp.arange(-shape, shape+1, 1), jnp.arange(-shape, shape+1, 1))
    gxy = jnp.exp(- (gx*gx + gy*gy) / (2 * stardata['sigma']**2))
    gxy /= gxy.sum()
    
    H = signal.convolve(H, gxy, mode='same', method='fft')
    
    H /= jnp.max(H)
    H = H**stardata['lum_power']
    
    return X, Y, H
@jit
def spiral_grid_w_bins(particles, weights, stardata, xbins, ybins):
    ''' Takes in the particle positions and weights and calculates the 2D histogram, ignoring those points at (0,0,0), and
        applying a Gaussian blur.
    Parameters
    ----------
    particles : ndarray (Ndim, Nparticles)
        Particle positions in cartesian coordinates
    weights : array (Nparticles)
        Weight of each particle in the histogram (for orbital/azimuthal variations)
    sigma : 
    '''
    im_size = 256
    
    x = particles[0, :]
    y = particles[1, :]
    
    weights = jnp.where((x != 0) & (y != 0), weights, 0)
    
    H, xedges, yedges = jnp.histogram2d(x, y, bins=[xbins, ybins], weights=weights)
    X, Y = jnp.meshgrid(xedges, yedges)
    H = H.T
    H /= jnp.max(H)
    
    H = jnp.minimum(H, jnp.ones((im_size, im_size)) * stardata['histmax'])
    
    shape = 30 // 2  # choose just large enough grid for our gaussian
    gx, gy = jnp.meshgrid(jnp.arange(-shape, shape+1, 1), jnp.arange(-shape, shape+1, 1))
    gxy = jnp.exp(- (gx*gx + gy*gy) / (2 * stardata['sigma']**2))
    gxy /= gxy.sum()
    
    H = signal.convolve(H, gxy, mode='same', method='fft')
    
    H /= jnp.max(H)
    H = H**stardata['lum_power']
    
    return X, Y, H

def plot_spiral(X, Y, H):
    ''' Plots the histogram given by X, Y edges and H densities
    '''
    fig, ax = plt.subplots()
    ax.set_facecolor('k')
    ax.pcolormesh(X, Y, H, cmap='hot')
    # import matplotlib.colors as cols
    # ax.pcolormesh(X, Y, H, norm=cols.LogNorm(vmin=1, vmax=H.max()))
    # ax.pcolormesh(X, Y, H, norm=cols.PowerNorm(gamma=1/2), cmap='hot')
    ax.set(aspect='equal', xlabel='Relative RA (")', ylabel='Relative Dec (")')
    return ax


# @jit
def spiral_gif(stardata):
    '''
    '''
    starcopy = stardata.copy()
    fig, ax = plt.subplots()
    
    # im_size = 256
    # im = np.zeros((im_size, im_size))
    starcopy['phase'] = 0.01
    starcopy['sigma'] = 2
    particles, weights = dust_plume(stardata)
    X, Y, H = spiral_grid(particles, weights, starcopy)
    xmin, xmax = jnp.min(X), jnp.max(X)
    ymin, ymax = jnp.min(Y), jnp.max(Y)
    # border = [[xmin, xmax], [ymin, ymax]]
    # bins = [X, Y]
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal', 
           xlabel='Relative RA (")', ylabel='Relative Dec (")')
    
    starcopy['phase'] = 0.5
    particles, weights = dust_plume(starcopy)
    X, Y, H = spiral_grid(particles, weights, starcopy)
    # vmin, vmax = jnp.min(H), jnp.max(H)
    
    every = 1
    length = 10
    # now calculate some parameters for the animation frames and timing
    nt = int(stardata['period'])    # roughly one year per frame
    # nt = 10
    frames = jnp.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
    fps = len(frames) // length  # fps for the final animation
    
    phases = jnp.linspace(0, 1, nt)
    
    # @jit
    def animate(i):
        if (i // every)%20 == 0:
            print(f"{i // every} / {len(frames)}")
        # print(i)
        starcopy['phase'] = phases[i] + 0.5
        particles, weights = dust_plume(starcopy)
        X, Y, H = spiral_grid(particles, weights, starcopy)
        # ax.imshow(H, extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax, cmap='Greys')
        # ax.pcolormesh(xedges, yedges[::-1], H, vmax=vmax)
        ax.pcolormesh(X, Y, H, cmap='hot')
        return fig, 

    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
    ani.save(f"animation.gif", writer='pillow', fps=fps)
    
def plot_3d(particles, weights):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    n = 23
    ax.scatter(particles[0, ::n], particles[1, ::n], particles[2, ::n], marker='.', s=100, alpha=0.1)
    
def plot_orbit(stardata):
    ## plots orbits
    theta = np.linspace(0, 2 * np.pi, 100)
    r1 = stardata['p1'] / (1 + stardata['eccentricity'] * np.cos(theta))
    r2 = stardata['p2'] / (1 + stardata['eccentricity'] * np.cos(theta))

    x1, y1 = r1 * np.cos(theta), r1 * np.sin(theta)
    x2, y2 = -r2 * np.cos(theta), -r2 * np.sin(theta)

    fig, ax = plt.subplots()

    ax.plot(x1, y1)
    ax.plot(x2, y2)
    ax.set_aspect('equal')
    
    
def orbital_position(stardata):
    phase = stardata['phase']%1
    
    period_s = stardata['period'] * 365.25 * 24 * 60 * 60
    
    time = period_s * phase
    ecc = stardata['eccentricity']
    # E, true_anomaly = kepler_solve(times, period_s, ecc)
    
    E, true_anomaly = kepler(2 * jnp.pi * time / period_s, jnp.array([ecc]))
    
    a1, a2 = calculate_semi_major(period_s, stardata['m1'], stardata['m2'])
    r1 = a1 * (1 - ecc * jnp.cos(E)) * 1e-3     # radius in km 
    r2 = a2 * (1 - ecc * jnp.cos(E)) * 1e-3
    # ws_ratio = stardata['windspeed1'] / stardata['windspeed2']
    
    positions1 = jnp.array([jnp.cos(true_anomaly), 
                            jnp.sin(true_anomaly), 
                            [0]])
    positions2 = jnp.copy(positions1)
    positions1 *= r1      # position in the orbital frame
    positions2 *= -r2     # position in the orbital frame
    
    return positions1, positions2

@jit
def add_stars(xedges, yedges, H, stardata):
    ''' Superimposes the actual locations of the binary system stars onto the existing histogrammed image. 
    Also includes the third companion star for Apep.
    
    Parameters
    ----------
    xedges : j/np.array
        1x(im_size+1) length array with the border values of each histogram bin along the x axis
    yedges : j/np.array
        1x(im_size+1) length array with the border values of each histogram bin along the y axis
    H : j/np.array
        im_size x im_size array with the histogram values of each bin
    stardata : dict
        Our dictionary of system parameters
    
    Returns
    -------
    H : j/np.array
        The same H as input, but now with gaussians overlaid on the positions of each star in the system according to the 
        parameters in `stardata`
    '''
    # start by recreating the spatial grid of the H array
    bound = jnp.max(xedges)                                 # get max value in the grid
    bins = jnp.linspace(-bound, bound, len(xedges) - 1)     # set up our bin locations -- they'll be the same for both x and y if the smooth_histogram2d function was used to create the x/yedges arrays
    binx, biny = jnp.meshgrid(bins, bins)                   # set up the meshgrid for us to calculate the gaussians with
    
    pos1, pos2 = orbital_position(stardata)                 # now get the orbital positions of the two stars (in km from the inner binary barycenter)
    
    # now if we have a third star, we need to use the stardata parameters to determine its position
    star3dist = stardata['star3dist'] * AU2km               # get the dist in km (the value in the dict is in AU)
    incl, az = jnp.deg2rad(stardata['comp_incl']), jnp.deg2rad(stardata['comp_az']) # convert dict angular coordinates to radians
    # now get the cartesian coordinates of the third star from the spherical coordinates
    pos3 = star3dist * jnp.array([jnp.sin(incl) * jnp.cos(az),
                                  jnp.sin(incl) * jnp.sin(az),
                                  jnp.cos(incl)])
    # we now need to rotate according to the system geometry and then convert to an angular measurement from an absolute one
    pos1, pos2  = transform_orbits(pos1, pos2, stardata)
    pos3, _     = transform_orbits(pos3, jnp.zeros(3), stardata)    # can just ignore the 2nd star in the function by setting it to zeros
    
    # the spread of each star sprite is stored in a logarithmic value, so lets undo that now
    star1sd = 10**stardata['star1sd']
    star2sd = 10**stardata['star2sd']
    star3sd = 10**stardata['star3sd']
    
    # now finally spread the brightness of each star over a the bins
    gaussian_spread = lambda amp, pos, sd: amp * jnp.exp(-((binx - pos[0])**2 + (biny - pos[1])**2) / (2 * sd**2))  # 2d gaussian function for the xy plane of 3d pos data
    star1gaussian = gaussian_spread(stardata['star1amp'], pos1, star1sd)
    star2gaussian = gaussian_spread(stardata['star2amp'], pos2, star2sd)
    star3gaussian = gaussian_spread(stardata['star3amp'], pos3, star3sd)
    
    H = H + star1gaussian + star2gaussian + star3gaussian   # add the gaussians to the existing data
    H /= jnp.max(H)                                         # finally normalise the data again
    
    return H
    
    
    
    
def orbital_positions(stardata):
    
    phase = stardata['phase']%1
    
    period_s = stardata['period'] * 365.25 * 24 * 60 * 60
    
    n_orbits = 1
    n_t = 100       # circles per orbital period
    n_points = 40   # points per circle
    n_particles = n_points * n_t * n_orbits
    n_time = n_t * n_orbits
    theta = 2 * jnp.pi * jnp.linspace(0, 1, n_points)
    times = period_s * jnp.linspace(phase, n_orbits + phase, n_time)
    n_time = len(times)
    n_t = n_time / n_orbits
    ecc = stardata['eccentricity']
    # E, true_anomaly = kepler_solve(times, period_s, ecc)
    
    E, true_anomaly = kepler(2 * jnp.pi * times / period_s, jnp.array([ecc]))
    
    a1, a2 = calculate_semi_major(period_s, stardata['m1'], stardata['m2'])
    r1 = a1 * (1 - ecc * jnp.cos(E)) * 1e-3     # radius in km 
    r2 = a2 * (1 - ecc * jnp.cos(E)) * 1e-3
    # ws_ratio = stardata['windspeed1'] / stardata['windspeed2']
    
    positions1 = jnp.array([jnp.cos(true_anomaly), 
                            jnp.sin(true_anomaly), 
                            jnp.zeros(n_time)])
    positions2 = jnp.copy(positions1)
    positions1 *= r1      # position in the orbital frame
    positions2 *= -r2     # position in the orbital frame
    
    return positions1, positions2

def transform_orbits(pos1, pos2, stardata):
    pos1 = euler_angles(pos1, stardata['asc_node'], stardata['inclination'], stardata['arg_peri'])
    pos2 = euler_angles(pos2, stardata['asc_node'], stardata['inclination'], stardata['arg_peri'])
    pos1 = 60 * 60 * 180 / jnp.pi * jnp.arctan(pos1 / (stardata['distance'] * 3.086e13))
    pos2 = 60 * 60 * 180 / jnp.pi * jnp.arctan(pos2 / (stardata['distance'] * 3.086e13))
    return pos1, pos2

# @jit
def orbit_spiral_gif(stardata):
    '''
    '''
    @jit
    def dust_plume_for_gif(stardata):
        '''
        Parameters
        ----------
        stardata : dict
        '''
        phase = stardata['phase']%1
        
        period_s = stardata['period'] * 365.25 * 24 * 60 * 60
        
        n_orbits = 1
        n_t = 1000       # circles per orbital period
        n_points = 400   # points per circle
        n_particles = n_points * n_t * n_orbits
        n_time = n_t * n_orbits
        theta = 2 * jnp.pi * jnp.linspace(0, 1, n_points)
        times = period_s * jnp.linspace(phase, n_orbits + phase, n_time)
        particles, weights = dust_plume_sub(theta, times, n_orbits, period_s, stardata)
        return particles, weights
    starcopy = stardata.copy()
    fig, ax = plt.subplots(figsize=(6, 6))
    
    every = 1
    length = 10
    # now calculate some parameters for the animation frames and timing
    # nt = int(stardata['period'])    # roughly one year per frame
    nt = 100
    # nt = 10
    frames = jnp.arange(0, nt, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
    fps = len(frames) // length  # fps for the final animation
    
    phases = jnp.linspace(0, 1, nt)
    pos1, pos2 = orbital_positions(stardata)
    pos1, pos2 = transform_orbits(pos1, pos2, starcopy)
    
    
    lim = 2 * max(np.max(np.abs(pos1)), np.max(np.abs(pos2)))
    xbins = np.linspace(-lim, lim, 257)
    ybins = np.linspace(-lim, lim, 257)
    ax.set_aspect('equal')
    
    
    # @jit
    def animate(i):
        ax.cla()
        if i%20 == 0:
            print(i)
        starcopy['phase'] = phases[i] + 0.5
        particles, weights = dust_plume_for_gif(starcopy)
        
        pos1, pos2 = orbital_positions(starcopy)
        pos1, pos2 = transform_orbits(pos1, pos2, starcopy)

        X, Y, H = spiral_grid_w_bins(particles, weights, starcopy, xbins, ybins)
        ax.pcolormesh(X, Y, H, cmap='hot')
        
        
        ax.plot(pos1[0, :], pos1[1, :], c='w')
        ax.plot(pos2[0, :], pos2[1, :], c='w')
        ax.scatter([pos1[0, -1], pos2[0, -1]], [pos1[1, -1], pos2[1, -1]], c=['tab:cyan', 'w'], s=100)
        
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
        ax.set_facecolor('k')
        ax.set_axis_off()
        ax.text(0.3 * lim, -0.8 * lim, f"Phase = {starcopy['phase']%1:.2f}", c='w', fontsize=14)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        return fig, 

    ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
    ani.save(f"orbit_spiral.gif", writer='pillow', fps=fps)
    

def generate_lightcurve(stardata, n=100, shells=1):
    phases = jnp.linspace(0, 1, n)
    fluxes = np.zeros(n)
    
    for i in range(n):
        starcopy = stardata.copy()
        starcopy['phase'] = phases[i]
        
        particles, weights = gui_funcs[shells](starcopy)
        
        
        im_size = 256
        
        x = particles[0, :]
        y = particles[1, :]
        
        H, xedges, yedges = jnp.histogram2d(x, y, bins=im_size, weights=weights)
        X, Y = jnp.meshgrid(xedges, yedges)
        H = H.T
        
        H = jnp.minimum(H, jnp.ones((im_size, im_size)) * stardata['histmax'] * jnp.max(H))
        
        shape = 30 // 2  # choose just large enough grid for our gaussian
        gx, gy = jnp.meshgrid(jnp.arange(-shape, shape+1, 1), jnp.arange(-shape, shape+1, 1))
        gxy = jnp.exp(- (gx*gx + gy*gy) / (2 * stardata['sigma']**2))
        gxy /= gxy.sum()
        
        H = signal.convolve(H, gxy, mode='same', method='fft')
        
        # H = add_stars(X[0, :], Y[:, 0], H, starcopy)
        
        fluxes[i] = jnp.max(H)
        # fluxes[i] = np.percentile(H, 75)
    
    return phases, fluxes

system = wrb.apep.copy()
# system = wrb.WR140.copy()
# # apep['comp_reduction'] = 0
# # # # for i in range(10):
# # t1 = time.time()
particles, weights = dust_plume(system)
X, Y, H = smooth_histogram2d(particles, weights, system)
# print(time.time() - t1)
H = add_stars(X[0, :], Y[:, 0], H, system)
plot_spiral(X, Y, H)

# # # # plot_3d(particles, weights)


# # # # spiral_gif(apep)

# H_test = H.T.flatten()
# H_test = jnp.nan_to_num(H_test, 1e4)

# def test_function(params):
#     samp_particles, samp_weights = dust_plume(params)
#     _, _, samp_H = smooth_histogram2d(samp_particles, samp_weights, params)
#     samp_H = samp_H.flatten()
#     samp_H = jnp.nan_to_num(samp_H, 1e4)
#     return jnp.std(samp_H - H_test)

# test_grad = grad(test_function)

# for i in range(10):
#     t1 = time.time()
#     test_vals = test_grad(wrb.apep)
#     print(time.time() - t1)

# test_vals_arr = [test_vals[i] for i in test_vals.keys()]

# assert np.all(np.isfinite(test_vals_arr))






# wr112 = wrb.WR112.copy()
# wr112['phase'] = 0.47948
# particles, weights = gui_funcs[10](wrb.WR112)
# X, Y, H = smooth_histogram2d(particles, weights, wrb.WR112)
# plot_spiral(X, Y, H)

# np.savetxt('particles.csv', particles, delimiter=',')
# np.savetxt('weights.csv', weights, delimiter=',')

# particles = np.array(particles)
# weights = np.array(weights)

# import pickle

# with open('particles.pickle', 'wb') as handle:
#     pickle.dump(particles, handle)

# with open('weights.pickle', 'wb') as handle:
#     pickle.dump(weights, handle)






# orbit_spiral_gif(wrb.test_system)

# test_48a = wrb.WR48a.copy()
# # test_48a['hist_max'] = 1.
# # test_48a['eccentricity'] = 0.26
# # test_48a['star2amp'] = 3.

# phases, fluxes = generate_lightcurve(wrb.WR140, shells=2)
# shift = -0.1 
# shift = 0
# fig, ax = plt.subplots()
# ax.scatter((phases+shift)%1, np.log(fluxes))











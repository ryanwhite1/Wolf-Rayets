# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:05:33 2024

@author: ryanw
"""

# below are rough params for Apep 
apep = {"m1":15.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0.7, 
        "inclination":23.8,      # degrees
        "asc_node":350.,           # degrees
        "arg_peri":10.6,         # degrees
        "open_angle":125.,       # degrees (full opening angle)
        "period":125.,           # years
        "distance":2400.,        # pc
        "windspeed1":700.,       # km/s
        "windspeed2":2400.,      # km/s
        "turn_on":-114.,         # true anomaly (degrees)
        "turn_off":145.,         # true anomaly (degrees)
        "oblate":0.,
        "nuc_dist":1., "opt_thin_dist":2.,           # nucleation and optically thin distance (AU)
        "acc_max":0.1,                                 # maximum acceleration (km/s/yr)
        "orb_sd":0., "orb_amp":0., "orb_min":180., "az_sd":30., "az_amp":0.5, "az_min":270.,
        "comp_incl":55.6, "comp_az":68.8, "comp_open":90., "comp_reduction":1.09, "comp_plume":1.,
        "phase":0.6, 
        "sigma":2.,              # sigma for gaussian blur
        "histmax":1.}

# below are rough params for WR 48a
WR48a = {"m1":15.,                  # solar masses
        "m2":10.,                   # solar masses
        "eccentricity":0.1, 
        "inclination":75,           # degrees
        "asc_node":0,               # degrees
        "arg_peri":20,              # degrees
        "open_angle":110,           # degrees (full opening angle)
        "period":32.5,              # years
        "distance":3500,            # pc
        "windspeed1":700,           # km/s
        "windspeed2":2400,          # km/s
        "turn_on":-140,             # true anomaly (degrees)
        "turn_off":140,             # true anomaly (degrees)
        "oblate":0.,
        "nuc_dist":0.1, "opt_thin_dist":0.2,           # nucleation and optically thin distance (AU)
        "acc_max":1e-9,                               # maximum acceleration (km/s/yr)
        "orb_sd":0., "orb_amp":0., "orb_min":180, "az_sd":0., "az_amp":0., "az_min":270,
        "comp_incl":0, "comp_az":0, "comp_open":0, "comp_reduction":0., "comp_plume":0,
        "phase":0.6, 
        "sigma":2,                  # sigma for gaussian blur
        "histmax":1}


# below are rough params for WR 112
WR112 = {"m1":15.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0., 
        "inclination":100.,       # degrees
        "asc_node":75.,         # degrees
        "arg_peri":170.,           # degrees
        "open_angle":110.,       # degrees (full opening angle)
        "period":19,           # years
        "distance":2400,        # pc
        "windspeed1":700,       # km/s
        "windspeed2":2400,      # km/s
        "turn_on":-180,         # true anomaly (degrees)
        "turn_off":180,         # true anomaly (degrees)
        "oblate":0.,
        "nuc_dist":0.1, "opt_thin_dist":0.2,           # nucleation and optically thin distance (AU)
        "acc_max":1e-9,                               # maximum acceleration (km/s/yr)
        "orb_sd":0., "orb_amp":0., "orb_min":180, "az_sd":0., "az_amp":0., "az_min":270, 
        "comp_incl":0, "comp_az":0, "comp_open":0, "comp_reduction":0., "comp_plume":0,
        "phase":0.6, 
        "sigma":2,              # sigma for gaussian blur
        "histmax":1}

# below are rough params for WR 140
WR140 = {"m1":8.4,                # solar masses
        "m2":20,                # solar masses
        "eccentricity":0.8964, 
        "inclination":119.6,       # degrees
        "asc_node":0,         # degrees
        "arg_peri":180-46.8,           # degrees
        "open_angle":80,       # degrees (full opening angle)
        "period":2896.35/365.25,           # years
        "distance":1670,        # pc
        "windspeed1":2600,       # km/s
        "windspeed2":2400,      # km/s
        "turn_on":-135,         # true anomaly (degrees)
        "turn_off":135,         # true anomaly (degrees)
        "oblate":0.,
        "nuc_dist":50., "opt_thin_dist":220.,           # nucleation and optically thin distance (AU)
        "acc_max":900,                               # maximum acceleration (km/s/yr)
        "orb_sd":80., "orb_amp":0., "orb_min":180, "az_sd":60., "az_amp":0., "az_min":270,
        "comp_incl":0, "comp_az":0, "comp_open":0, "comp_reduction":0., "comp_plume":0,
        "phase":0.6, 
        "sigma":2,              # sigma for gaussian blur
        "histmax":1}

# below are rough params for WR 104
WR104 = {"m1":10,                # solar masses
        "m2":20,                # solar masses
        "eccentricity":0.06, 
        "inclination":180-15,       # degrees
        "asc_node":90,         # degrees
        "arg_peri":0,           # degrees
        "open_angle":60,       # degrees (full opening angle)
        "period":241.5/365.25,           # years
        "distance":2580,        # pc
        "windspeed1":1200,       # km/s
        "windspeed2":2000,      # km/s
        "turn_on":-180,         # true anomaly (degrees)
        "turn_off":180,         # true anomaly (degrees)
        "oblate":0.,
        "nuc_dist":0.1, "opt_thin_dist":0.2,           # nucleation and optically thin distance (AU)
        "acc_max":1e-9,                               # maximum acceleration (km/s/yr)
        "orb_sd":0., "orb_amp":0., "orb_min":180, "az_sd":0., "az_amp":0., "az_min":270, 
        "comp_incl":0, "comp_az":0, "comp_open":0, "comp_reduction":0., "comp_plume":0,
        "phase":0.7, 
        "sigma":6,              # sigma for gaussian blur
        "histmax":0.2}
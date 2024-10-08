# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:05:33 2024

@author: ryanw
"""

# below are rough params for Apep 
apep = {"m1":15.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0.82, 
        "inclination":23.8,      # degrees
        "asc_node":164.1,        # degrees
        "arg_peri":10.6,         # degrees
        "open_angle":127.,       # degrees (full opening angle)
        "period":234.,           # years
        "distance":2400.,        # pc
        "windspeed1":1165.,       # km/s
        "windspeed2":2400.,      # km/s
        "turn_on":-108.,         # true anomaly (degrees)
        "turn_off":141.,         # true anomaly (degrees)
        "gradual_turn":0.1,       # gradual turn off/on (deg) -- standard deviation of gaussian fall off
        "oblate":0.,
        "nuc_dist":1., "opt_thin_dist":2.,           # nucleation and optically thin distance (AU)
        'term_windspeed':880, 'accel_rate':-2.02,
        "orb_sd":0., "orb_amp":0., "orb_min":180., "az_sd":30., "az_amp":0.5, "az_min":90.,
        "comp_incl":124., "comp_az":238.8, "comp_open":90., "comp_reduction":1.75, "comp_plume":1.,
        "comp_plume_sd":20., "comp_plume_max":373.,
        "phase":0.29, 
        "sigma":2.,              # sigma for gaussian blur
        "histmax":1., "lum_power":1, 
        "spin_inc":0., "spin_Omega":0.,
        "windspeed_polar":2400, "aniso_vel_mult":-6.2, "aniso_vel_power":3.53, "open_angle_polar":180, "aniso_OA_mult":-6.05, "aniso_OA_power":3.53,
        'star1amp':0.7, 'star1sd':-0.7, 'star2amp':0.7, 'star2sd':-0.7, 'star3amp':0.7, 'star3sd':-1.12, 'star3dist':1700}
apep_aniso = {"m1":15.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0.83, 
        "inclination":23.8,      # degrees
        "asc_node":164.1,        # degrees
        "arg_peri":10.6,         # degrees
        "open_angle":116.5,       # degrees (full opening angle)
        "period":220.6,           # years
        "distance":2400.,        # pc
        "windspeed1":956.,       # km/s
        "windspeed2":2400.,      # km/s
        "turn_on":-111.2,         # true anomaly (degrees)
        "turn_off":141.3,         # true anomaly (degrees)
        "gradual_turn":0.1,       # gradual turn off/on (deg) -- standard deviation of gaussian fall off
        "oblate":0.,
        "nuc_dist":1., "opt_thin_dist":2.,           # nucleation and optically thin distance (AU)
        'term_windspeed':1105., 'accel_rate':-2.11,
        "orb_sd":0., "orb_amp":0., "orb_min":180., "az_sd":30., "az_amp":0.5, "az_min":90.,
        "comp_incl":121.8, "comp_az":238.8, "comp_open":90., "comp_reduction":1.75, "comp_plume":1.,
        "comp_plume_sd":20., "comp_plume_max":373.,
        "phase":0.32, 
        "sigma":2.,              # sigma for gaussian blur
        "histmax":1., "lum_power":1, 
        "spin_inc":22.5, "spin_Omega":317.65, 
        'star1amp':0.7, 'star1sd':-0.7, 'star2amp':0.7, 'star2sd':-0.7, 'star3amp':0.7, 'star3sd':-1.12, 'star3dist':1700}
apep_aniso_decel = {"m1":15.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0.82, 
        "inclination":23.8,      # degrees
        "asc_node":164.1,        # degrees
        "arg_peri":10.6,         # degrees
        "open_angle":125,       # degrees (full opening angle)
        "period":242.6,           # years
        "distance":2400.,        # pc
        "windspeed1":1176.,       # km/s
        "windspeed2":2400.,      # km/s
        "turn_on":-108.5,         # true anomaly (degrees)
        "turn_off":141.3,         # true anomaly (degrees)
        "gradual_turn":0.1,       # gradual turn off/on (deg) -- standard deviation of gaussian fall off
        "oblate":0.,
        "nuc_dist":1., "opt_thin_dist":2.,           # nucleation and optically thin distance (AU)
        # 'term_windspeed':885., 'accel_rate':-2.25,
        'term_windspeed':925., 'accel_rate':-2.15,
        "orb_sd":0., "orb_amp":0., "orb_min":180., "az_sd":30., "az_amp":0.5, "az_min":90.,
        "comp_incl":124.4, "comp_az":238.8, "comp_open":90., "comp_reduction":1.75, "comp_plume":1.,
        "comp_plume_sd":20., "comp_plume_max":373.,
        "phase":0.27, 
        "sigma":2.,              # sigma for gaussian blur
        "histmax":1., "lum_power":1, 
        "spin_inc":22.5, "spin_Omega":196, 
        "windspeed_polar":2400, "aniso_vel_mult":-6.2, "aniso_vel_power":3.53, "open_angle_polar":180, "aniso_OA_mult":-6.05, "aniso_OA_power":3.53, 
        'star1amp':0.7, 'star1sd':-0.7, 'star2amp':0.7, 'star2sd':-0.7, 'star3amp':0.7, 'star3sd':-1.12, 'star3dist':1700}
apep_aniso_decel2 = {"m1":15.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0.82, 
        "inclination":25.,      # degrees
        "asc_node":169.,        # degrees
        "arg_peri":0.,         # degrees
        "open_angle":109.,       # degrees (full opening angle)
        "period":242.6,           # years
        "distance":2400.,        # pc
        "windspeed1":1176.,       # km/s
        "windspeed2":2400.,      # km/s
        "turn_on":-114.,         # true anomaly (degrees)
        "turn_off":137.6,         # true anomaly (degrees)
        "gradual_turn":0.1,       # gradual turn off/on (deg) -- standard deviation of gaussian fall off
        "oblate":0.,
        "nuc_dist":1., "opt_thin_dist":2.,           # nucleation and optically thin distance (AU)
        # 'term_windspeed':885., 'accel_rate':-2.25,
        'term_windspeed':925., 'accel_rate':-2.15,
        "orb_sd":0., "orb_amp":0., "orb_min":180., "az_sd":30., "az_amp":0.5, "az_min":90.,
        "comp_incl":124.4, "comp_az":233., "comp_open":90., "comp_reduction":1.75, "comp_plume":1.,
        "comp_plume_sd":20., "comp_plume_max":373.,
        "phase":0.27, 
        "sigma":2.,              # sigma for gaussian blur
        "histmax":1., "lum_power":1, 
        "spin_inc":22.5, "spin_Omega":286., 
        "windspeed_polar":2400, "aniso_vel_mult":-6.2, "aniso_vel_power":3.53, "open_angle_polar":180, "aniso_OA_mult":-6.05, "aniso_OA_power":3.53,
        'star1amp':0.7, 'star1sd':-0.7, 'star2amp':0.7, 'star2sd':-0.7, 'star3amp':0.7, 'star3sd':-1.12, 'star3dist':1700}
apep_aniso_accel = {"m1":15.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0.7, 
        "inclination":23.8,      # degrees
        "asc_node":164.1,        # degrees
        "arg_peri":10.6,         # degrees
        "open_angle":108.5,       # degrees (full opening angle)
        "period":181.,           # years
        "distance":2400.,        # pc
        "windspeed1":1075.,       # km/s
        "windspeed2":2400.,      # km/s
        "turn_on":-108.5,         # true anomaly (degrees)
        "turn_off":141.3,         # true anomaly (degrees)
        "gradual_turn":0.1,       # gradual turn off/on (deg) -- standard deviation of gaussian fall off
        "oblate":0.,
        "nuc_dist":1., "opt_thin_dist":2.,           # nucleation and optically thin distance (AU)
        # 'term_windspeed':885., 'accel_rate':-2.25,
        'term_windspeed':1190., 'accel_rate':-2.7,
        "orb_sd":0., "orb_amp":0., "orb_min":180., "az_sd":30., "az_amp":0.5, "az_min":90.,
        "comp_incl":124.4, "comp_az":238.8, "comp_open":90., "comp_reduction":1.75, "comp_plume":1.,
        "comp_plume_sd":20., "comp_plume_max":373.,
        "phase":0.41, 
        "sigma":2.,              # sigma for gaussian blur
        "histmax":1., "lum_power":1, 
        "spin_inc":22.5, "spin_Omega":317.65, 
        "windspeed_polar":2400, "aniso_vel_mult":-6.2, "aniso_vel_power":3.53, "open_angle_polar":180, "aniso_OA_mult":-6.05, "aniso_OA_power":3.53,
        'star1amp':0.7, 'star1sd':-0.7, 'star2amp':0.7, 'star2sd':-0.7, 'star3amp':0.7, 'star3sd':-1.12, 'star3dist':1700}

# below are rough params for WR 48a
WR48a = {"m1":15.,                  # solar masses
        "m2":10.,                   # solar masses
        "eccentricity":0.74, 
        "inclination":74.,           # degrees
        "asc_node":174.,               # degrees
        "arg_peri":124.,              # degrees
        "open_angle":37.,           # degrees (full opening angle)
        "period":32.5,              # years
        "distance":4000.,            # pc
        "windspeed1":2100.,           # km/s
        "windspeed2":900.,          # km/s
        "turn_on":-121.,             # true anomaly (degrees)
        "turn_off":137.,             # true anomaly (degrees)
        "gradual_turn":19.,
        "oblate":0.,
        "nuc_dist":0.1, "opt_thin_dist":0.2,           # nucleation and optically thin distance (AU)
        'term_windspeed':2100., 'accel_rate':-5.,
        "orb_sd":40., "orb_amp":0., "orb_min":180, "az_sd":45., "az_amp":0., "az_min":90,
        "comp_incl":0, "comp_az":0, "comp_open":0, "comp_reduction":0., "comp_plume":0,
        "comp_plume_sd":0., "comp_plume_max":0.,
        "phase":0.78, 
        "sigma":2,                  # sigma for gaussian blur
        "histmax":0.3, "lum_power":1., 
        "spin_inc":0., "spin_Omega":0., 
        "windspeed_polar":2400, "aniso_vel_mult":-6.2, "aniso_vel_power":3.53, "open_angle_polar":180, "aniso_OA_mult":-6.05, "aniso_OA_power":3.53,
        'star1amp':0., 'star1sd':-1., 'star2amp':0., 'star2sd':-1., 'star3amp':0., 'star3sd':-1., 'star3dist':0.}


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
        "gradual_turn":5.,       # gradual turn off/on (deg) -- standard deviation of gaussian fall off
        "oblate":0.,
        "nuc_dist":0.1, "opt_thin_dist":0.2,           # nucleation and optically thin distance (AU)
        'term_windspeed':880, 'accel_rate':-2.02,
        "orb_sd":0., "orb_amp":0., "orb_min":180, "az_sd":0., "az_amp":0., "az_min":90, 
        "comp_incl":0, "comp_az":0, "comp_open":0, "comp_reduction":0., "comp_plume":0,
        "comp_plume_sd":0., "comp_plume_max":0.,
        "phase":0.7, 
        "sigma":6,              # sigma for gaussian blur
        "histmax":0.2, "lum_power":1., 
        "spin_inc":0., "spin_Omega":0., 
        "windspeed_polar":2400, "aniso_vel_mult":-6.2, "aniso_vel_power":3.53, "open_angle_polar":180, "aniso_OA_mult":-6.05, "aniso_OA_power":3.53,
        'star1amp':0., 'star1sd':-1., 'star2amp':0., 'star2sd':-1., 'star3amp':0., 'star3sd':-1., 'star3dist':0.}

# below are rough params for WR 112
WR112 = {"m1":15.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0., 
        "inclination":100.,       # degrees
        "asc_node":360-75.,         # degrees
        "arg_peri":170.,           # degrees
        "open_angle":110.,       # degrees (full opening angle)
        "period":19,           # years
        "distance":2400,        # pc
        "windspeed1":700,       # km/s
        "windspeed2":2400,      # km/s
        "turn_on":-180,         # true anomaly (degrees)
        "turn_off":180,         # true anomaly (degrees)
        "gradual_turn":5.,       # gradual turn off/on (deg) -- standard deviation of gaussian fall off
        "oblate":0.,
        "nuc_dist":0.1, "opt_thin_dist":0.2,           # nucleation and optically thin distance (AU)
        'term_windspeed':880, 'accel_rate':-2.02,
        "orb_sd":0., "orb_amp":0., "orb_min":180, "az_sd":0., "az_amp":0., "az_min":90, 
        "comp_incl":0, "comp_az":0, "comp_open":0, "comp_reduction":0., "comp_plume":0,
        "comp_plume_sd":0., "comp_plume_max":0.,
        "phase":0.6, 
        "sigma":2,              # sigma for gaussian blur
        "histmax":0.03, "lum_power":1.3, 
        "spin_inc":0., "spin_Omega":0.,
        "windspeed_polar":2400, "aniso_vel_mult":-6.2, "aniso_vel_power":3.53, "open_angle_polar":180, "aniso_OA_mult":-6.05, "aniso_OA_power":3.53,
        'star1amp':0., 'star1sd':-1., 'star2amp':0., 'star2sd':-1., 'star3amp':0., 'star3sd':-1., 'star3dist':0.}

# below are rough params for WR 112
WR125 = {"m1":15.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0.29, 
        "inclination":87.,       # degrees
        "asc_node":233.,         # degrees
        "arg_peri":175.,           # degrees
        "open_angle":35.,       # degrees (full opening angle)
        "period":28.12,           # years
        "distance":2400,        # pc
        "windspeed1":2700,       # km/s
        "windspeed2":2400,      # km/s
        "turn_on":-82.,         # true anomaly (degrees)
        "turn_off":111.,         # true anomaly (degrees)
        "gradual_turn":1.,       # gradual turn off/on (deg) -- standard deviation of gaussian fall off
        "oblate":0.,
        "nuc_dist":0.1, "opt_thin_dist":0.2,           # nucleation and optically thin distance (AU)
        'term_windspeed':880, 'accel_rate':-2.02,
        "orb_sd":0., "orb_amp":0., "orb_min":180, "az_sd":0., "az_amp":0., "az_min":90, 
        "comp_incl":0, "comp_az":0, "comp_open":0, "comp_reduction":0., "comp_plume":0,
        "comp_plume_sd":0., "comp_plume_max":0.,
        "phase":0.6, 
        "sigma":2,              # sigma for gaussian blur
        "histmax":0.18, "lum_power":1.3, 
        "spin_inc":0., "spin_Omega":0., 
        "windspeed_polar":2400, "aniso_vel_mult":-6.2, "aniso_vel_power":3.53, "open_angle_polar":180, "aniso_OA_mult":-6.05, "aniso_OA_power":3.53,
        'star1amp':0., 'star1sd':-1., 'star2amp':0., 'star2sd':-1., 'star3amp':0., 'star3sd':-1., 'star3dist':0.}

# below are rough params for WR 137
WR137 = {"m1":10,                # solar masses
        "m2":20,                # solar masses
        "eccentricity":0.315, 
        "inclination":97.2,       # degrees
        "asc_node":117.91,         # degrees
        "arg_peri":0.6,           # degrees
        "open_angle":18.6,       # degrees (full opening angle)
        "period":13.1,           # years
        "distance":1941,        # pc
        "windspeed1":1700,       # km/s
        "windspeed2":2000,      # km/s
        "turn_on":-180,         # true anomaly (degrees)
        "turn_off":180,         # true anomaly (degrees)
        "gradual_turn":5.,       # gradual turn off/on (deg) -- standard deviation of gaussian fall off
        "oblate":0.,
        "nuc_dist":0.1, "opt_thin_dist":0.2,           # nucleation and optically thin distance (AU)
        'term_windspeed':880, 'accel_rate':-2.02,
        "orb_sd":0., "orb_amp":0., "orb_min":180, "az_sd":0., "az_amp":0., "az_min":90, 
        "comp_incl":0, "comp_az":0, "comp_open":0, "comp_reduction":0., "comp_plume":0,
        "comp_plume_sd":0., "comp_plume_max":0.,
        "phase":0.9, 
        "sigma":3,              # sigma for gaussian blur
        "histmax":1., "lum_power":1., 
        "spin_inc":0., "spin_Omega":0., 
        "windspeed_polar":2400, "aniso_vel_mult":-6.2, "aniso_vel_power":3.53, "open_angle_polar":180, "aniso_OA_mult":-6.05, "aniso_OA_power":3.53,
        'star1amp':0., 'star1sd':-1., 'star2amp':0., 'star2sd':-1., 'star3amp':0., 'star3sd':-1., 'star3dist':0.}


# below are rough params for WR 140
WR140 = {"m1":8.4,                # solar masses
        "m2":20,                # solar masses
        "eccentricity":0.8964, 
        "inclination":119.6,       # degrees
        "asc_node":275.,         # degrees
        "arg_peri":180-46.8,           # degrees
        "open_angle":80,       # degrees (full opening angle)
        "period":2896.35/365.25,           # years
        "distance":1670,        # pc
        "windspeed1":2600,       # km/s
        "windspeed2":2400,      # km/s
        "turn_on":-135,         # true anomaly (degrees)
        "turn_off":135,         # true anomaly (degrees)
        "gradual_turn":0.5,       # gradual turn off/on (deg) -- standard deviation of gaussian fall off
        "oblate":0.,
        "nuc_dist":50., "opt_thin_dist":220.,           # nucleation and optically thin distance (AU)
        "acc_max":900,                               # maximum acceleration (km/s/yr)
        'term_windspeed':880, 'accel_rate':-2.02,
        "orb_sd":80., "orb_amp":0., "orb_min":180, "az_sd":60., "az_amp":0., "az_min":90,
        "comp_incl":0, "comp_az":0, "comp_open":0, "comp_reduction":0., "comp_plume":0,
        "comp_plume_sd":0., "comp_plume_max":0.,
        "phase":0.6, 
        "sigma":2,              # sigma for gaussian blur
        "histmax":1, "lum_power":1., 
        "spin_inc":0., "spin_Omega":0.,
        "windspeed_polar":2400, "aniso_vel_mult":-6.2, "aniso_vel_power":3.53, "open_angle_polar":180, "aniso_OA_mult":-6.05, "aniso_OA_power":3.53,
        'star1amp':0., 'star1sd':-1., 'star2amp':0., 'star2sd':-1., 'star3amp':0., 'star3sd':-1., 'star3dist':0.}


test_system = {"m1":22.,                # solar masses
        "m2":10.,                # solar masses
        "eccentricity":0.5, 
        "inclination":60.,       # degrees
        "asc_node":254.1,         # degrees
        "arg_peri":10.6,           # degrees
        "open_angle":40.,       # degrees (full opening angle)
        "period":1.,           # years
        "distance":10.,        # pc
        "windspeed1":0.1,       # km/s
        "windspeed2":2400.,      # km/s
        "turn_on":-180.,         # true anomaly (degrees)
        "turn_off":180.,         # true anomaly (degrees)
        "gradual_turn":5.,       # gradual turn off/on (deg) -- standard deviation of gaussian fall off
        "oblate":0.,
        "nuc_dist":0.0001, "opt_thin_dist":2.,           # nucleation and optically thin distance (AU)
        'term_windspeed':880, 'accel_rate':-2.02,
        "orb_sd":0., "orb_amp":0., "orb_min":180., "az_sd":30., "az_amp":0., "az_min":270.,
        "comp_incl":127.1, "comp_az":116.5, "comp_open":0., "comp_reduction":0., "comp_plume":1.,
        "comp_plume_sd":0., "comp_plume_max":0.,
        "phase":0.6, 
        "sigma":1.5,              # sigma for gaussian blur
        "histmax":1., "lum_power":1., 
        "spin_inc":0., "spin_Omega":0., 
        "windspeed_polar":2400, "aniso_vel_mult":-6.2, "aniso_vel_power":3.53, "open_angle_polar":180, "aniso_OA_mult":-6.05, "aniso_OA_power":3.53,
        'star1amp':0., 'star1sd':-1., 'star2amp':0., 'star2sd':-1., 'star3amp':0., 'star3sd':-1., 'star3dist':0.}

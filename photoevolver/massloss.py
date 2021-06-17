"""
[photoevolver.massloss]

Includes several atmosoheric mass loss formulations
"""

import numpy as np
import astropy.constants as Const
import astropy.units as U


 
def _keyword_check(keywords, params):
    for f in keywords:
        if f not in params:
            raise KeyError(f"model parameter '{f}' undefined")

def _bound_check(bounds, params):
    for f in bounds:
        if f not in params: continue
        if not (bounds[f][0] <= params[f] <= bounds[f][1]):
            raise ValueError(f"model parameter '{f}' out of safe bounds ({bounds[f][0]},{bounds[f][1]})")


def salz16_beta(Fxuv, mp, rp):
    """
    Parameters
        Fxuv: erg/cm2/s
        mp: Earth masses
        rp: Earth radii
    """
    potential = Const.G.to('erg*m/g^2').value * mp * Const.M_earth.to('g').value / (rp * Const.R_earth.to('m').value)
    log_beta = -0.185*np.log10(potential) + 0.021*np.log10(Fxuv) + 2.42
    if log_beta < 0.0: log_beta = 0.0
    return 10**(log_beta)

def salz16_eff(mp, rp):
    """
    Parameters
        mp: Planet mass in Earth masses
        rp: Planet radius in Earth radii
    """
    potential = Const.G.to('erg*m/g^2').value * mp * Const.M_earth.to('g').value / (rp * Const.R_earth.to('m').value)
    v = np.log10(potential)
    if   ( v < 12.0):           log_eff = np.log10(0.23) # constant
    if   (12.0  < v <= 13.11):  log_eff = -0.44*(v-12.0) - 0.5
    elif (13.11 < v <= 13.6):   log_eff = -7.29*(v-13.11) - 0.98
    elif (v > 13.6):            log_eff = -7.29*(13.6-13.11) - 0.98 # stable atmospheres, no evaporation (< 1e-5)
    return 10**(log_eff)*5/4 # Correction evaporation efficiency to heating efficiency


def EnergyLimited(**kwargs):
    """
    Calculates the atmospheric mass loss rate driven by photoevaporation
    This is based on the energy balance between stellar influx and the potential of the planet.
    Sources: Watson et al (1981), Lecavelier des Etangs (2007), Erkaev (2007).

    Required keywords:
        mass: planet M_earth
        radius: planet R_earth
        Lxuv: XUV luminosity of the star in erg/s
        dist: planet-star separation in AU
        mstar: M_sun

    Optional keywords:
        safe: checks if the input parameters are within safe model bounds.
        eff: mass loss efficiency. Use value (e.g. 0.15 for 15%) or formulation: 'salz16'.
        beta: XUV radius to optical radius ratio.

    Returns:
        mloss: mass loss rate (M_earth per Myr)

    """
    # --
    req_kw = ['radius', 'mass', 'Lxuv', 'dist', 'mstar']
    _keyword_check(req_kw, kwargs)
    # --
    bounds = {
            "radius": [0.5,   50.0],
            "mass":   [0.5, 20.0],
            "Lxuv":   [1.0, 1e38],
            "dist":   [0.01,   100.0],
            "mstar":  [0.5, 2.5]
    }
    if 'safe' in kwargs and kwargs['safe'] is True: _bound_check(bounds, kwargs)
    # --
    # Unit conversions
    kwargs['Lxuv']   *= 1e-7 # erg/s to Watt
    kwargs['mstar']  *= Const.M_sun.value # M_sun to kg
    kwargs['mass']   *= Const.M_earth.value # M_earth to kg
    kwargs['radius'] *= Const.R_earth.value # R_earth to m
    kwargs['dist']   *= Const.au.value # AU to m
    Fxuv = kwargs['Lxuv'] / ( 4 * np.pi * (kwargs['dist'])**2 )
    # Variable efficiency and Rxuv
    if 'eff' not in kwargs: kwargs['eff'] = 0.15
    elif kwargs['eff'] == 'salz16': kwargs['eff'] = salz16_eff(kwargs['mass']/Const.M_earth.value, kwargs['radius']/Const.R_earth.value)
    elif type(kwargs['eff']) is str: kwargs['eff'] = 0.15

    if 'beta' not in kwargs: kwargs['beta'] = 1.0
    elif kwargs['beta'] == 'salz16':
        kwargs['beta'] = salz16_beta(Fxuv*1e3, kwargs['mass']/Const.M_earth.value, kwargs['radius']/Const.R_earth.value)
    elif type(kwargs['beta']) is str: kwargs['beta'] = 1.0
    # Energy-limited equation
    xi =( kwargs['dist'] / kwargs['radius'] ) * ( kwargs['mass'] / kwargs['mstar'] / 3)**(1/3)
    K_tide = 1 - 3/(2*xi) + 1/(2*(xi)**3) 
    mloss = kwargs['beta']**2 * kwargs['eff'] * np.pi * Fxuv * kwargs['radius']**3 / (Const.G.value * K_tide * kwargs['mass'])
    return mloss * 5.28e-12 # Earth masses per Myr



def Kubyshkina18(**kwargs):
    """
    Calculates the atmospheric mass loss rate driven by photoevaporation
    This is based on the hydrodynamic models by Kubyshkina et al (2018)

    Required keywords:
        mass: planet M_earth
        radius: planet R_earth
        Lxuv: XUV luminosity of the star in erg/s
        dist: planet-star separation in AU

    Optional keywords:
        safe: (bool) checks if the input parameters are within safe model bounds.

    Returns:
        mloss: mass loss rate (M_earth per Myr)

    """
    # --
    req_kw = ['mass', 'radius', 'Lxuv', 'Lbol', 'dist']
    _keyword_check(req_kw, kwargs)
    # --
    bounds = {
            "radius": [1.0,  39.0],
            "mass":   [1.0,  10.0],
            "Lxuv":   [1e26, 5e30],
            "Lbol":   [1.0,  1e50],
            "dist":   [0.02, 1.3]
    }
    if 'safe' in kwargs and kwargs['safe'] is True: _bound_check(bounds, kwargs)
    # --
    # Constants and parameters
    large_delta = {
        'beta':  16.4084,
        'alpha': [1.0, -3.2861, 2.75],
        'zeta':  -1.2978,
        'theta': 0.8846
    }
    small_delta = {
        'beta': 32.0199,
        'alpha': [0.4222, -1.7489, 3.7679],
        'zeta': -6.8618,
        'theta': 0.0095
    }

    def Epsilon(rp, Fxuv, dist):
        numerator = 15.611 - 0.578*np.log(Fxuv) + 1.537*np.log(dist) + 1.018*np.log(rp)
        denominator = 5.564 + 0.894*np.log(dist)
        return numerator / denominator

    mp = kwargs['mass']
    rp = kwargs['radius']
    Lxuv = kwargs['Lxuv']
    Lbol = kwargs['Lbol']
    dist = kwargs['dist']

    conv = (U.erg / U.cm**2 / U.s).to('W/m^2') # erg/cm^2/s to W/m^2
    Fxuv = Lxuv / (4 * np.pi * (dist*Const.au.to('cm').value)**2 )
    Fbol = Lbol / (4 * np.pi * (dist*Const.au.to('cm').value)**2 )
    Teq =  ( Fbol * conv / (4*Const.sigma_sb.value) )**(1/4)
    mH = Const.m_p.value +  Const.m_e.value # H-atom mass (kg)

    Jeans_param = Const.G.value * (mp*Const.M_earth.value) * (mH) / (Const.k_B.value * Teq * (rp*Const.R_earth.value) )
    eps = Epsilon(rp, Fxuv, dist) 
    xp = small_delta if Jeans_param < np.exp(eps) else large_delta
    Kappa = xp['zeta'] + xp['theta']*np.log(dist)
    mloss = np.exp(xp['beta']) * (Fxuv)**xp['alpha'][0] * (dist)**xp['alpha'][1] * (rp)**xp['alpha'][2] * (Jeans_param)**Kappa
    return mloss * 5.28e-12 * 1e-3 # g/s to Earth masses per Myr








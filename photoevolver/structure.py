
"""
[photoevolve.structure]

Define several models that relate the envelope mass fraction of a planet to its total radius.

Models:
    LopezFortney14
    ChenRogers16
    OwenWu17

"""

import numpy as np
import astropy.constants as Const
import astropy.units as U

from .EvapMass.planet_structure import solve_structure

# Bolometric flux at Earth (W/m^2)
Fbol_earth = Const.L_sun.value / (4 * np.pi * (Const.au.value)**2 )
ergcm2s_to_Wm2 = ( U.erg/U.s/U.cm**2 ).to('W/m^2')


def _keyword_check(keywords, params):
    for f in keywords:
        if f not in params:
            raise KeyError(f"model parameter '{f}' undefined")

def _bound_check(bounds, params):
    for f in bounds:
        if f not in params: continue
        if not (bounds[f][0] <= params[f] <= bounds[f][1]):
            raise ValueError(f"model parameter '{f}' out of safe bounds ({bounds[f][0]},{bounds[f][1]})")

def LopezFortney14(**kwargs):
    """
    Calculates the envelope radius of a planet based on the thermal evolution model by Lopez & Fortney (2014)
    nasa/ads:

    Required keywords:
        mass: M_earth
        fenv: Envelope mass fraction (Menv / Mplanet)
        fbol: stellar bolometric flux at planet distance (erg/s/cm^2)
        age: in Myr

    Optional keywords:
        safe: checks if the input parameters are within safe model bounds

    Returns:
        renv: Envelope radius (R_earth)

    """
    # --
    req_kw = ['mass', 'fenv', 'fbol', 'age']
    _keyword_check(req_kw, kwargs)
    # --
    bounds = {
            "mass": [0.5,   20.0],
            "fenv": [1e-4,  0.2],
            "fbol": [0.1,   400.0],
            "age":  [100.0, 10000.0]
    }
    if 'safe' in kwargs and kwargs['safe'] is True: _bound_check(bounds, kwargs)
    # --
    mass_term = 2.06 * ( kwargs['mass'] )**(-0.21)
    flux_term = ( kwargs['fbol'] * ergcm2s_to_Wm2 / Fbol_earth)**(0.044)
    age_term  = ( kwargs['age'] /5000)**(-0.11) # t ** (-0.18) for enhanced opacity
    fenv_term = ( kwargs['fenv'] /0.05)**(0.59)
    renv = mass_term * fenv_term * flux_term * age_term
    return renv


def ChenRogers16(**kwargs):
    """
    Calculates the envelope radius of a planet based on the MESA model by Chen & Rogers (2016)
    nasa/ads:

    Required keywords:
        mass: M_earth
        fenv: Envelope mass fraction (Menv / Mplanet)
        fbol: stellar bolometric flux at planet distance (erg/s/cm^2)
        age: in Myr

    Optional keywords:
        safe: checks if the input parameters are within safe model bounds

    Returns:
        renv: Envelope radius (R_earth)

    """
    # --
    req_kw = ['mass', 'fenv', 'fbol', 'age']
    _keyword_check(req_kw, kwargs)
    # --
    bounds = {
            "mass": [0.5,   20.0],
            "fenv": [1e-4,  0.2],
            "fbol": [0.1,   400.0],
            "age":  [100.0, 10000.0]
    }
    if 'safe' in kwargs and kwargs['safe'] is True: _bound_check(bounds, kwargs)
    # --
    c0 = 0.131
    c1 = [-0.348, 0.631,  0.104, -0.179]

    c2 = [[ 0.209,  0.028, -0.168,  0.008],
            [ None,   0.086, -0.045, -0.036],
            [ None,   None,   0.052,  0.031],
            [ None,   None,   None,  -0.009]
    ]
    # --
    terms = [ \
        np.log10( kwargs['mass'] ),
        np.log10( kwargs['fenv'] /0.05),
        np.log10( kwargs['fbol'] * ergcm2s_to_Wm2 / Fbol_earth ),
        np.log10( kwargs['age']/5000 )
    ]
    # zeroth oder
    log_renv = c0
    # first order
    for i in range(len(terms)):
        log_renv += terms[i] * c1[i]
    # second order
    for i in range(len(terms)): 
        subterm = 0.0
        for j in range(len(terms)):
            if (j < i): continue
            subterm += c2[i][j] * terms[i] * terms[j]
        log_renv += subterm
    return 10**(log_renv)



def OwenWu17(**kwargs):
    """
    Calculates the envelope radius of a planet based on the MESA model by Chen & Rogers (2016)
    nasa/ads:

    Required keywords:
        mass: M_earth
        fenv: Envelope mass fraction (Menv / Mplanet)   
        fbol: stellar bolometric flux at planet distance (erg/s/cm^2)
        age: in Myr

    Optional keywords:
        Xiron: Iron mass fraction of the rocky core
        Xice: Ice mass fraction of the rocky core
        safe: checks if the input parameters are within safe model bounds

    Returns:
        renv: Envelope radius (R_earth)

    """
    # --
    req_kw = ['mass', 'fenv', 'fbol', 'age']
    _keyword_check(req_kw, kwargs)
    # --
    bounds = {
            "mass": [0.5,   20.0],
            "fenv": [1e-4,  0.2],
            "fbol": [0.1,   400.0],
            "age":  [100.0, 10000.0]
    }
    if 'safe' in kwargs and kwargs['safe'] is True: _bound_check(bounds, kwargs)
    # --
    Xenv = kwargs['fenv'] / (1 - kwargs['fenv'])
    Mcore = kwargs['mass'] * kwargs['fenv'] / Xenv
    Teq = ( kwargs['fbol'] * ergcm2s_to_Wm2 / (4*Const.sigma_sb.value) ) ** (1/4)
    Tkh = kwargs['age'] if kwargs['age'] > 100.0 else 100.0
    Xiron = kwargs['Xiron'] if 'Xiron' in kwargs else 1/3
    Xice = kwargs['Xice'] if 'Xice' in kwargs else 0.0
    # --
    ret = solve_structure(X = Xenv, Teq = Teq, Mcore = Mcore, Tkh_Myr = Tkh, Xiron = Xiron, Xice = Xice)
    #return Rrcb_sol, f, Rplanet, Rrcb_sol-Rcore
    return float( ret[2] - ret[0] + ret[3] ) * 1.56786e-9 # cm to Earth radius



def fenv_solve(fstruct, renv, mass, fbol, age, **kwargs):
    """
    Given a planet structure function (which calculates the envelope radius from the mass fraction), and a envelope radius, it solves the structure equation and returns the planet's envelope mass fraction.
    Parameters:
        fstruct:    (callable) Structure function
        renv:       (float) Envelope radius (Earth radii)
        mass:       (float) Planet mass (Earth masses)
        fbol:       (float) Stellar bolometric flux at the planet (erg/s/cm^2)
        age:        (float) Age of the planet in Myr.
        kwargs:     Keyword arguments to pass to structure function
    """
    fenv_guess = 0.01
    def wrapper(fenv, kwargs):
        kwargs['fenv'] = fenv
        kwargs['mass'] = kwargs['mcore'] / (1 - kwargs['fenv']) if kwargs['mass'] is None else kwargs['mass']
        return fstruct(**kwargs) - kwargs['renv']
    from scipy.optimize import fsolve
    kwargs['renv'] = renv
    kwargs['mass'] = mass
    kwargs['fbol'] = fbol
    kwargs['age'] = age
    solution = fsolve(func=wrapper, x0=fenv_guess, args=(kwargs) )
    return float(solution[0])




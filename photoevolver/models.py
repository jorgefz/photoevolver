
"""
File description
"""

import numpy as np
import uncertainties as uncert
from uncertainties import umath

from .evostate import EvoState
from . import physics, utils


def core_otegi20(state :EvoState, model_kw :dict) -> float|uncert.UFloat:
    """
    Calculates the radius of the planet's core using the relations
    by Otegi et a. (2020) based on an empirical fit to planet populations.
    
    Parameters
    ----------
    mcore   : float, core mass in Earth masses
    ot20_errors : bool, enables uncertainties

    Returns
    -------
    rcore   : float, core radius. It is of type uncertainties.ufloat
            if errors are enabled, and a python float otherwise.
    """
    bounds = [0.0, 100.0]
    if not state.mcore or umath.isnan(state.mcore) or state.mcore<=0:
        raise ValueError("Invalid core mass")
    scaling = uncert.ufloat(1.03, 0.02)
    exponent = uncert.ufloat(0.29, 0.01)
    rcore = scaling * state.mcore ** exponent
    if not model_kw.get('ot20_errors', False):
        return rcore.nominal_value
    return rcore


def envelope_lopez14(state :EvoState, model_kw :dict) -> float:
    """
    Returns the envelope thickness in Earth radii using the model
    by Lopez & Fortney (2014).
    Required parameters must be defined in the state,
    optional parameters can be provided through model keywords.
    
    Parameters
    ----------
    mass   : float, planet mass in Earth masses
    fenv   : float, envelope mass fraction = (mass-mcore)/mcore
    lbol   : float, bolometric luminosity of the hos star (erg/s/cm^2)
    sep    : float, orbital separation of the planet in AU
    age    : float, system age in Myr
    l14_opaque : bool (optional), enables enhanced opacity

    Returns
    -------
    renv   : float, envelope thickness in Earth radii
    """
    
    # bounds_check : bool (optional), ensures parameters are within model bounds
    bounds = {
        "mass": [0.5,   20.0],   # Earth mass
        "fenv": [1e-4,  0.2],    # mass/mcore
        "fbol": [0.1,   400.0],  # W/m^2
        "age":  [100.0, 10000.0] # Myr
    }
    
    age_power = -0.11 # solar metallicity
    if model_kw.get("lf14_opaque", False):
        age_power = -0.18 # enhanced opacity

    # Simulation uses fenv = (mass-mcore)/mcore
    # This model uses fenv2 = (mass-mcore)/mass
    fenv2 = state.fenv / (state.fenv + 1)
    fbol = physics.get_flux(lum=state.lbol, dist_au=state.sep)
    
    mass_term = 2.06 * ( state.mass )**(-0.21)
    flux_term = ( physics.SI_flux(fbol/physics.fbol_earth()) )**(0.044)
    age_term  = ( state.age/5000 )**(age_power)
    fenv_term = ( fenv2/0.05)**(0.59)
    renv = mass_term * fenv_term * flux_term * age_term
    return renv


def envelope_chen16(state :EvoState, model_kw :dict) -> float:
    """
    Returns the envelope thickness in Earth radii using the 
    analytical approximation to the MESA model by Chen & Rogers (2016).
    Required parameters must be defined in the state,
    optional parameters can be provided through model keywords.
    
    Parameters
    ----------
    mass   : float, planet mass in Earth masses
    fenv   : float, envelope mass fraction = (mass-mcore)/mcore
    lbol   : float, bolometric luminosity of the hos star (erg/s/cm^2)
    sep    : float, orbital separation of the planet in AU
    age    : float, system age in Myr
    c16_water : bool (optional), use coefficients for a water core.

    Returns
    -------
    renv   : float, envelope thickness in Earth radii
    """
    # TODO: fenv correction

    # bounds_check : bool (optional), ensures parameters are within model bounds
    bounds = {
        "mass": [0.5,   20.0],
        "fenv": [1e-4,  0.2],
        "fbol": [0.1,   400.0],
        "age":  [100.0, 10000.0]
    }

    # Coefficients
    c0 = 0.131
    c1 = np.array([-0.348, 0.631,  0.104, -0.179])
    c2 = np.array([
        [ 0.209, 0.028, -0.168,  0.008],
        [ 0.000, 0.086, -0.045, -0.036],
        [ 0.000, 0.000,  0.052,  0.031],
        [ 0.000, 0.000,  0.000, -0.009]
    ])
    if model_kw.get("c16_water", False):
        c0 = 0.169
        c1 = np.array([-0.436, 0.572,  0.154, -0.173])
        c2 = np.array([
            [ 0.246, 0.014, -0.210,  0.006],
            [ 0.000, 0.074, -0.048, -0.040],
            [ 0.000, 0.000,  0.059,  0.031],
            [ 0.000, 0.000,  0.000, -0.006]
        ])

    # Simulation uses fenv = (mass-mcore)/mcore
    # This model uses fenv2 = (mass-mcore)/mass
    fenv2 = state.fenv / (state.fenv + 1)
    fbol = physics.get_flux(lum=state.lbol, dist_au=state.sep)
    fbol_term = physics.SI_flux(fbol)/physics.SI_flux(physics.fbol_earth())

    terms = np.array([
        np.log10( state.mass ),
        np.log10( fenv2/0.05 ),
        np.log10( fbol_term ),
        np.log10( state.age/5000.0 )
    ])
    # zeroth oder
    log_renv :float = c0
    # first order
    log_renv += (terms*c1).sum()
    # second order
    log_renv += ((c2.T*terms).sum(axis=1)*terms).sum()
    return 10**log_renv


def envelope_owen17(state :EvoState, model_kw :dict) -> float:
    """
    Returns the envelope thickness in Earth radii using the 
    analytical model by Owen & Wu (2017).
    Required parameters must be defined in the state,
    optional parameters can be provided through model keywords.
    
    Parameters
    ----------
    mass   : float, planet mass in Earth masses
    fenv   : float, envelope mass fraction = (mass-mcore)/mcore
    lbol   : float, bolometric luminosity of the hos star (erg/s/cm^2)
    sep    : float, orbital separation of the planet in AU
    age    : float, system age in Myr
    
    Returns
    -------
    renv   : float, envelope thickness in Earth radii
    """
    raise NotImplementedError("envelope_owen17 not yet implemented")


def massloss_energy_limited(state :EvoState, model_kw :dict) -> float:
    """
    Returns the mass loss rate from a planet using the
    energy-limited model.
    Required parameters must be defined in the state,
    optional parameters can be provided through model keywords.

    Parameters
    ----------
    lxuv    : float, XUV luminosity from the star in erg/s
    mstar   : float, Host star mass in Solar masses
    mass    : float, planet mass in Earth masses
    radius  : float, planet radius in Earth radii
    sep     : float, orbital separation of the planet in AU
    el_eff  : float|callable (optional),
            evaporation efficiency. It may also be a function
            that calculates and returns the efficiency, based
            on the simulation state, with signature
            identical to this function.
    el_rxuv : float|callable (optional),
            ratio of XUV to optical planet radius.
            It may also be a function that calculates
            this parameter. See the description for `el_eff`.
    
    Returns
    -------
    mloss  : float, mass loss rate in g/s
    """
    lxuv   = (state.lx + state.leuv) * physics.units.erg.to('J')
    fxuv   = physics.get_flux(lum=state.lx+state.leuv, dist_au=state.sep)
    fxuv  *= physics.units.erg.to('J') / physics.units.cm.to('m')**2
    mstar  = state.mstar  * physics.constants.M_sun.value
    mass   = state.mass   * physics.constants.M_earth.value
    radius = state.radius * physics.constants.R_earth.value
    sep    = state.sep    * physics.units.au.to('m')
    eff    = model_kw.get("el_eff", 0.15)
    rxuv   = model_kw.get("el_rxuv", 1.0)
    if callable(eff):  eff  = eff(state, model_kw)
    if callable(rxuv): rxuv = rxuv(state, model_kw)
    grav_const = physics.constants.G.value
    
    xi = (sep/radius)*(mass/mstar/3)**(1/3)
    ktide = 1 - 3/(2*xi) + 1/(2*(xi)**3) # Correction for the Roche lobe
    mloss = (rxuv**2)*eff*np.pi*fxuv*(radius**3)/(grav_const*ktide*mass)
    return mloss * physics.units.kg.to('g')


def rxuv_salz16(state :EvoState, model_kw :dict) -> float:
    """
    Returns the ratio of XUV to optical radius for a planet's envelope
    using the model of Salz et al. (2016).
    Required parameters must be defined in the state,
    optional parameters can be provided through model keywords.

    Parameters
    ----------
    mass    : float, planet mass in Earth masses
    radius  : float, planet radius in Earth radii
    lxuv    : float, XUV luminosity from the star in erg/s
    sep     : float, orbital separation of the planet in AU
    """
    fxuv = physics.get_flux(lum=state.lx+state.leuv, dist_au=state.sep)
    grav_cgs = physics.constants.G.to('erg*m/g^2').value
    mass   = state.mass * physics.constants.M_earth.to('g').value
    radius = state.mass * physics.constants.R_earth.to('m').value
    gpot = grav_cgs * mass / radius
    log_beta = max(0.0, -0.185*np.log10(gpot) + 0.021*np.log10(fxuv) + 2.42)
    # upper limit to beta
    # if 10**log_beta > 1.05 and gpot < 1e12: log_beta = np.log10(1.05)
    return 10**(log_beta)


def efficiency_salz16(state :EvoState, model_kw :dict) -> float:
    """
    Returns the evaporation efficiency on a planet's envelope
    using the model of Salz et al. (2016).
    Required parameters must be defined in the state,
    optional parameters can be provided through model keywords.

    Parameters
    ----------
    mass    : float, planet mass in Earth masses
    radius  : float, planet radius in Earth radii
    """
    grav_cgs = physics.constants.G.to('erg*m/g^2').value
    mass   = state.mass * physics.constants.M_earth.to('g').value
    radius = state.mass * physics.constants.R_earth.to('m').value
    gpot = grav_cgs * mass / radius
    v = np.log10(potential)
    if   ( v < 12.0):           log_eff = np.log10(0.23) # constant
    if   (12.0  < v <= 13.11):  log_eff = -0.44*(v-12.0) - 0.5
    elif (13.11 < v <= 13.6):   log_eff = -7.29*(v-13.11) - 0.98
    elif (v > 13.6):            log_eff = -7.29*(13.6-13.11) - 0.98
    # for the last one, atmosphere is stable - no photevaporation
    return 10**(log_eff)*5/4 # Correction evaporation efficiency to heating efficiency


def massloss_salz16(state :EvoState, model_kw :dict) -> float:
    """
    Returns the mass loss rate from a planet using the
    energy-limited model with the approximation by
    Salz et al. (2016) for the XUV radius.
    
    Parameters
    ----------
    lxuv    : float, XUV luminosity from the star in erg/s
    mstar   : float, Host star mass in Solar masses
    mass    : float, planet mass in Earth masses
    radius  : float, planet radius in Earth radii
    sep     : float, orbital separation of the planet in AU
    
    Returns
    -------
    mloss  : float, mass loss rate in g/s
    """
    rxuv = rxuv_salz16(state, model_kw)
    kwargs = model_kw.copy()
    kwargs['el_rxuv'] = rxuv
    return massloss_energy_limited(state, kwargs)


@np.vectorize
def euv_king18(
        lx     : float,
        rstar  : float,
        energy : str = "0.1-2.4"
    ):
    """
    Calculates the EUV luminosity from the X-rays
    using the empirical relation by King et al. (2018).
    Source: https://ui.adsabs.harvard.edu/abs/2018MNRAS.478.1193K/abstract

    Parameters
    ----------
    lx      : float, X-ray luminosity of the star in erg/s
    rstar   : float, radius of the star in solar radii
    energy  : str, input energy range of the X-rays in keV.
            The allowed ranges are: 0.1-2.4, 0.124-2.48, 0.15-2.4,
            0.2-2.4, and 0.243-2.4 keV.

    Returns
    -------
    leuv    : float, the EUV luminosity of the star with energy between
            from 0.0136 keV and the lower range of the input X-rays.
    """
    # Model parameters
    params = {       # norm, exponent
        "0.1-2.4"   : [460,  -0.425],
        "0.124-2.48": [650,  -0.450],
        "0.15-2.4"  : [880,  -0.467],
        "0.2-2.4"   : [1400, -0.493],
        "0.243-2.4" : [2350, -0.539],
    }

    assert energy in params, (
        f"Energy range not covered by model.\n"
        +f"Options: {allowed_ranges.keys()}"
    )
    
    # This relation uses X-ray and EUV fluxes at the stellar surface.
    c, p = params[energy]
    conv = physics.constants.R_sun.to('cm').value
    fx_at_rstar = lx / (4.0 * np.pi * (rstar * conv)**2)
    feuv = c * (fx_at_rstar ** (p+1))  # EUV-to-Xrays flux ratio
    leuv = feuv * 4.0 * np.pi * (feuv * physics.units.cm.to('R_sun'))**2
    return leuv

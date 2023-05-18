
"""
File description
"""
import sys
import os
import contextlib
import numpy as np
import uncertainties as uncert
from uncertainties import umath
from . import physics, utils
from . import cmodels

def core_otegi20(
        mcore :float,
        ot20_errors :bool = False,
        **kwargs
    ) -> float|uncert.UFloat:
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
    if not mcore or umath.isnan(mcore) or mcore<=0:
        raise ValueError(f"[core_otegi20] Invalid core mass ({bounds})")
    scaling = uncert.ufloat(1.03, 0.02)
    exponent = uncert.ufloat(0.29, 0.01)
    if ot20_errors:
        rcore = scaling * mcore ** exponent
    else:
        rcore = scaling.n * mcore ** exponent.n
    return rcore


def core_fortney07(
        mcore     : float,
        ft07_iron : float = 1/3,
        ft07_ice  : float = 0.0,
        **kwargs
    ):
    """
    Calculates the radius of the planet's core using the relations
    by Fortney et a. (2007) for both rocky and icy cores.
    
    Parameters
    ----------
    mcore     : float, core mass in Earth masses
    ft07_iron : float (optional), iron mass fraction (0, 1)
    ft07_ice  : float (optional), ice mass fraction (0, 1).
        The iron and ice mass fractions cannot both be greater than zero.
        By default, an iron mass fraction of 1/3 is used.
        If the ice mass fraction is given a value above zero,
        a rock-ice composition will be used instead of a rock-iron composition.

    Returns
    -------
    rcore   : float, core radius.
    """
    iron_coeff = [
        [0.0592, 0.0975], # 2nd order -> (a * rock + b) * m**2
        [0.2337, 0.4938], # 1st order -> (a * rock + b) * m
        [0.3102, 0.7932]  # 0th order -> (a * rock + b)
    ]

    ice_coeff = [
        [0.0912, 0.1603], # 2nd order
        [0.3330, 0.7387], # 1st order
        [0.4639, 1.1193]  # 0th order
    ]

    if ft07_ice > 0.0:
        # ice relation
        coeff : list  = ice_coeff
        comp  : float = ft07_ice
    else:
        # iron relation (uses the rock mass fraction)
        coeff : list  = iron_coeff
        comp  : float = 1 - ft07_iron
    
    if comp < 0.0 or comp > 1.0:
        raise ValueError(
            "[core_fortney07] Invalid ice/iron mass fraction."
            + f"{comp} not in range (0,1)"
        )
    
    rcore = 0.0
    for i, c in enumerate(coeff):
        rcore += (c[0] * comp + c[1]) * np.log10(mcore) ** (2 - i)
    return rcore


def envelope_lopez14(
        mass :float,
        fenv :float,
        lbol :float,
        sep  :float,
        age  :float,
        lf14_opaque :bool = False,
        **kwargs
    ) -> float:
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
    if lf14_opaque:
        age_power = -0.18 # enhanced opacity
    
    # Simulation uses fenv = (mass-mcore)/mcore
    # This model uses fenv2 = (mass-mcore)/mass
    fenv2 = fenv / (fenv + 1)
    fbol = physics.get_flux(lum=lbol, dist_au=sep)
    
    mass_term = 2.06 * ( mass )**(-0.21)
    flux_term = ( physics.SI_flux(fbol/physics.fbol_earth()) )**(0.044)
    age_term  = ( age/5000 )**(age_power)
    fenv_term = ( fenv2/0.05)**(0.59)
    renv = mass_term * fenv_term * flux_term * age_term
    return renv


def envelope_chen16(
        mass :float,
        fenv :float,
        lbol :float,
        sep  :float,
        age  :float,
        cr16_water : bool = False,
        **kwargs
    ) -> float:
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
    cr16_water : bool (optional), use coefficients for a water core.

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
    if cr16_water:
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
    fenv2 = fenv / (fenv + 1)
    fbol = physics.get_flux(lum=lbol, dist_au=sep)
    fbol_term = physics.SI_flux(fbol)/physics.SI_flux(physics.fbol_earth())

    terms = np.array([
        np.log10( mass ),
        np.log10( fenv2/0.05 ),
        np.log10( fbol_term ),
        np.log10( age/5000.0 )
    ])
    # zeroth oder
    log_renv :float = c0
    # first order
    log_renv += (terms*c1).sum()
    # second order
    log_renv += ((c2.T*terms).sum(axis=1)*terms).sum()
    return 10**log_renv


def envelope_owen17(
        mass  :float,
        fenv  :float,
        lbol  :float,
        sep   :float,
        age   :float,
        rcore :float,
        **kwargs
    ) -> float:
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
    rcore  : float, core radius in Earth radii
    
    Returns
    -------
    renv   : float, envelope thickness in Earth radii
    """
    fbol = physics.get_flux(lum = lbol, dist_au = sep)
    fenv2 = physics.get_fenv_planet_ratio(fenv)
    return cmodels.envelope_owen17(
        mass = mass, fenv = fenv2, fbol = fbol, age = age, rcore = rcore
    )


def massloss_energy_limited(
    lx     :float,
    leuv   :float,
    mstar  :float,
    mass   :float,
    radius :float,
    sep    :float,
    el_eff  :float = 0.15,
    el_rxuv :float = 1.0,
    **kwargs
    ) -> float:
    """
    Returns the mass loss rate from a planet using the
    energy-limited model.
    Required parameters must be defined in the state,
    optional parameters can be provided through model keywords.

    Parameters
    ----------
    lx      : float, X-ray luminosity of the star in erg/s
    leuv    : float, X-ray luminosity of the star in erg/s
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
    lxuv   = (lx + leuv) * physics.units.erg.to('J')
    fxuv   = physics.get_flux(lum=lx+leuv, dist_au=sep)
    fxuv  *= physics.units.erg.to('J') / physics.units.cm.to('m')**2
    mstar  = mstar  * physics.constants.M_sun.value
    mass   = mass   * physics.constants.M_earth.value
    radius = radius * physics.constants.R_earth.value
    sep    = sep    * physics.units.au.to('m')
    kwargs.update(lx=lx, leuv=leuv, mstar=mstar, mass=mass, radius=radius, sep=sep)
    eff  = el_eff(**kwargs)  if callable(el_eff)  else el_eff
    rxuv = el_rxuv(**kwargs) if callable(el_rxuv) else el_rxuv
    grav_const = physics.constants.G.value
    
    xi = (sep/radius)*(mass/mstar/3)**(1/3)
    ktide = 1 - 3/(2*xi) + 1/(2*(xi)**3) # Correction for the Roche lobe
    mloss = (rxuv**2)*eff*np.pi*fxuv*(radius**3)/(grav_const*ktide*mass)
    return mloss * physics.units.kg.to('g')


def rxuv_salz16(
        mass   :float,
        radius :float,
        lx     :float,
        leuv   :float,
        sep    :float,
        **kwargs
    ) -> float:
    """
    Returns the ratio of XUV to optical radius for a planet's envelope
    using the model of Salz et al. (2016).
    Required parameters must be defined in the state,
    optional parameters can be provided through model keywords.

    Parameters
    ----------
    mass    : float, planet mass in Earth masses
    radius  : float, planet radius in Earth radii
    lx      : float, X-ray luminosity of the star in erg/s
    leuv    : float, X-ray luminosity of the star in erg/s
    sep     : float, orbital separation of the planet in AU
    """
    fxuv = physics.get_flux(lum=lx+leuv, dist_au=sep)
    grav_cgs = physics.constants.G.to('erg*m/g^2').value
    mass   = mass * physics.constants.M_earth.to('g').value
    radius = radius * physics.constants.R_earth.to('m').value
    gpot = grav_cgs * mass / radius
    log_beta = max(0.0, -0.185*np.log10(gpot) + 0.021*np.log10(fxuv) + 2.42)
    # upper limit to beta
    # if 10**log_beta > 1.05 and gpot < 1e12: log_beta = np.log10(1.05)
    return 10**(log_beta)


def efficiency_salz16(
        mass   :float,
        radius :float,
        **kwargs
    ) -> float:
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
    mass   = mass * physics.constants.M_earth.to('g').value
    radius = radius * physics.constants.R_earth.to('m').value
    gpot = grav_cgs * mass / radius
    v = np.log10(gpot)
    if   ( v < 12.0):           log_eff = np.log10(0.23) # constant
    if   (12.0  < v <= 13.11):  log_eff = -0.44*(v-12.0) - 0.5
    elif (13.11 < v <= 13.6):   log_eff = -7.29*(v-13.11) - 0.98
    elif (v > 13.6):            log_eff = -7.29*(13.6-13.11) - 0.98
    # for the last one, atmosphere is stable - no photevaporation
    return 10**(log_eff)*5/4 # Correction evaporation efficiency to heating efficiency


def massloss_salz16(
        lx     :float,
        leuv   :float,
        mstar  :float,
        mass   :float,
        radius :float,
        sep    :float,
        **kwargs
    ) -> float:
    """
    Returns the mass loss rate from a planet using the
    energy-limited model with the approximation by
    Salz et al. (2016) for the XUV radius.
    
    Parameters
    ----------
    lx      : float, X-ray luminosity of the star in erg/s
    leuv    : float, X-ray luminosity of the star in erg/s
    mstar   : float, Host star mass in Solar masses
    mass    : float, planet mass in Earth masses
    radius  : float, planet radius in Earth radii
    sep     : float, orbital separation of the planet in AU
    
    Returns
    -------
    mloss  : float, mass loss rate in g/s
    """
    kwargs.update(mass=mass, radius=radius, lx=lx, leuv=leuv,
        sep=sep, mstar=mstar)
    kwargs['el_rxuv'] = rxuv_salz16(**kwargs)
    return massloss_energy_limited(**kwargs)


def massloss_kubyshkina18(
        mass   :float,
        radius :float,
        leuv   :float,
        lbol   :float,
        sep    :float,
        mstar  :float,
        **kwargs
    ) -> float:
    """ Calculates the mass loss rate using the hydrodynamic models
    and interpolator by Kubyshkina et al (2018). """
    from .K18grid import interpolator
    # INTERPOL(Mss,EUV,T_i,r_i,m_i, dataset_file = None)
    ## INPUT: Mstar [Msun], EUV [erg/cm/s], Teq [K], Rpl [Re], Mpl [Me]
    feuv = physics.get_flux(leuv, dist_au=sep)
    fbol = physics.get_flux(lbol, dist_au=sep)
    teq  = physics.temp_eq(fbol)
    args = dict(Mss = mstar, EUV = feuv, T_i = teq, r_i = radius, m_i = mass)

    # Redirect stdout to avoid inteprolator messages clogging up buffer
    with utils.supress_stdout():
        result = interpolator.INTERPOL(**args)
    
    mloss = float(result)
    return mloss


def massloss_kubyshkina18_approx(
        mass   :float,
        radius :float,
        lx     :float,
        leuv   :float,
        lbol   :float,
        sep    :float,
        **kwargs
    ) -> float:
    """
    Calculates the atmospheric mass loss rate driven by photoevaporation
    This is based on the hydro-based expression by Kubyshkina et al (2018).

    Parameters
    ----------
        mass   : float, planet mass in Earth masses
        radius : float, planet radius in Earth radii
        lx      : float, X-ray luminosity of the star in erg/s
        leuv    : float, X-ray luminosity of the star in erg/s
        lbol   : float, bolometric luminosity in erg/s
        sep    : float, planet-star separation in AU
        
    Returns
    -------
        mloss  : float, mass loss rate in grams/sec

    """
    bounds = {
        "radius": [1.0,  39.0],
        "mass":   [1.0,  10.0],
        "lxuv":   [1e26, 5e30],
        "lbol":   [1.0,  1e40],
        "sep":    [0.002, 1.3]
    }

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

    def get_epsilon(radius:float, fxuv:float, sep:float) -> float:
        """ Calculates model parameter `Epsilon` """
        numerator = 15.611 - 0.578*np.log(fxuv)
        numerator += 1.537*np.log(sep) + 1.018*np.log(radius)
        denominator = 5.564 + 0.894*np.log(sep)
        return numerator / denominator
    
    lxuv = lx + leuv
    fxuv = physics.get_flux(lxuv, sep)

    jeans = physics.jeans_parameter(mass, radius, lbol, sep)
    eps = get_epsilon(radius, fxuv, sep) 
    par = small_delta if jeans < np.exp(eps) else large_delta
    kappa = par['zeta'] + par['theta'] * np.log(sep)

    mloss  =  np.exp(par['beta'])
    mloss *= (fxuv)**(par['alpha'][0])
    mloss *= (sep)**(par['alpha'][1])
    mloss *= (radius)**(par['alpha'][2])
    mloss *= (jeans)**kappa
    return mloss


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
            from 0.0136 keV and the lower range of the input X-rays in erg/s.
    """
    # Model parameters
    params = {       # norm, exponent
        "0.1-2.4"   : [460,  -0.425],
        "0.124-2.48": [650,  -0.450],
        "0.15-2.4"  : [880,  -0.467],
        "0.2-2.4"   : [1400, -0.493],
        "0.243-2.4" : [2350, -0.539],
    }

    if energy not in params:
        raise ValueError(
            f"Energy range not covered by model.\n"
            +f"Options: {list(params.keys())}"
        )
    
    # This relation uses X-ray and EUV fluxes at the stellar surface.
    c, p = params[energy]
    conv = physics.constants.R_sun.to('cm').value
    fx_at_rstar = lx / (4.0 * np.pi * (rstar * conv)**2)
    xratio = c * (fx_at_rstar ** p)  # EUV-to-Xrays flux ratio
    feuv_at_rstar = fx_at_rstar * xratio
    leuv = feuv_at_rstar * 4.0 * np.pi * (rstar * conv)**2
    return leuv

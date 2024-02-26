import numpy as np
from photoevolver import physics, utils
from .K18grid import interpolator as _k18_interpolator

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
    # INTERPOL(Mss,EUV,T_i,r_i,m_i, dataset_file = None)
    ## INPUT: Mstar [Msun], EUV [erg/cm/s], Teq [K], Rpl [Re], Mpl [Me]
    feuv = physics.get_flux(leuv, dist_au=sep)
    fbol = physics.get_flux(lbol, dist_au=sep)
    teq  = physics.temp_eq(fbol)
    args = dict(Mss = mstar, EUV = feuv, T_i = teq, r_i = radius, m_i = mass)

    # Redirect stdout to avoid inteprolator messages clogging up buffer
    with utils.supress_stdout():
        result = _k18_interpolator.INTERPOL(**args)
    
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

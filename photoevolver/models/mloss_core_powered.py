
import numpy as np
from astropy import units, constants as const
from photoevolver import physics

_MODEL_PARAMS = dict(
    gamma = 7/5, # adiabatic index
    mu    = 2 * const.m_p, # atmospheric molecular mass in kg
)


def modified_bondi_radius(mcore :float, teq :float):
    """
    Calculates modified Bondi radius, the height at which the escape velocity and the sound speed are equal.

    Parameters
    ----------
    mcore :float, core mass (Earth masses)
    teq   :float, equilibrium temperature (Kelvin)
    
    Returns
    -------
    rbondi :float, Bondi radius (Earth radii)
    """
    gamma = _MODEL_PARAMS['gamma']
    mu    = _MODEL_PARAMS['mu']
    rbondi = (gamma-1)/gamma * const.G * mcore * mu / (const.k_B * teq)
    return rbondi


def density_rcb(
        mcore :float,
        rcore :float,
        fenv :float,
        renv :float,
        rbondi :float
    ):
    """
    Calculates the density at the radiative-convective boundary of the atmosphere.
    
    Parameters
    ----------
    mcore  :float, core mass
    rcore  :float, core radius
    fenv   :float, envelope mass fraction
    renv   :float, envelope thickness
    rbondi :float, Bondi radius

    Returns
    -------
    dens_rcb :float, density at RCB
    """
    gamma = _MODEL_PARAMS['gamma']
    mu    = _MODEL_PARAMS['mu']
    
    dens_rcb = gamma / (gamma-1)
    dens_rcb *= (fenv * mcore) / (4 * np.pi * rcore**2 * renv)
    dens_rcb *= (rbondi * renv / rcore**2)**(-1/(gamma-1))
    return dens_rcb


def sound_speed(teq :float):
    """
    Calculates the speed of sound in an atmosphere
    
    Parameters
    ----------
    teq :float, equilibrium temperature

    Returns
    -------
    vsound :float, sound speed
    """
    mu = _MODEL_PARAMS['mu']
    return np.sqrt( const.k_B * teq / mu )


def sonic_point_radius(mass :float, teq :float):
    """
    Calculates the radius at which the planetary outflow becomes supersonic
    
    Parameters
    ----------
    mass :float, planet mass
    teq  :float, equilibrium temperature

    Returns
    -------
    rsonic :float, sonic point radius
    """
    vsound = sound_speed(teq)
    rsonic = const.G * mass / (2 * vsound**2)
    return rsonic


def atmospheric_opacity_rcb(dens_rcb, zstar = 1.0):
    """
    Calculates the opacity of the atmosphere from its density
    at the radiative-convective boundary.

    Parameters
    ----------
    dens_rcb : float, density at RCB.
    zstar    : float (optional), metallicity

    Returns
    -------
    opacity : float
    """
    beta = 0.6
    opacity = (0.1 * units.cm**2 / units.g) * zstar * (dens_rcb / (1e-3 * units.g / units.cm**3))**beta
    return opacity


def planet_luminosity(teq :float, rbondi :float, dens_rcb :float):
    """
    Calculates the thermal luminosity of a planet.

    Parameters
    ----------
    teq      :float, equilibrium temperature
    rbondi   :float, Bondi radius
    dens_rcb :float, density at RCB

    Returns
    -------
    planet_lum :float
    """
    opacity = atmospheric_opacity_rcb(dens_rcb)
    lum = 64*np.pi/3 * const.sigma_sb * teq**4 * rbondi / opacity / dens_rcb
    return lum


def planet_energy_limited_mloss(
        mass        :float,
        rcore       :float,
        radius_rcb  :float,
        rbondi      :float,
        dens_rcb    :float,
        teq         :float
    ):
    """ Calculates the energy-limited mass loss rate on a planet powered by its thermal energy """
    lum_rcb = planet_luminosity(teq, rbondi, dens_rcb)
    g_rcb = const.G * mass / radius_rcb**2
    return lum_rcb / g_rcb / rcore


def planet_bondi_limited_mloss(
        mass       :float,
        radius_rcb :float,
        rsonic     :float,
        vsound     :float,
        dens_rcb   :float 
    ):
    """ Calculates the Bondi-limited mass loss rate on a planet """
    expn = const.G * mass / vsound**2 / radius_rcb
    return 4 * np.pi * rsonic**2 * vsound * dens_rcb * np.exp( - expn )


def massloss_core_powered(
        mass   :float,
        radius :float,
        lbol   :float,
        sep    :float,
        mcore  :float,
        rcore  :float,
        fenv   :float,
        renv   :float,
        **kwargs
    ):
    """
    Returns the mass loss rate predicted by the core-pwoered model.

    Parameters
    ----------
    mass   :float, planet mass
    radius :float, planet radius
    lbol   :float, bolometric luminosity of the host star
    sep    :float, orbital separation
    mcore  :float, core mass
    rcore  :float, core radius
    fenv   :float, envelope mass fraction, defined as Menv/Mcore
    renv   :float, envelope thickness

    Returns
    -------
    mloss   :float, mass loss rate in grams per second.
    """

    jeans = physics.jeans_parameter(mass=mass, radius=radius, lbol=lbol, sep=sep)
    if jeans >= 25.0:
        return 0.0 # Negligible mass loss rate

    # Give units to planet parameters
    mass   *= units.M_earth
    mcore  *= units.M_earth
    radius *= units.R_earth
    rcore  *= units.R_earth
    renv   *= units.R_earth
    teq = units.K * physics.temp_eq(fbol = physics.get_flux(lbol, dist_au = sep))

    radius_rcb = radius
    rsonic     = sonic_point_radius(mass, teq)
    vsound     = sound_speed(teq)
    rbondi     = modified_bondi_radius(mcore, teq)
    dens_rcb   = density_rcb(mcore, rcore, fenv, renv, rbondi)

    mloss_e = planet_energy_limited_mloss(mass, rcore, radius_rcb, rbondi, dens_rcb, teq)
    mloss_b = planet_bondi_limited_mloss(mass, radius_rcb, rsonic, vsound, dens_rcb)
    mloss = min(mloss_e, mloss_b)

    return mloss.to('g/s').value
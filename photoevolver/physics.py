

import numpy as np
from astropy import units, constants


def fbol_earth() -> float:
    """Returns the bolometric flux on Earth in erg/s/cm^2"""
    return get_flux(lum = constants.L_sun.to("erg/s").value, dist_au = 1.0)


def SI_flux(flux_erg :float) -> float:
    """ Converts flux from erg/s/cm^2 to SI units W/m^2 """
    return flux_erg * (units.erg / units.s / units.cm**2).to("W m^-2")


def get_flux(
        lum     :float,
        dist_au :float = None,
        dist_pc :float = None
    ) -> float:
    """Calculates the flux that would be observed from a distance
    in either AU (`dist_au`) or parsecs (`dist_pc`)
    given the luminosity of the source.
    If both distances are specified, the one in AU takes precedence.
    
    Parameters
    ----------
    lum     : float
        Luminosity of the source in units of erg/s.
    dist_au : float, optional
        Distance to the source in units of AU.
    dist_pc : float, optional
        Distance to the source in units of parsecs.

    Returns
    -------
    flux    : float
        Corresponding flux in units of erg/s/cm^2.

    Raises
    ------
    ValueError
        If neither the distance in AU or parsecs is provided.

    """
    eqn = lambda lum,dist: lum/(4.0*np.pi*(dist)**2)
    if dist_au: return eqn(lum, dist_au*units.au.to('cm'))
    if dist_pc: return eqn(lum, dist_pc*units.pc.to('cm'))
    raise ValueError(
        "Specify distance in either AU (`dist_au`) or parsecs (`dist_pc`)")

    
def get_luminosity(
        flux    :float,
        dist_au :float = None,
        dist_pc :float = None
    ) -> float:
    """Calculates the luminosity of a source
    from its measured flux at a given distance
    in either AU (`dist_au`) or parsecs (`dist_pc`)
    If both distances are specified, the one in AU takes precedence.
    
    Parameters
    ----------
    flux    : float
        Flux from the source in units of erg/s/cm^2.
    dist_au : float, optional
        Distance to the source in units of AU.
    dist_pc : float, optional
        Distance to the source in units of parsecs.

    Returns
    -------
    lum    : float
        Corresponding luminosity in units of erg/s.

    Raises
    ------
    ValueError
        If neither the distance in AU or parsecs is provided.

    """
    eqn = lambda lum,dist: flux*(4.0*np.pi*(dist)**2)
    if dist_au: return eqn(flux, dist_au*units.au.to('cm'))
    if dist_pc: return eqn(flux, dist_pc*units.pc.to('cm'))
    raise ValueError(
        "Specify distance in either AU (`dist_au`) or parsecs (`dist_pc`)")


def keplers_third_law(
        big_mass   :float,
        small_mass :float = 0.0,
        period     :float = None,
        sep        :float = None
    ) -> float:
    """Application of Kepler's third law.
    For a system of two bodies orbiting each other, it returns either
    the orbital period (if separation specified), or orbital separation
    (if period specified) of their common orbit.
    The mass of the two bodies can be specified, but only the
    mass of the larger one is required, which is an acceptable approximation
    when the difference in mass is large.
    If both period and separation are specified, the period takes precedence.

    Parameters
    ----------
    big_mass    : float
        Mass of the more massive body in units of solar masses.
    small_mass  : float, optional
        Mass of the lighter body in units of Earth masses (different unit!)
    period      : float, optional
        Orbital period in units of days.
    sep         : float, optional
        Orbital separation in AU

    Returns
    -------
    period_or_sep : float
        Period if separation specified, or separation if period specified.

    Raises
    ------
    ValueError
        If neither period or separation are defined.

    """
    total_mass = big_mass * units.M_sun + small_mass * units.M_earth
    const = constants.G*(total_mass)/(4.0*np.pi**2)
    if period: return np.cbrt(const*(period*units.day)**2).to("AU").value
    if sep:    return np.sqrt((sep*units.au)**3/const).to("day").value
    raise ValueError("Specify either orbital period or separation")


def temp_eq(fbol: float) -> float:
    """Calculate equilibrium temperature of a planet.

    Parameters
    ----------
    fbol    : float, bolometric flux in erg/s/cm^2

    Returns
    -------
    t_eq    : float, equilibrium temperature in K
    """
    return ( SI_flux(fbol) / (4.0 * constants.sigma_sb.value))**(0.25)


def planet_density(mass :float, radius :float) -> float:
    """Calculates planet density in g/cm^3
    
    Parameters
    ----------
    mass    : float, planet's mass in Earth masses
    radius  : float, planet's mass in Earth radii

    Returns
    -------
    density : float, in units of g/cm^3
    """
    conv = (constants.M_earth / constants.R_earth**3).to("g cm^-3").value
    return conv * mass / (4/3 * np.pi * radius**3)


def prot_from_vsini(vsini :float, rstar :float) -> float:
    """ Calculates the maximum rotation period of a star
    from its rotational velocity at line of sight.

    Parameters
    ----------
    vsini   : float, rotational velocity times the sine
            of the inclination in km/s.
    rstar   : float, stellar radius in Solar radii.

    Returns
    -------
    prot    : float, rotation period in days
    """
    k = (constants.R_sun * units.s).to("km day").value
    return k * 2.0 * np.pi * rstar / vsini


def rossby_number(
        vkcolor  :float = 0,
        prot     :float = 0,
        full_out :bool = False,
        safe     :bool = False
    ) -> float | tuple[float,float]:
    """
    Estimates the Rossby number of a star given its V-K color index and its rotation period in days.
    It returns the Rossby number and the turnover time in days.
    The empirical relation comes from the Wright et al 2011 paper:
            (https://ui.adsabs.harvard.edu/abs/2011ApJ...743...48W/abstract)

    Parameters
    ----------
    vkcolor  : float, V-K color index
    prot     : float, rotation period in days
    full_out : bool (optional), if True, returns both rossby number
                and turnover time
    safe     : bool (optional), enforce model limits by raising ValueError

    Returns:
    --------
    rossby  : float, rossby number
    """
    vk_bounds = [1.1, 6.6]
    if safe and not (vk_bounds[0] <= vkcolor <= vk_bounds[1]):
        raise ValueError(
            f"V-K color {vkcolor:3g} out of model bounds {vk_bounds}"
        )
    if(vkcolor <= 3.5):
        turnover = 10**(0.73 + 0.22*vkcolor)
    else:
        turnover = 10**(-2.16 + 1.5*vkcolor - 0.13*vkcolor**2)                     
    rossby = prot/turnover
    if full_out: return (rossby,turnover)
    else:        return rossby


def rossby_number_from_mass(
        mass    :float,
        prot    :float,
        full_out :bool = False,
        safe    :bool = False
    ) -> float:
    """
    Estimates Rossby number using stellar mass and rotation period.

    Parameters
    ----------
    mass    : float, stellar mass in Solar masses
    prot    : float, stellar spin period in days
    full_out : bool (optional), if True, returns both rossby number
            and turnover time
    safe    : float (optional), enforce model limits by raising ValueError

    Returns
    -------
    rossby  : float, rossby number
    """
    mass_bounds = [0.09, 1.36]
    if safe and not (mass_bounds[0] < mass < mass_bounds[1]):
        raise ValueError(f"Mass {mass} out of model bounds {mass_bounds}")
    log_t = 1.16 - 1.49 * np.log10(mass) - 0.54 * np.log10(mass)**2
    if full_out:
        return prot / (10**log_t), 10**log_t
    return prot / (10**log_t)


def xray_hardness(hard :float, soft :float) -> float:
    """ Returns the hardness ratio of high and low energu X-ray counts"""
    return (hard - soft)/(hard + soft)

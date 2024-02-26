import numpy as np
from photoevolver import physics, cmodels

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
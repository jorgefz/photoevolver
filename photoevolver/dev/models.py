
"""
File description
"""

from .planet import EvoState, Planet, get_flux
from . import physics


def envelope_lopez14(state :EvoState, model_kw :dict) -> float:
    """
    Returns the envelope thickness in Earth radii using the model
    by Lopez & Fortney (2014).
    
    Parameters
    ----------
    mass   : float, planet mass in Earth masses
    fenv   : float, envelope mass fraction = (mass-mcore)/mcore
    lbol   : float, bolometric luminosity of the hos star (erg/s/cm^2)
    sep    : float, orbital separation of the planet in AU
    age    : float, system age in Myr
    lf14_opaque : bool (optional), enables enhanced opacity

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

    fbol = physics.get_flux(lum=state.lbol, dist_au=state.sep)
    
    mass_term = 2.06 * ( state.mass )**(-0.21)
    flux_term = ( physics.SI_flux(fbol/physics.fbol_earth()) )**(0.044)
    age_term  = ( state.age/5000 )**(age_power)
    fenv_term = ( state.fenv/0.05)**(0.59)
    renv = mass_term * fenv_term * flux_term * age_term
    return renv
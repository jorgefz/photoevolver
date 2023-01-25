
import numpy as np
from astropy import units
import astropy.constants as const
import uncertainties as uc

# General utilities

def kprint(**kwargs):
    """Prints input keyword arguments"""
    for k in kwargs.keys():
            print(k, '=', kwargs[k])

def ezip(*args):
    """Zip with enumerate generator. First item is index"""
    for i,a in enumerate(zip(*args)):
            yield i, *a

def ezip_r(*args):
    """`ezip` that reverses the elements on the input iterables"""
    rargs = tuple(reversed(arg) for arg in args)
    return ezip(*rargs)

def indexable(obj):
    """ Returns true if an input object is indexable.
    E.g. lists, dicts, or anything that can be addressed with obj[index]
    """
    return hasattr(obj, '__getitem__')

def is_mors_star(obj):
	try:
		import Mors
		return isinstance(obj, Mors.Star)
	except ImportError:
		return False


####################
### Custom units ###
####################

flux_units = units.erg / units.cm**2 / units.s
lum_units = units.erg / units.s

# Mass loss unit - Earth masses per Myr
# Conversion to grams per second:
#   mass_loss * MeMyr.to("g/s")
MeMyr = const.M_earth / units.Myr


##########################
### Physical equations ###
##########################

@np.vectorize
def luminosity(flux :float, dist_au :float = np.nan, dist_pc :float = np.nan):
    """Calculates luminosity from incident flux in erg/s"""
    if(~np.isnan(dist_au)):
        dist = dist_au * units.au.to("cm")
    elif(~np.isnan(dist_pc)):
        dist = dist_pc * units.pc.to("cm")
    else:
        dist = np.nan
    return flux * 4.0 * np.pi * (dist)**2

@np.vectorize
def flux(
        luminosity: float,
        dist_au :float = np.nan,
        dist_pc :float = np.nan
    ) -> float:
    """Calculates flux at a given distance from a source with given luminosity in erg/cm^2/s"""
    if(~np.isnan(dist_au)):
        dist = dist_au * units.au.to("cm")
    elif(~np.isnan(dist_pc)):
        dist = dist_pc * units.pc.to("cm")
    else:
        dist = np.nan
    return luminosity / (4.0 * np.pi * (dist)**2)

@np.vectorize
def equilibrium_temperature(fbol: float) -> float:
    """Calculate equilibrium temperature from bolometric flux at the planet,
    in units of erg/cm^2/s"""
    return (fbol * flux_units.to("W/m^2") / (4.0 * const.sigma_sb.value))**(0.25)

@np.vectorize
def planet_density(mass :float, radius :float) -> float:
    """Calculates planet density in g/cm^3"""
    conv = const.M_earth.to('g').value / const.R_earth.to('cm').value ** 3
    return conv * mass / (4/3 * np.pi * radius**3)


@uc.wrap
def prot_from_vsini(vsini :float, rstar :float):
    return 2.0 * np.pi * rstar * const.R_sun.to("km").value * units.s.to("day") / vsini

def rossby_number(
        vkcolor :float = 0,
        prot :float = 0,
        full_out :bool = False,
        safe :bool = False
    ):
    """
    Estimates the Rossby number of a star given its V-K color index and its rotation period in days.
    It returns the Rossby number and the turnover time in days.
    The empirical relation comes from the Wright et al 2011 paper:
            (https://ui.adsabs.harvard.edu/abs/2011ApJ...743...48W/abstract)

    Input:
            vkcolor:    (float) V-K color index
            prot:               (float) Rotation period in days
            full_out:   (bool) Whether to return rossby number + turnover time
            safe:               (bool) Enforce model limits
    Returns:
            rossby number (float)
    """
    if safe and (vkcolor < 1.1 or vkcolor > 6.6):
        raise ValueError(f"V-K color {vkcolor:3g} out of model bounds (1.1, 6.6)")
    if(3.5 >= vkcolor):
        turnover_time = 10**(0.73 + 0.22*vkcolor)
    else:
        turnover_time = 10**(-2.16 + 1.5*vkcolor - 0.13*vkcolor**2)                     
    rossby = prot / turnover_time
    return rossby

def rossby_number_from_mass(mass:float, prot:float, safe:bool=False):
        """
        Estimates Rossby number using stellar mass and rotation period.
        """
        if safe and (mass < 0.09 or mass > 1.36):
                raise ValueError(f"Input mass {mass} out of model bounds (0.09,1.36)")
        log_t = 1.16 - 1.49 * np.log10(mass) - 0.54 * np.log10(mass)**2
        return prot / (10**log_t)


def hardness(hard, soft):
        return (hard - soft)/(hard + soft)


def rebin_array(input_data :list, factor: float, func=sum):
    """
    Rebins an array to a lower resolution by a factor,
    and returns the new array.
    Leftover values are added up on the last bin.
    """
    factor = int(factor)
    if(factor < 1):
            raise ValueError("binning factor must be an integer greater than 1")
    data = np.copy(input_data)
    leftover = len(data) % factor #Extra values that don't fit reshape
    leftover_ind = len(data) - leftover
    ndata = data[:leftover_ind].reshape((len(data)//factor, factor))
    ndata = np.array([func(newbin) for newbin in ndata])
    # Append leftover
    if (leftover > 0):
            ndata = np.append(ndata, func(data[leftover_ind:]))
    return ndata


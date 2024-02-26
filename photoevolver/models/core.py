from photoevolver.settings import _MODEL_DATA_DIR
from photoevolver import physics
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
import uncertainties as uncert
from uncertainties import umath


_zeng19_cache :dict[interp1d|None] = {
    'rock'       : None, # 100% MgSiO2
    'rock_iron'  : None, # 33% Fe + 67% MgSiO2 (Earth-like)
    'iron'       : None, # 100% Fe
    'water'      : None, # 100% H2O
    'water_rock' : None  # 50% H2O + 50% Earth-like
}


def core_zeng19_rock(mcore: float, **kwargs) -> float:
    """ Calculates planet radius from its mass assuming a pure rock (100% MgSiO2) composition using the model by Zeng+19 """
    if _zeng19_cache['rock'] is None:
        data = np.loadtxt(_MODEL_DATA_DIR + 'zeng19/core_zeng19_rock.csv', delimiter = ',')
        masses, radii = data.T
        _zeng19_cache['rock'] = interp1d(x = masses, y = radii,
            kind = 'cubic', fill_value="extrapolate", assume_sorted = True)

    model = _zeng19_cache['rock']
    return float(model(mcore))


def core_zeng19_iron(mcore: float, **kwargs) -> float:
    """ Calculates planet radius from its mass assuming a pure iron (100% Fe) composition using the model by Zeng+19 """
    if _zeng19_cache['iron'] is None:
        data = np.loadtxt(_MODEL_DATA_DIR + 'zeng19/core_zeng19_iron.csv', delimiter = ',')
        masses, radii = data.T
        _zeng19_cache['iron'] = interp1d(x = masses, y = radii,
            kind = 'cubic', fill_value="extrapolate", assume_sorted = True)

    model = _zeng19_cache['iron']
    return float(model(mcore))


def core_zeng19_rock_iron(mcore: float, **kwargs) -> float:
    """ Calculates planet radius from its mass assuming an Earth-like (1/3 iron, 2/3 rock) composition using the model by Zeng+19 """
    if _zeng19_cache['rock_iron'] is None:
        data = np.loadtxt(_MODEL_DATA_DIR + 'zeng19/core_zeng19_rock_iron.csv', delimiter = ',')
        masses, radii = data.T
        _zeng19_cache['rock_iron'] = interp1d(x = masses, y = radii,
            kind = 'cubic', fill_value="extrapolate", assume_sorted = True)

    model = _zeng19_cache['rock_iron']
    return float(model(mcore))


def core_zeng19_water(mcore :float, lbol :float, sep :float, albedo :float = 0.2, **kwargs):
    """ Calculates planet radius from its mass assuming a pure water (100% H2O) composition using the model by Zeng+19 """
    fbol = physics.get_flux(lbol, dist_au = sep)
    teq = physics.temp_eq(fbol)
    
    # Load cache of model sequences interpolating equilibrium temperature
    if _zeng19_cache['water'] is None:
        # Load model sequences
        temp = 300, 500, 700, 1000
        sequences = [ np.loadtxt(_MODEL_DATA_DIR + f"zeng19/core_zeng19_water_{t}K.csv", delimiter = ',') for t in temp ]
        models    = [
            interp1d(x = seq.T[0], y = seq.T[1],
                kind = 'cubic', fill_value="extrapolate", assume_sorted = True)
            for seq in sequences
        ]
        # Resample into a regular grid
        regular_masses = np.logspace(np.log10(0.05), np.log10(500), base = 10, num = 50)
        regular_radii  = [ model(regular_masses) for model in models ]

        # 2D interpolation
        _zeng19_cache['water'] = RegularGridInterpolator(
            points = (temp, regular_masses),
            values = regular_radii,
            method = 'slinear',
            bounds_error = False,
            fill_value = None # extrapolate
        )

    model = _zeng19_cache['water']
    return float(model([teq, mcore]))


def core_zeng19_water_rock(mcore :float, lbol :float, sep :float, albedo :float = 0.2, **kwargs):
    """ Calculates planet radius from its mass assuming a water + rock (50% H2O+ 50% Earth-like) composition using the model by Zeng+19 """
    fbol = physics.get_flux(lbol, dist_au = sep)
    teq = physics.temp_eq(fbol)
    
    # Load cache of model sequences interpolating equilibrium temperature
    if _zeng19_cache['water_rock'] is None:
        # Load model sequences
        temp = 300, 500, 700, 1000
        sequences = [ np.loadtxt(_MODEL_DATA_DIR + f"zeng19/core_zeng19_water_rock_{t}K.csv", delimiter = ',') for t in temp ]
        models    = [
            interp1d(x = seq.T[0], y = seq.T[1],
                kind = 'cubic', fill_value="extrapolate", assume_sorted = True)
            for seq in sequences
        ]
        # Resample into a regular grid
        regular_masses = np.logspace(np.log10(0.5), np.log10(64), base = 10, num = 50)
        regular_radii  = [ model(regular_masses) for model in models ]

        # 2D interpolation
        _zeng19_cache['water_rock'] = RegularGridInterpolator(
            points = (temp, regular_masses),
            values = regular_radii,
            method = 'slinear',
            bounds_error = False,
            fill_value = None # extrapolate
        )

    model = _zeng19_cache['water_rock']
    return float(model([teq, mcore]))


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



from photoevolver.settings import _MODEL_DATA_DIR
from photoevolver import physics

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d as ScipyInterp1d
from scipy.interpolate import LinearNDInterpolator


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


@np.vectorize
def rotation_activity_wright11(
        prot  :float,
        mstar :float,
    ) -> float:
    """
    Converts rotation period to X-ray activity
    using the rotation-activity relation by Wright et al. 2011
    for convective-radiative stars (FGK and early M dwarfs).

    Parameters
    ----------
    prot    :float, Spin period of the star in days
    mstar   :float, Mass of the star in solar masses
    
    Returns
    -------
    xray_activity :float, X-ray activity (Lx/Lbol)
    """
    rossn = physics.rossby_number_from_mass(mass = mstar, prot = prot)
    rossn_sat = 0.13
    rx_sat    = 10**(-3.13)
    pow_exp   = -2.18
    norm = rx_sat / np.power(rossn_sat, pow_exp)
    if rossn <= rossn_sat:
        return rx_sat
    return norm * np.power(rossn, pow_exp)


@np.vectorize
def rotation_activity_wright18(
        prot  :float,
        mstar :float,
    ) -> float:
    """
    Converts rotation period to X-ray activity
    using the rotation-activity relation by Wright et al. 2018
    for fully convective stars (late M dwarfs).

    Parameters
    ----------
    prot    :float, Spin period of the star in days
    mstar   :float, Mass of the star in solar masses
    
    Returns
    -------
    xray_activity :float, X-ray activity (Lx/Lbol)
    """
    rossn = physics.rossby_number_from_mass(mass = mstar, prot = prot)
    rossn_sat = 0.14
    rx_sat    = 10**(-3.05)
    pow_exp   = -2.3
    norm = rx_sat / np.power(rossn_sat, pow_exp)
    if rossn <= rossn_sat:
        return rx_sat
    return norm * np.power(rossn, pow_exp)


@np.vectorize
def rotation_activity_johnstone21(
        prot  :float,
        mstar :float,
    ) -> float:
    """
    Converts rotation period to X-ray activity
    using the rotation-activity relation by Johnstone et al. 2021
    for main sequence stars.

    Note: Johnstone+21 use the period-rossby number relation
    from Spada et al. (2013), which is very similar to the one
    that the `rossby_number_from_mass` function and
    Wright et al. (2011) use.

    Parameters
    ----------
    prot    :float, Spin period of the star in days
    mstar   :float, Mass of the star in solar masses

    Returns
    -------
    xray_activity :float, X-ray activity (Lx/Lbol)
    """
    p1, p2 = -0.135, -1.889
    ro_sat, rx_sat = 0.0605, 5.135e-4
    c1 = rx_sat/np.power(ro_sat,p1)
    c2 = rx_sat/np.power(ro_sat,p2)
    ro = physics.rossby_number_from_mass(mass = mstar, prot = prot)    
    if ro <= ro_sat:
        return c1 * np.power(ro, p1)
    return c2 * np.power(ro, p2)


def rossby_wright11(prot :float, mstar :float = None, vk :float = None):
    """
    Estimates the Rossby number of a star in days.
    Requires its rotation period and either its V-K colour or its mass.
    The empirical relation comes from the Wright et al 2011 paper:
            (https://ui.adsabs.harvard.edu/abs/2011ApJ...743...48W/abstract)

    Parameters
    ----------
    prot     : float, rotation period in days
    mstar    : float (optional), stellar mass in solar masses
    vk       : float (optional), V-K color index

    Returns:
    --------
    rossby  : float, rossby number
    """
    # vk_bounds = [1.1, 6.6]
    # mstar_bounds = [0.09, 1.36]

    if mstar is not None:
        log_tconv = 1.16 - 1.49*np.log10(mstar) - 0.54*np.log10(mstar)**2
        return prot / (10**log_tconv) 

    if vk is None:
        raise ValueError("[rossby_wright11] Provide either mstar or vk colour.")

    # np.where works like an element-wise 'if' statement for arrays
    vk = np.array(vk)
    log_tconv = np.where(vk <= 3.5,
        0.73 + 0.22*vk,
        -2.16 + 1.5*vk - 0.13*vk**2
    )
    return prot/(10**log_tconv)



class star_pecaut13:
	"""
	Interpolates the stellar sequences by Pecaut et al. (2013).
    "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
    http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    Eric Mamajek
    Version 2022.04.16
	"""

	_table :pd.DataFrame = pd.read_csv(_MODEL_DATA_DIR+'/pecaut13/pecaut13.csv')
	_fn_cache :dict[dict[callable]] = {}

	def fields() -> list[str]:
		"""
		Returns a list of stellar parameters that can be interpolated
		"""
		return star_pecaut13._table.columns.tolist()

	def spt(field: str, value: float) -> str:
		"""
		Estimates the spectral type from a given parameter.
		"""
		if field not in star_pecaut13.fields():
			raise KeyError(f"[startable] Unknown field '{field}'")
		idx = star_pecaut13._table[field].sub(value).abs().idxmin()
		row = star_pecaut13._table.loc[[idx]]
		return row['SpT'].iloc[0]

	def interpolate(field: str, value: float) -> dict:
		"""
		Interpolates all stellar parameters given a value for one of them.
		Spectral type is not supported as an input field.
		"""
		if field not in star_pecaut13.fields():
			raise KeyError(f"[startable] Unknown field '{field}'")

		fields = star_pecaut13.fields()[1:-1]
		row = {
			key: star_pecaut13.get(key, field, value)
			for key in fields
		}
		row['SpT'] = star_pecaut13.spt(field, value)
		return row

	def get(field :str, using :str, value :str) -> float:
		from_col   = star_pecaut13._table[using].to_numpy()
		to_col = star_pecaut13._table[field].to_numpy()
		fn = ScipyInterp1d(x = from_col, y = to_col, kind='linear')
		return float(fn(value))



class star_spada13:
    """
    Stellar evolution models by Spada+13.
    (Spada, F.,Demarque, P.,Kim, Y.-C. & Sills, A. 2013, ApJ, 776, 87)
    Calculate Lbol, TauConv, Teff, and Rstar using Mstar and Age.
    """

    _table        :pd.DataFrame = pd.read_hdf(_MODEL_DATA_DIR+'spada13/Spada2013.hdf5')
    _default_age  :float        = 2000 # Myr
    _fields       :list[str]    = ['Lbol', 'TauConv', 'Teff', 'Rstar']
    _interp_cache :dict         = {}

    @staticmethod
    def get(field :str, mstar :float, age :float = _default_age):
        """
        Returns the requested stellar parameter for a star.
        
        Parameters
        ----------
        field   :str, stellar parameter to calculate.
                One of the following:
                    'Lbol': bolometric luminosity in Lsun.
                    'Teff': effective temperature in K.
                    'Rstar': stellar radius in Rsun.
                    'TauConv': convective turnover time in days.
        mstar   :float|list, stellar mass in Msun.
        age     :float|list (optional), age of the star in Myr.
        """
        if field not in star_spada13._fields:
            return None
        
        # Cache exists
        if field in star_spada13._interp_cache:
            return star_spada13._interp_cache[field](mstar, age)
        
        # Create interpolation function
        points = star_spada13._table[['Mstar','Age']].to_numpy()
        values = star_spada13._table[field]
        interp = LinearNDInterpolator(points, values)
        star_spada13._interp_cache[field] = interp

        return interp(mstar, age)

    
    @staticmethod
    def star(mstar :float, age :float = _default_age) -> float:
        """
        Returns all available stellar parameters for a star.
        
        Parameters
        ----------
        mstar   :float|list, stellar mass in Msun.
        age     :float|list (optional), age of the star in Myr.

        Returns
        -------
        params :dict[float]
        """
        return { field: star_spada13.get(field, mstar, age) for field in star_spada13._fields }

    @staticmethod
    def lbol(mstar :float, age :float = _default_age) -> float:
        """ Returns the bolometric luminosity in Lsun for a given mass (and age)"""
        return star_spada13.get('Lbol', mstar, age)

    @staticmethod
    def tconv(mstar :float, age :float = _default_age) -> float:
        """ Returns the convective turnover time in days for a given mass (and age)"""
        return star_spada13.get('TauConv', mstar, age)

    @staticmethod
    def rossby(prot :float, mstar :float, age :float = _default_age) -> float:
        """ Returns the Rossby number for a given spin period, mass (and age)"""
        return prot / star_spada13.get('TauConv', mstar, age)

    @staticmethod
    def teff(mstar :float, age :float = _default_age) -> float:
        """ Returns the effective temperature in Kelvin for a given stellar mass (and age)"""
        return star_spada13.get('Teff', mstar, age)

    @staticmethod
    def rstar(mstar :float, age :float = _default_age) -> float:
        """ Returns the stellar radius in Rsun for a given mass (and age)"""
        return star_spada13.get('Rstar', mstar, age)




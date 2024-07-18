

from photoevolver.settings import _MODEL_DATA_DIR
from photoevolver import physics
from photoevolver.utils import HelperFunctions

import numpy as np
import pandas as pd
import functools
from scipy.interpolate import interp1d as ScipyInterp1d
from scipy.interpolate import LinearNDInterpolator


def euv_king18(
		lx	 : float,
		rstar  : float,
		energy : str = "0.1-2.4"
	):
	"""
	Calculates the EUV luminosity from the X-rays
	using the empirical relation by King et al. (2018).
	Source: https://ui.adsabs.harvard.edu/abs/2018MNRAS.478.1193K/abstract

	Parameters
	----------
	lx	  : float, X-ray luminosity of the star in erg/s
	rstar   : float, radius of the star in solar radii
	energy  : str, input energy range of the X-rays in keV.
			The allowed ranges are: 0.1-2.4, 0.124-2.48, 0.15-2.4,
			0.2-2.4, and 0.243-2.4 keV.

	Returns
	-------
	leuv	: float, the EUV luminosity of the star with energy between
			from 0.0136 keV and the lower range of the input X-rays in erg/s.
	"""
	# Model parameters
	params = {	   # norm, exponent
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


def _xray_evo_powerlaw_eqn(
		age    :float,
		lbol   :float,
		rx_sat :float,
		t_sat  :float,
		expn   :float,
		*args, **kwargs
	):
	""" Saturated power law for X-ray evolution """
	if age < t_sat:
		return rx_sat * lbol
	norm = rx_sat / t_sat**expn
	return lbol * norm * age**expn

def xray_evo_powerlaw(
		lbol   :float,
		rx_sat :float,
		t_sat  :float,
		expn   :float,
	):
	"""
	Simple model for the X-ray evolution of a star,
	with a saturated phase with Lx/Lbol = `rx_sat` that lasts until age `t_sat`,
	and then decays as a power law with exponent `expn`.
	Returns a function that, when called with an age, returns the X-ray luminosity at that age.

	Parameters
	----------
		lbol    :float, stellar bolometric luminosity in erg/s
		rx_sat  :float, saturated x-ray activity level (Lx/Lbol)
		t_sat   :float, age at which star becomes unsaturated in Myr
		expn    :float, power law index for unsaturated phase

	Returns
	-------
		lx_evo  :callable, function that that returns the X-ray luminosity at a given age.

	Example
	-------
		lx_evo = ph.models.xray_evo_powerlaw(mstar=1.0, lbol=1e33, rx_sat=1e-3, t_sat=100, exp=-2)
		lx = lx_evo(age = 100)
	"""
	evo_fn = functools.partial(_xray_evo_powerlaw_eqn, lbol=lbol, t_sat=t_sat, rx_sat=rx_sat, expn=expn)
	evo_fn = np.vectorize(evo_fn)
	evo_fn.__name__ = f"xray_evo_powerlaw(age :float, *args, **kwargs)"
	return evo_fn
	

def xray_evo_jackson12(lbol :float, bv_color :float):
	"""
	X-ray evolution models by Jackson et al. (2012).

	Parameters
	----------
		lbol     :float, bolometric luminosity of the star in erg/s.
		bv_color :float, B-V colour of the star

	Returns
	-------
		lx_evo   : callable that returns X-ray luminosity (erg/s) at a given age (Myr).

	Example
	-------
		lx_evo = xray_evo_jackson12(lbol = 3.8e33, bv_color = 0.71) # Sun-like star
		lx = lx_evo(age = 100) # X-ray luminosity at 100 Myr
	"""
	model_params = {
		'bv_min'    : [ 0.290, 0.450, 0.565, 0.675, 0.790, 0.935, 1.275],
		'bv_max'    : [ 0.450, 0.565, 0.675, 0.790, 0.935, 1.275, 1.410],
		'log_rx_sat': [-4.28, -4.24, -3.67, -3.71, -3.36, -3.35, -3.14],
		'log_t_sat' : [ 7.87,  8.35,  7.84,  8.03,  7.90,  8.28,  8.21],
		'expn'      : [-1.22, -1.24, -1.13, -1.28, -1.40, -1.09, -1.18]
	}

	# Find matching model for input B-V colour
	model_index = None
	for i, (bvmin, bvmax) in enumerate(zip(model_params['bv_min'], model_params['bv_max'])):
		if bvmin <= bv_color < bvmax:
			model_index = i
			break
	if model_index is None:
		raise ValueError(f"[xray_evo_jackson12] Input B-V color ({bv_color}) not covered by Jackson+12 models")

	rx_sat = 10**model_params['log_rx_sat'][i]
	t_sat  = 10**model_params['log_t_sat'][i] / 1e6 # yr to Myr
	expn   = model_params['expn'][i]
	lx_evo = xray_evo_powerlaw(lbol=lbol, rx_sat=rx_sat, t_sat=t_sat, expn=expn)
	return lx_evo


def rossby_wright11(prot :float, mstar :float = None, vk :float = None):
	"""
	Estimates the Rossby number of a star in days.
	Requires its rotation period and either its V-K colour or its mass.
	The empirical relation comes from the Wright et al 2011 paper:
			(https://ui.adsabs.harvard.edu/abs/2011ApJ...743...48W/abstract)

	Parameters
	----------
	prot	 : float, rotation period in days
	mstar	: float (optional), stellar mass in solar masses
	vk	   : float (optional), V-K color index

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
	prot	:float, Spin period of the star in days
	mstar   :float, Mass of the star in solar masses
	
	Returns
	-------
	xray_activity :float, X-ray activity (Lx/Lbol)
	"""
	rossn = rossby_wright11(mstar = mstar, prot = prot)
	rossn_sat = 0.13
	rx_sat	= 10**(-3.13)
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
	prot	:float, Spin period of the star in days
	mstar   :float, Mass of the star in solar masses
	
	Returns
	-------
	xray_activity :float, X-ray activity (Lx/Lbol)
	"""
	rossn = rossby_wright11(mstar = mstar, prot = prot)
	rossn_sat = 0.14
	rx_sat	= 10**(-3.05)
	pow_exp   = -2.3
	norm = rx_sat / np.power(rossn_sat, pow_exp)
	if rossn <= rossn_sat:
		return rx_sat
	return norm * np.power(rossn, pow_exp)


@np.vectorize
def rotation_activity_johnstone21(
		prot  :float,
		mstar :float,
		age   :float = 1000 # Myr
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
	prot	:float, Spin period of the star in days
	mstar   :float, Mass of the star in solar masses
	age	 :float, Age in Myr

	Returns
	-------
	xray_activity :float, X-ray activity (Lx/Lbol)
	"""
	p1, p2 = -0.135, -1.889
	ro_sat, rx_sat = 0.0605, 5.135e-4
	c1 = rx_sat/np.power(ro_sat,p1)
	c2 = rx_sat/np.power(ro_sat,p2)
	ro = star_spada13.rossby(mstar = mstar, prot = prot, age = age)
	if ro <= ro_sat:
		return c1 * np.power(ro, p1)
	return c2 * np.power(ro, p2)



class star_pecaut13:
	"""
	Interpolates the stellar sequences by Pecaut et al. (2013).
	"A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
	http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
	Eric Mamajek
	Version 2022.04.16

	Fields
	------
		spt   : str, Spectral type
		lbol  : float, Bolometric luminosity in Lsun
		b_v   : float, B-V color
		teff  : float, Effective temperature in K
		bp_rp : float, Bp-Rp color
		g_rp  : float, G-Rp color
		Mg    : float, Absolute G magnitude
		v_k   : float, V-K color
		rstar : float, Stellar radius in Rsun
		mstar : float, Stellar mass in Msun
	"""

	_table :pd.DataFrame = None
	_fn_cache :dict[dict[callable]] = {}

	@staticmethod
	def _load_dataset():
		""" Loads dataset from disk, renames some columns, and calculates others """
		dset = pd.read_csv(_MODEL_DATA_DIR+'/pecaut13/pecaut13.csv', engine='c')
		dset['lbol'] = 10**dset['logL']
		rename = {
			'SpT'   : 'spt',   'Teff'  : 'teff',
			'B-V'   : 'b_v',   'V-Ks'  : 'v_k',
			'Bp-Rp' : 'bp_rp', 'G-Rp'  : 'g_rp',
			'R_Rsun': 'rstar', 'Msun'  : 'mstar',
			'M_G'   : 'Mg'
		}
		keep = list(rename.values()) + ['lbol']
		dset = dset.rename(columns = rename)
		dset = dset.loc[:, keep]
		return dset

	@staticmethod
	def _check_dataset_loaded():
		"""
		Avoid loading dataset on startup to reduce import time.
		Load only when class is used.
		"""
		if star_pecaut13._table is None:
			star_pecaut13._table = star_pecaut13._load_dataset()

	@staticmethod
	def _check_input(mapping :dict):
		""" Ensures only one valid input stellar parameter is provided """
		mapping.pop('spt', None) # Interpolating using spt not supported
		if len(mapping) != 1:
			raise KeyError("[star_pecaut13] Multiple/none valid parameters provided")
		return list(mapping.items())[0]

	@staticmethod
	def fields() -> list[str]:
		"""
		Returns a list of stellar parameters that can be interpolated
		"""
		star_pecaut13._check_dataset_loaded()
		return star_pecaut13._table.columns.tolist()

	@staticmethod
	def spt(**kwargs: dict) -> str:
		""" Determines spectral type from input stellar parameter """
		field, value = star_pecaut13._check_input(kwargs)
		star_pecaut13._check_dataset_loaded()
		if field not in star_pecaut13.fields():
			raise KeyError(f"[star_pecaut13] Unknown field '{field}'")
		idx = star_pecaut13._table[field].sub(value).abs().idxmin()
		row = star_pecaut13._table.loc[[idx]]
		return row['spt'].iloc[0]

	@staticmethod
	def get(field :str, extrapolate :bool = False, **kwargs :dict) -> float:
		"""
		Uses an input stellar parameter to calculate another parameter `field`.
		E.g. calculate stellar mass (`mstar`) from a given Bp-Rp color.
		"""
		star_pecaut13._check_dataset_loaded()
		key, value = star_pecaut13._check_input(kwargs)
		if field not in star_pecaut13.fields():
			raise KeyError(f"[star_pecaut13] Unknown stellar parameter '{field}'")
		if key not in star_pecaut13.fields():
			raise KeyError(f"[star_pecaut13] Unknown stellar parameter '{key}'")
		from_col = star_pecaut13._table[key].to_numpy()
		to_col   = star_pecaut13._table[field].to_numpy()

		fill_value = 'extrapolate' if extrapolate is True else np.nan
		fn = ScipyInterp1d(
			x = from_col, y = to_col, kind='linear',
			bounds_error=False, fill_value = fill_value
		)
		return fn(value)

	@staticmethod
	def star(extrapolate :bool = False, **kwargs :dict) -> dict:
		"""
		Returns all stellar parameters (except spectral type) from an input parameter.
		"""
		key, value = star_pecaut13._check_input(kwargs)
		fields = star_pecaut13.fields()
		params = {
			field: star_pecaut13.get(field, **{key:value}, extrapolate=extrapolate)
			for field in fields if field != 'spt'
		}
		return params

	@staticmethod
	def lbol(extrapolate :bool = False, **kwargs :dict):
		""" Return bolometric luminosity from an input stellar parameter """
		key, value = star_pecaut13._check_input(kwargs)
		return star_pecaut13.get(field='lbol', **{key:value}, extrapolate=extrapolate)

	@staticmethod
	def teff(extrapolate :bool = False, **kwargs :dict):
		""" Return effective temperature from an input stellar parameter """
		key, value = star_pecaut13._check_input(kwargs)
		return star_pecaut13.get(field='teff', **{key:value}, extrapolate=extrapolate)

	@staticmethod
	def rstar(extrapolate :bool = False, **kwargs :dict):
		""" Return stellar radius from an input stellar parameter """
		key, value = star_pecaut13._check_input(kwargs)
		return star_pecaut13.get(field='rstar', **{key:value}, extrapolate=extrapolate)

	@staticmethod
	def mstar(extrapolate :bool = False, **kwargs :dict):
		""" Return stellar mass from an input stellar parameter """
		key, value = star_pecaut13._check_input(kwargs)
		return star_pecaut13.get(field='mstar', **{key:value}, extrapolate=extrapolate)



class star_spada13:
	"""
	Stellar evolution models by Spada+13.
	(Spada, F.,Demarque, P.,Kim, Y.-C. & Sills, A. 2013, ApJ, 776, 87)
	Calculate Lbol, TauConv, Teff, and Rstar using Mstar and Age.
	"""

	_table		  :pd.DataFrame = None
	_default_age  :float	    = 2000 # Myr
	_fields	      :list[str]	= ['Lbol', 'TauConv', 'Teff', 'Rstar']
	_interp_cache :dict		    = {}

	@staticmethod
	def _check_dataset_loaded():
		if star_spada13._table is None:
			star_spada13._table = pd.read_hdf(_MODEL_DATA_DIR+'spada13/Spada2013.hdf5')


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
		age	 :float|list (optional), age of the star in Myr.
		"""
		star_spada13._check_dataset_loaded()
		
		if field not in star_spada13._fields:
			return None
		
		# Cache exists
		if field in star_spada13._interp_cache:
			return star_spada13._interp_cache[field](mstar, age)
		
		# Create interpolation function
		points = star_spada13._table[['Mstar','Age']].to_numpy()
		values = star_spada13._table[field]
		interp = LinearNDInterpolator(points, values, rescale=True)
		star_spada13._interp_cache[field] = interp

		return interp(mstar, age)

	
	@staticmethod
	def star(mstar :float, age :float = _default_age) -> float:
		"""
		Returns all available stellar parameters for a star.
		
		Parameters
		----------
		mstar   :float|list, stellar mass in Msun.
		age	 :float|list (optional), age of the star in Myr.

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




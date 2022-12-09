
import warnings
from typing import Any, Union, List, Callable

from copy import deepcopy
import numpy as np
import astropy.constants as Const
import matplotlib.pyplot as plt
import pickle

import astropy.units as U

from .structure import fenv_solve
from .owenwu17 import mass_to_radius as owen_radius
from .owenwu17 import radius_to_mass as owen_mass
from .planet import Planet
from .utils import indexable, is_mors_star
from .tracks import Tracks
from .core import globals


def lum_to_flux_au(lum: float, dist: float):
	""" Luminosity to flux (per cm^2) using distance in AU """
	return lum / (4 * np.pi * ( dist * U.au.to('cm') )**2)

def lum_to_flux_pc(lum: float, dist: float):
	""" Luminosity to flux (per cm^2) using distance in parsecs """
	return lum / (4 * np.pi * ( dist * U.pc.to('cm') )**2)

def flux_to_lum_au(flux: float, dist: float):
	""" Flux (per cm^2) to luminosity using distance in AU """
	return flux * (4 * np.pi * ( dist * U.au.to('cm') )**2)

def flux_to_lum_pc(flux: float, dist: float):
	""" Flux (per cm^2) to luminosity using distance in parsecs """
	return flux * (4 * np.pi * ( dist * U.pc.to('cm') )**2)

def equilibrium_temperature(fx: float) -> float:
	""" Returns equlibrium temperature in K of a planet given bolometric influx in erg/cm^2/s """
	flux_si = (U.erg / (U.cm)**2 / U.s).to(" W/m^2 ")
	return (fx * flux_si / (4 * Const.sigma_sb.value))**(1/4)


class EvoState:

	fields = [
		'mp', 'rp', 'menv', 'renv', 'mcore', 'rcore', 'fenv',
		'age', 'dist', 'mstar',
		'lx', 'leuv', 'lbol', 'fx', 'feuv', 'fbol', 'lxuv', 'fxuv',
		'mloss', 'mloss_fn', 'struct_fn',
		'args'
	]

	def __init__(self, **kwargs):
		_fields = EvoState.fields
		# enforce certain default members
		for k in _fields:
			self.__dict__[k] = kwargs.get(k, None)
			# if k in kwargs: self.__dict__[k] = kwargs[k]
			# else: self.__dict__[k] = None

	def validate() -> bool:
		"""
		Checks simulation state for invalid values (e.g. nan, negative mass, etc)
		Returns true if all parameters are ok, and false otherwise.
		Raises ValueError if any problems are found. 
		"""
		pass

	def as_dict(self) -> dict:
		""" Get simulation state as a dictionary """
		return self.__dict__

	def to_dict(self) -> dict:
		""" Get simulation state as a dictionary """
		return deepcopy(self.__dict__)

	def update_fluxes(self, lx: float, leuv: float, lbol: float):
		self.lx, self.leuv, self.lbol = lx, leuv, lbol
		self.fx   = lum_to_flux_au(lx  , self.dist)
		self.feuv = lum_to_flux_au(leuv, self.dist)
		self.fbol = lum_to_flux_au(lbol, self.dist)
		self.lxuv = lx + leuv
		self.fxuv = lum_to_flux_au(self.lxuv, self.dist)
		# alternate names
		self.Lbol = self.lbol
		self.Lxuv = self.lx + self.leuv
		self.Lx = self.lx
		self.Leuv = self.leuv

	def __repr__(self):
		msg = "--- Evolution State --- \n"
		for k,v in self.as_dict().items():
			msg += f"{k}: ".ljust(15)
			msg += f"{v:.2f}" if isinstance(v,float) else f"{v}"
			msg += '\n'
		return msg + "-----------------------\n"

	def __str__(self):
		return self.__repr__()


def King18Leuv(
		Lx     : Union[float,List],
		Rstar  : Union[float,List],
		energy : str = "0.1-2.4"
	):
	"""
	EUV Relation by King et al 2018: https://ui.adsabs.harvard.edu/abs/2018MNRAS.478.1193K/abstract

	Parameters
	----------
	Lx      : float, np.ndarray
		X-ray luminosity of the star in erg/s
	Rstar   : float, np.ndarray
		Radius of the star in solar radii
	energy	: str
		Energy range in the format "lower-upper" in keV.
		The allowed ranges are:
			"0.1-2.4", "0.124-2.48", "0.15-2.4", "0.2-2.4", "0.243-2.4"

	Returns
	-------
	Leuv    : float, np.ndarray
		The EUV luminosity of the star in the range 0.0136 keV to the lower range of the input X-rays

	Dataset
	-------
		X-ray range low energy; X-ray range high energy; constant; power-law index; EUV lower range; EUV higher range; Comment by authors
		keV;		keV;		erg/cm^2/s;	--;		keV;	keV;	--
		xrange_i;	xrange_f;	const;		pwlaw;	euv_i;	euv_f;	comment
		0.1;		2.4;		460;		-0.425;	0.0136;	0.1;	ROSAT PSPC
		0.124;		2.480;		650;		-0.450;	0.0136;	0.124;	Widely used
		0.150;		2.4;		880;		-0.467;	0.0136;	0.150;	XMM PN (lowest)
		0.2;		2.4;		1400;		-0.493;	0.0136;	0.2;	XMM PN (usual), XMM MOS, & Swift XRT
		0.243;		2.4;		2350;		-0.539;	0.0136;	0.243;	Chandra ACIS
	"""
	# model parameters
	allowed_ranges = "0.1-2.4", "0.124-2.48", "0.15-2.4", "0.2-2.4", "0.243-2.4"
	const = 460, 650, 880, 1400, 2350
	pwlaw = -0.425, -0.450, -0.467, -0.493, -0.539

	assert energy in allowed_ranges, (
		f"Energy range not covered by model.\n + \
		Options: {allowed_ranges}"
	)
	
	idx = allowed_ranges.index(energy)
	c, p = const[idx], pwlaw[idx]
	radius_au = Rstar * Const.R_sun.to('au').value

	# This relation uses X-ray and EUV fluxes at the stellar surface.
	fxr = lum_to_flux_au(lum = Lx, dist = radius_au) # Xray flux at stellar radius
	reuv = c * (fxr ** p)  # EUV-to-Xrays flux ratio
	Leuv = flux_to_lum_au(flux = reuv * fxr, dist = radius_au)
	return Leuv


def parse_star(state: EvoState, star: Any, ages: list[float]):
	count = len(ages)
	
	if isinstance(star, dict):
		if star.keys() != dict(lx=None, leuv=None, lbol=None).keys():
			raise KeyError("Star dict must only have keys 'lx', 'leuv', and 'lbol'.")
		lengths = np.unique([ len(star[k]) for k in star ])
		assert (len(star['lx']) == count), f"Star Lx track must have length {count}"
		assert (len(star['leuv']) == count), f"Star Leuv track must have length {count}"
		assert (len(star['lbol']) == count), f"Star Lbol track must have length {count}"
	
	elif is_mors_star(star):
		assert max(ages) <= max(star.AgeTrack), "Max age out of bounds for mors.Star" 
		lx_track = np.array([star.Lx(a) for a in ages])
		leuv_track = np.array([star.Leuv(a) for a in ages])
		lbol_track = np.array([star.Lbol(a) for a in ages])
		state.mstar = star.Mstar
		star = dict(lx = lx_track, leuv = leuv_track, lbol = lbol_track)
	
	elif callable(star):
		lx_track, leuv_track, lbol_track = [ np.zeros(count) ] * 3
		for i,a in enumerate(ages):
			lx_track[i], leuv_track[i], lbol_track[i] = star(a)
		star = dict(lx=lx_track, leuv = leuv_track, lbol=lbol_track)
	
	else:
		raise ValueError(
			"Star must be dictionary, Mors.Star instance or callable"
		)

	state.update_fluxes(star['lx'][0], star['leuv'][0], star['lbol'][0])
	return star


def planet_density(mp, rp):
	"""
	mp: mass in Earth masses
	rp: radius in Earth radii
	Returns: density in g/cm^3
	"""
	mass_g = Const.M_earth.to('g').value * mp
	denominator = 4*np.pi/3 * (Const.R_earth.to('cm').value * rp)**3
	return mass_g / denominator


def new_tracks(planet: Planet):
	keys = 'Age', 'Lbol', 'Lx', 'Leuv', 'Lxuv', 'Rp', 'Mp', 'Menv', 'Renv', 'Fenv', 'Dens', 'Mloss', 'Teq'
	return Tracks(base_pl = planet, keys = keys)


def update_tracks(tracks: Tracks, state: EvoState):
	tracks['Age']   += [ state.age ]
	tracks['Lbol']  += [ state.lbol ]
	tracks['Lx']    += [ state.lx ]
	tracks['Leuv']  += [ state.leuv ]
	tracks['Lxuv']  += [ state.lx + state.leuv ]
	tracks['Rp']    += [ state.rp ]
	tracks['Mp']    += [ state.mp ]
	tracks['Menv']  += [ state.menv ]
	tracks['Renv']  += [ state.renv ]
	tracks['Fenv']  += [ state.fenv ]
	tracks['Dens']  += [ planet_density(mp = state.mp, rp = state.rp) ]
	tracks['Mloss'] += [ state.mloss ]
	tracks['Teq']   += [ equilibrium_temperature(state.fbol) ]


def reverse_tracks(tracks: Tracks):
	for k in tracks.keys():
		tracks[k].reverse()
	return tracks


def solve_planet(state: EvoState):
	""" Fills out missing planet parameters with a structure formulation """
	# solve for envelope radius
	if state.fenv is not None:
		if state.fenv > 0.0:
			state.renv = state.struct_fn(**state.as_dict())
			if np.isnan(state.renv):
				nan_callback(state, state.struct_fn, "renv")
		else: state.renv = 0.0
		state.rp = state.rcore + state.renv
	
	# solve for envelope mass
	elif state.renv is not None:
		if state.renv > 0.0:
			state.fenv = fenv_solve(fstruct = state.struct_fn, **state.as_dict())
			if np.isnan(state.fenv):
				nan_callback(state, state.struct_fn, "fenv")
			if state.fenv < 0.0: state.fenv = 0.0
		else: state.fenv = 0.0
		state.mp = state.mcore / (1 - state.fenv)
		state.menv = state.mp * state.fenv
	
	### alternate names
	state.mass = state.mp
	state.radius = state.rp

	return state


def generate_ages(start: float, end: float, tstep: float):
	count = int(abs(start - end)/tstep) + 1
	ages = np.linspace(start, end + tstep, count)
	return ages


def nan_callback(state: EvoState, origin: callable, target: str, **kws):
	"""
	This function is called when a NaN is generated in the simulation,
	most probably by the structure or the mass loss formulation.
	"""
	if globals.verbose or globals.warnings_as_errors:
		warnings.warn(f"Function '{origin.__name__}' returned NaN\n{state}")
	
	if state.nan_callback(state, origin, target) is False:
		state.__dict__[target] = np.nan


def default_user_nan_callback(state: EvoState, origin: callable, target: str) -> bool:
	"""
	Default nan callback if one isn't provided.
	Called when NaN comes up during simulation.
	Should modify state accordingly, and return
	'True' if NaN was dealt with, or 'False' if not.
	"""
	return False




def evolve(
		planet :Planet,
		star   :Union[dict, callable],
		struct :callable,
		mloss  :callable = None,
		mode   :str = 'future',
		**kwargs
	) -> Tracks:
	"""
	Evolves a planet's gaseous envelope forward in time, taking into account
	stellar flux, thermal evolution of the envelope, and atmospheric mass loss.

	Parameters
	----------
	planet	:Planet
		Planet state to use as starting point.
	star	:mors.Star or dict or callable
		Object that provides stellar state at every point of the simulation
		(e.g. bolometric, X-ray, and EUV luminosities).
		If mors.Star instance, luminosities are drawn from the stellar tracks at each step
		using the Star.Lbol, Star.Lx, and Star.Leuv methods.
		If dict, must contain the keys Lbol, Lx, and Leuv, which are arrays of luminosities
		at the ages at which the simulation state evaluated in erg/s.
		If callable, must be a function that is provided with an age in Myr and returns
		three values: Lbol, Lx, and Leuv in erg/s.
	struct	:callable
		Function that takes the simulation state as input and returns the envelope radius
		in Earth radii.
	mloss	:callable or None, default: None
		If None, mass loss will be turned off.
		If callable, must be a function that takes the simulation state and returns
		the mass loss rate in Earth masses per Myr (NOT GRAMS PER SECOND!).
	mode	:str, default: 'future'
		Simulation mode and time direction.
		'future' for forwards time evolution.
		'past' for backwards time evolution.
	timestep	:float, default: 1.0
		Time step of the simulation in Myr.
	end 	:float, default: 5000.0 if mode='future' or 10.0 if mode='past'
		Age in Myr at which to end the simulation.
		If mode is 'future', value must be greater than planet age.
		If mode is 'past, value must be lesser than planet age.	
	ages	:list[float] or None, default: None
		List of ages at which to evaluate the simulation.
		If None, ages are drawn from planet age (start) to end using timestep.
		If list, ages from the provided list are used instead.
	nan_callback	:callable[ [EvoState, Callable, str], bool ]
		Function that is called when a NaN comes up during simulation.
		-- Parameters --
			state  :EvoState, state of the simulation
			origin :Callable, function that returned NaN
			target :str,      state variable name that was calculated to be NaN (unchanged) 
		-- Returns --
			handled :bool, true if NaN was dealt with, False for default behaviour (parameter is set to NaN)
		An example is a callback that keeps the parameter to its previous (not NaN) value,
		avoiding NaNs, e.g.:
			def handle_nan(state, origin, target):
				return True

	Returns
	-------
	tracks	:Tracks
		Object that stores the simulation states at every age as well as helper functions
		for interpolation, etc.

	Raises
	------
	ValueError
	AssertError
	Warning
	UserWarning
	RuntimeWarning
	"""

	# interpret any warning as an error
	if globals.warnings_as_errors is True:
		np.seterr(all="raise")
		warnings.filterwarnings("error")

	# assert isinstance(planet, Planet), "Planet must be an instance of ph.Planet"
	assert callable(struct), "Structure formulation must be a function"
	assert (mode in ['future','past']), "Simulation mode must be 'future' or 'past'"
	assert mloss is None or callable(mloss), "Mass loss formulation must be a function"
	if mloss is None:
		mloss = lambda *a,**kw: 0.0
		if globals.verbose:
			print(" [INFO] Mass loss is turned off")

	state = EvoState()
	tracks = new_tracks(planet)

	# copy planet properties
	for k in planet.__dict__:
		state.as_dict()[k] = planet.__dict__[k]

	# Init static state variables
	state.mode = mode
	state.mloss_factor = -1.0 if mode == 'future' else 1.0
	state.mloss_fn = mloss
	state.struct_fn = struct
	state.tstep = kwargs.get("timestep", 1.0)
	age_lims = [10,5000] if mode == 'future' else [5000,10]
	state.start = state.age if state.age is not None else age_lims[0]
	state.end = kwargs.get("end", age_lims[1])
	state.fenv_limits = kwargs.get("fenv_limits", [1e-4, 1.0]) # lower, upper
	ages = kwargs.get(
		"ages",
		generate_ages(state.start, state.end, state.tstep)
	)
	# NaN callback function gets called when NaNcomes up in sim.
	state.nan_callback = kwargs.get(
		"nan_callback",
		default_user_nan_callback
	)
	# alternative names
	state.mass = state.mp
	state.radius = state.rp

	# Zero-age params
	state.age = ages[0]
	star = parse_star(state = state, star = star, ages = ages)
	for k,v in kwargs.items(): state.as_dict()[k] = v  # append supplied kwargs to state
	state = solve_planet(state) # Apply structure formulation to missing planet

	# Zero-age mass loss (not used on evolution, just saved on tracks)
	state.mloss = state.mloss_fn(**state.as_dict())
	if np.isnan(state.mloss):
		nan_callback(state, state.mloss_fn, "mloss")

	# Save zero-age state
	state.update_fluxes(star['lx'][0], star['leuv'][0], star['lbol'][0])
	update_tracks(tracks, state)
	if globals.verbose: print(state)

	# Main evolution loop
	for i,a in zip( range(1,len(ages)-1), ages[1:]):
		state.age = a

		# [1] Update fluxes
		state.update_fluxes(star['lx'][i], star['leuv'][i], star['lbol'][i])

		# [2] Update mass loss rate
		if state.mloss_fn is None: state.mloss = 0.0

		elif (state.fenv_limits[0] <= state.fenv <= state.fenv_limits[1]):
			# fenv is in allowed range
			new_mloss = state.mloss_fn(**state.as_dict())
			if np.isnan(new_mloss): nan_callback(state,state.mloss_fn,"mloss")
			else: state.mloss = new_mloss
		
		elif state.fenv < state.fenv_limits[0]:
			# fenv is too small
			state.menv, state.fenv = 0.0, 0.0

		elif state.fenv > state.fenv_limits[1]:
			# fenv is too large
			pass

		# [3] Update envelope
		state.menv += state.mloss_factor * state.mloss * state.tstep

		if state.menv <= state.mp * state.fenv_limits[0]: # no envelope
			state.menv = 0.0
			state.fenv = 0.0
			state.mp = state.mcore
			state.rp = state.rcore
			state.renv = 0.0
		else:
			state.mp = state.mass = state.mcore + state.menv
			state.fenv = state.menv / state.mp
			new_renv = state.struct_fn(**state.as_dict())
			if np.isnan(new_renv): nan_callback(state, state.struct_fn, "renv")
			else: state.renv = new_renv
			state.rp = state.radius = state.renv + state.rcore

		update_tracks(tracks, state)
		if globals.verbose: print(state)

	if mode == 'past': tracks = reverse_tracks(tracks)
	tracks.interpolate()
	return tracks



############################################


def generate_star_track(star: Union[dict,object,callable], ages: list[float]):

	if isinstance(star, dict):
		if star.keys() != dict(Lxuv=None,Lbol=None).keys():
			raise KeyError("Star dict must only have keys 'Lxuv' and 'Lbol'.")
		if not indexable(star['Lxuv']) and not indexable(star['Lbol']):
			raise ValueError("Star dict values must be array-like")
		if not (len(star['Lxuv']) == len(star['Lbol']) == len(ages)):
			print(len(star['Lxuv']), len(star['Lbol']), len(ages))
			raise ValueError("Star tracks and ages must have the same length")
		return star
	
	elif is_mors_star(star):
		if max(ages) > max(star.AgeTrack):
			raise ValueError("Max age out of bounds for star object ")
		LxuvTrack = np.array([star.Lx(a)+star.Leuv(a) for a in ages])
		LbolTrack = np.array([star.Lbol(a) for a in ages])
		return dict(Lxuv=LxuvTrack, Lbol=LbolTrack)
	
	elif callable(star):
		LxuvTrack , LbolTrack = [None]*len(ages), [None]*len(ages)
		for i,a in enumerate(ages):
			LxuvTrack[i], LbolTrack[i] = star(a)
		return dict(Lxuv=LxuvTrack, Lbol=LbolTrack) 
	
	else:
		raise ValueError(
			"Star must be dictionary, Mors.Star instance or callable"
		)

def _init_tracks():
	fields = ('Age', 'Lbol', 'Lxuv', 'Rp', 'Mp', 'Menv', 'Renv', 'Fenv', 'Dens', 'Mloss')
	tracks = {f:[] for f in fields}
	return tracks

def _planet_density(mp, rp):
	return planet_density(mp, rp)

def _update_tracks(tracks, planet, star, i=0, **kwargs):
	tracks['Age'].append(planet.age)
	#tracks['Lbol'].append(star.Lbol(planet.age))
	tracks['Lbol'].append(star['Lbol'][i])
	#tracks['Lxuv'].append(star.Lx(planet.age)+star.Leuv(planet.age))
	tracks['Lxuv'].append(star['Lxuv'][i])
	tracks['Rp'].append(planet.rp)
	tracks['Mp'].append(planet.mp)
	tracks['Menv'].append(planet.menv)
	tracks['Renv'].append(planet.renv)
	tracks['Fenv'].append(planet.fenv)
	tracks['Dens'].append( _planet_density(mp=planet.mp, rp=planet.rp) )
	tracks['Mloss'].append( kwargs['mloss'] )
	return tracks

def _reverse_tracks(tracks):
	for k in tracks.keys():
		tracks[k].reverse()
	return tracks

def _update_params(params, planet, star, i=0, **kwargs):
	fbol = star['Lbol'][i] / (planet.dist*Const.au.to('cm').value)**2 / (4*np.pi)
	params.update(planet.__dict__)
	params['fbol'] = fbol
	params['mass'] = planet.mp
	params['radius'] = planet.rp
	#params['Lxuv'] = star.Lx(planet.age) + star.Leuv(planet.age)
	#params['Lbol'] = star.Lbol(planet.age)
	params['Lxuv'] = star['Lxuv'][i]
	params['Lbol'] = star['Lbol'][i]
	for k in kwargs: params[k] = kwargs[k]
	return params


def evolve_forward(planet, mloss, struct, star, time_step=1.0, age_end=1e4,\
	ages=None, **kwargs):
	"""
	Evolves a planet's gaseous envelope forward in time, taking into account
	stellar high energy emission, natural cooling over time, and
	photoevaporative mass loss.

	Parameters
	----------
	planet : photoevolve.core.Planet instance
			Planet to be evolved. Its data will not be modified.
	mloss : callable
			Mass loss formulation, preferably the functions defined in the
			'photoevolver.massloss' module.
	struct : callable
			Function that relates the envelope radius to the envelope mass
			fraction and vice-versa, preferably the functions defined in
			the 'photoevolver.structure' module.
	star : Mors.Star | dict | callable
			Description of X-ray evolution of the host star. Defines the
			XUV and bolometric luminosities of the star at any given age.
			 - Mors.Star instance: the luminosities will be drawn from
			the Lx, Leuv, and Lbol tracks.
			 - dict: must have keys 'Lxuv' and 'Lbol', each being an array
			of appropriate length containing the luminosity at each age.
			Must be sorted from young to old.
			 - callable: NOT IMPLEMENTED. Must be a function that takes 
			the age in Myr as its single argument, and returns an array of two 
			values, Lxuv and Lbol, in erg/s. E.g. f(age) -> [Lxuv, Lbol]
	time_step : float, optional
			Time step of the simulation in Myr. Default is 1 Myr.
	age_end : float, optional
			Age at which to end the simulation. Must be greater than the
			planet age if evolving forward, and smaller if evolving back.
			Default is 10 000 Myr.
	ages : array_like, optional 
			Array of ages at which to update planet parameters.
			Must be sorted from young to old.
			If None (default), it is ignored and time_step and age_end
			are used instead.
			If array_like, the age of each time step is drawn from it,
			and the simulation will run for N-1 steps, where N is the
			length of the array.
	eff : float, optional
			Mass loss efficiency (0.0 to 1.0).
			Only used if using 'photoevolver.massloss.EnergyLimited'
			formulation as the mass loss.
	beta : float, optional
			XUV radius of the planet as a factor of its optical radius
			(Rxuv / Rp).
			Only used if using 'photoevolver.massloss.EnergyLimited'
			formulation as the mass loss.
	mstar : float, optional
			Mass of the host star in solar masses.
			Required if using ...
			If the input star is a Mors.Star instance, the star mass is
			retrieved from it.
	fenv_min : float, optional
			Minimum envelope mass fraction at which to consider the planet
			to be completely rocky. Default is ...
	renv_min : float, optional
			Minimum envelope radius at which to consider the planet
			to be completely rocky. Default is ...

	Returns
	-------
	photoevolver.core.Tracks
		The evolutionary tracks that describe how each of planet's
		parameters evolves in the time range given.

	"""
	if globals.warnings_as_errors is True:
		globals.verbose = True
		warnings.filterwarnings("error")

	pl = deepcopy(planet)
	if not isinstance(pl, Planet):
		raise ValueError("The planet must be an instance of the \
				photoevolve.core.Planet class")
	if not callable(struct):
		raise ValueError("struct must be a function")
	if ages is None and age_end <= pl.age:
		raise ValueError("The maximum age must be greater than the planet's age")
	if ages is not None and ages[0] < pl.age:
		raise ValueError("The starting age must be greater or equal \
				to the planet's age")

	# kwargs check
	if 'fenv_min' not in kwargs: kwargs['fenv_min'] = 1e-5
	if 'renv_min' not in kwargs: kwargs['renv_min'] = 0.05
	if 'mstar' not in kwargs and is_mors_star(star):
		kwargs['mstar'] = star.Mstar
		kwargs['mors_star'] = star
	
	if ages is None:
		length = int(abs(pl.age - age_end)/time_step) + 1
		ages = np.linspace(pl.age, age_end+time_step, length)
	star = generate_star_track(star, ages)

	tracks = _init_tracks()
	params = _update_params(kwargs, pl, star, i=0, mloss = 0.0)

	# Calculate current envelope properties
	# - radii
	if pl.fenv is not None:
		if pl.fenv > kwargs['fenv_min']:
			pl.renv = struct(**params)
		else: pl.renv = 0.0
		pl.rp = pl.rcore + pl.renv
	# - masses
	elif pl.renv is not None:
		if pl.renv > kwargs['renv_min']: 
			pl.fenv = fenv_solve(fstruct=struct, **params)
			if pl.fenv < 0.0: pl.fenv = 0.0
		else: pl.fenv = 0.0
		pl.mp = pl.mcore / (1 - pl.fenv)
		pl.menv = pl.mp * pl.fenv
	
	# MAIN EVOLUTION LOOP
	for i,a in enumerate(ages):
		if (i >= len(ages)-1): break # skip last age
		pl.age = a
		tstep = abs(pl.age - ages[i+1])
		params = _update_params(params, pl, star, i=i)
		mloss_rate = 0.0
		if mloss is not None and pl.fenv > kwargs['fenv_min']:
			mloss_rate = mloss(**params)
			if np.isnan(mloss_rate):
				if globals.verbose: warnings.warn(f"Mass loss '{mloss.__name__}' model returned nan, params are: {params}")
				mloss_rate = 0.0
			pl.menv -= mloss_rate * tstep
			pl.mp = pl.mcore + pl.menv 
			pl.fenv = pl.menv / pl.mp
			params = _update_params(params, pl, star, i=i, mloss = mloss_rate)
		elif pl.fenv < kwargs['fenv_min'] or pl.rp < pl.rcore:
			pl.fenv = 0.0
			pl.renv = 0.0
			pl.rp = pl.rcore
		if pl.renv > kwargs['renv_min']:
			pl.renv = struct(**params)
			if np.isnan(pl.renv):
				if globals.verbose: warnings.warn(f"Warning: structure model '{struct.__name__}' returned nan, params are: {params}")
				pl.renv = pl.rp - pl.rcore
			if pl.renv <= kwargs['renv_min']: pl.renv = 0.0
			pl.rp = pl.rcore + pl.renv
		tracks = _update_tracks(tracks, pl, star, i=i, mloss = mloss_rate)
	return Tracks(data = tracks, base_pl = pl)


def evolve_back(planet, mloss, struct, star, time_step=1.0, age_end=1.0, ages = None, **kwargs):
	"""
	Evolves a planet's gaseous envelope backwards in time, taking into account
	stellar high energy emission, natural cooling over time, and
	photoevaporative mass loss.

	Parameters
	----------
	planet : photoevolve.core.Planet instance
			Planet to be evolved. Its data will not be modified.
	mloss : callable
			Mass loss formulation, preferably the functions defined in the
			'photoevolver.massloss' module.
	struct : callable
			Function that relates the envelope radius to the envelope mass
			fraction and vice-versa, preferably the functions defined in
			the 'photoevolver.structure' module.
	star : Mors.Star object | dict | callable
			Description of X-ray evolution of the host star. Defines the
			XUV and bolometric luminosities of the star at any given age.
			 - Mors.Star instance: the luminosities will be drawn from
			the Lx, Leuv, and Lbol tracks.
			 - dict: must have keys 'Lxuv' and 'Lbol', each being an array
			of appropriate length containing the luminosity at each age.
			Must be sorted from young to old.
			 - callable: NOT IMPLEMENTED. Must be a function that takes 
			the age in Myr as its single argument, and returns an array of two 
			values, Lxuv and Lbol, in erg/s. E.g. f(age) -> [Lxuv, Lbol]
	time_step : float, optional
			Time step of the simulation in Myr. Default is 1 Myr.
	age_end : float, optional
			Age at which to end the simulation. Must be greater than the
			planet age if evolving forward, and smaller if evolving back.
			Default is 10 000 Myr.
	ages : array_like, optional 
			Array of ages at which to update planet parameters.
			Must be sorted from min to max.
			If None (default), it is ignored and time_step and age_end
			are used instead.
			If array_like, the age of each time step is drawn from it,
			and the simulation will run for N-1 steps, where N is the
			length of the array.
	**eff : float, optional
			Mass loss efficiency (0.0 to 1.0).
			Only used if using 'photoevolver.massloss.EnergyLimited'
			formulation as the mass loss.
	**beta : float, optional
			XUV radius of the planet as a factor of its optical radius
			(Rxuv / Rp).
			Only used if using 'photoevolver.massloss.EnergyLimited'
			formulation as the mass loss.
	**mstar : float, optional
			Mass of the host star in solar masses.
			Required if using ...
			If the input star is a Mors.Star instance, the star mass is
			retrieved from it.
	**fenv_min : float, optional
			Minimum envelope mass fraction at which to consider the planet
			to be completely rocky. Default is ...
	**renv_min : float, optional
			Minimum envelope radius at which to consider the planet
			to be completely rocky. Default is ...

	Note: parameters starting with '**' are keyword arguments (**kwargs)
		and should be called without the double asterisks.

	Returns
	-------
	tracks : photoevolver.core.Tracks instance
			The evolutionary tracks that describe how each of planet's
			parameters evolves in the time range given.

	"""
	# input parameters check
	if globals.warnings_as_errors is True:
		globals.verbose = True
		warnings.filterwarnings("error")

	pl = deepcopy(planet)
	if type(pl) != Planet:
		raise ValueError("the planet must be an instance of the \
				photoevolve.core.Planet class")
	if not callable(struct):
		raise ValueError("struct must be a function")
	if ages is None and age_end >= pl.age:
		raise ValueError("The miminum age must be lower than the planet's age")
	if ages is not None and ages[-1] > pl.age:
		raise ValueError("The oldest age must be lower or equal \
				to the planet's age")

	# kwargs check
	if 'fenv_min' not in kwargs: kwargs['fenv_min'] = 1e-9
	if 'renv_min' not in kwargs: kwargs['renv_min'] = 0.001
	if 'mstar' not in kwargs and is_mors_star(star):
		kwargs['mstar'] = star.Mstar 
		kwargs['mors_star'] = star

	if ages is None:
		length = int(abs(pl.age - age_end)/time_step) + 1
		ages = np.linspace(pl.age, age_end+time_step, length)
	star = generate_star_track(star, ages)

	# Zero envelope fix
	if pl.fenv is not None and pl.fenv < kwargs['fenv_min']:
		pl.fenv = kwargs['fenv_min']
	if pl.renv is not None and pl.renv < kwargs['renv_min']:
		pl.renv = kwargs['renv_min']
	
	tracks = _init_tracks()
	params = _update_params(kwargs, pl, star, i=0, mloss = 0.0)
   
	# Update current planet parameter, i=0s
	# - radii 
	if pl.renv is None: pl.renv = struct(**params)
	pl.rp = pl.rcore + pl.renv
	# - masses
	if pl.fenv is None: pl.fenv = fenv_solve(fstruct=struct, **params)
	if pl.fenv < 0.0: pl.fenv = 0.0
	pl.mp = pl.mcore / (1 - pl.fenv)
	pl.menv = pl.mp * pl.fenv

	for i,a in enumerate(ages):
		if (i == len(ages)-1): break
		pl.age = a
		tstep = abs(a - ages[i+1])
		params = _update_params(params, pl, star, i)
		mloss_rate = 0.0
		if mloss is not None:
			mloss_rate = mloss(**params)
			if np.isnan(mloss_rate):
				if globals.verbose: warnings.warn(f"Warning: mass model '{mloss.__name__}' returned nan, params are: {params}")
				mloss_rate = 0.0
			pl.menv += mloss_rate * tstep
			pl.mp = pl.mcore + pl.menv 
			pl.fenv = pl.menv / pl.mp
			params = _update_params(params, pl, star, i, mloss = mloss_rate)

		pl.renv = struct(**params)
		if np.isnan(pl.renv):
			if globals.verbose: warnings.warn(f"Warning: structure model '{struct.__name__}' returned nan, params are: {params}")
			pl.renv = pl.rp - pl.rcore
		pl.rp = pl.rcore + pl.renv
		tracks = _update_tracks(tracks, pl, star, i, mloss = mloss_rate)
	
	tracks = _reverse_tracks(tracks)
	return Tracks(data = tracks, base_pl = pl)


def plot_tracks(*tracks):
	# Parameters: tracks, labels, colors, linestyles
	if len(tracks) == 0: return
	fields = tracks[0].keys()
	colors = "rgbcmyk"
	lines = ('-', '--', '-.', ':')
	fmts = [c+l for l in lines for c in colors]
	fmts *= int(1+len(tracks)/len(fields))
	for i,f in enumerate(fields):
		if f == 'Age': continue
		plt.xlabel('Age [Myr]')
		plt.ylabel(f)
		plt.xscale('log')
		for j in range(len(tracks)):
			if len(tracks[j][f]) != len(tracks[j]['Age']): continue
			plt.plot(tracks[j]['Age'], tracks[j][f], fmts[j])
		plt.show()
		








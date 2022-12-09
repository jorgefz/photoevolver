

import numpy as np
from typing import Any, Union, List, Callable
import uncertainties as uncert
from uncertainties import ufloat, wrap as uwrap
from scipy.optimize import fsolve as scipy_fsolve
from astropy import units

from .utils import flux, luminosity

from .owenwu17 import mass_to_radius as owen_radius, radius_to_mass as owen_mass
from .structure import ChenRogers16


def Otegi20RockyMass(radius :float, no_errors :bool = False) -> float:
	"""Calculates the rocky mass from a radius using the M-R relations by Otegi+20"""
	scaling = ufloat(0.90, 0.06)
	exponent = ufloat(3.45, 0.12)
	if no_errors is True:
		return scaling.nominal_value * radius ** exponent.nominal_value
	return scaling * radius ** exponent

def Otegi20RockyRadius(mass :float, no_errors :bool = False) -> float:
	"""Calculates the rocky radius from a mass using the M-R relations by Otegi+20"""
	scaling = ufloat(1.03, 0.02)
	exponent = ufloat(0.29, 0.01)
	if no_errors is True:
		return scaling.nominal_value * mass ** exponent.nominal_value
	return scaling * mass ** exponent

def Otegi20MR(**kwargs) -> float:
	"""Calculates rocky radius if mass provided, and rocky mass if radius provided"""
	if 'mass' in kwargs:
		return Otegi20RockyRadius(kwargs['mass'], kwargs.get('no_errors',False))
	elif 'radius' in kwargs:
		return Otegi20RockyMass(kwargs['radius'], kwargs.get('no_errors',False))
	else:
		raise KeyError("[Otegi20MR] Specify either 'mass' or 'radius'")


def OtegiMassRadiusRelation(**kwargs) -> float:
	"""LEGACY FUNCTION: mass-radius relation for rocky cores by Otegi+20"""
	print("[OtegiMassRadiusRelation] Warning: deprecated function")
	return Otegi20MR(**kwargs).nominal_value


# Wrapper for EvapMass M-R relation
def OwenMassRadiusRelation(**kwargs):
	"""
	Parameters
	----------
	mass : float
	radius : float
	Xice : float
	Xiron : float
	"""
	has_mass = 'mass' in kwargs.keys()
	has_radius = 'radius' in kwargs.keys()
	if (has_mass and has_radius) or (not has_mass and not has_radius):
		print(f" Error: specify either mass or radius")
		return 0.0
	if 'Xice'  not in kwargs.keys(): kwargs['Xice']  = 0.0
	if 'Xiron' not in kwargs.keys(): kwargs['Xiron'] = 1/3
	if has_mass: return owen_radius(**kwargs)
	if has_radius: return owen_mass(**kwargs)[0]


def solve_structure(
		mass 	:float,
		radius 	:float,
		age  	:float,
		dist  	:float,
		lbol  	:float,
		env_fn 	:callable = ChenRogers16,
		mr_fn 	:callable = Otegi20MR,
		fenv_start :float = 0.05,
		no_errors  :bool  = False
		):
	"""
	Solves for the structure of a planet given its observed mass and radius,
	as well as orbital and stellar parameters.
	Accepts variables with uncertainties and returns planet parameters
	with calculated uncertainties as well.
	
	Parameters
	----------
		mass:	(float, ufloat) -> Mass of the planet in Earth masses.
		radius:	(float, ufloat) -> Radius of the planet in Earth radii.
		age:	(float, ufloat) -> Radius of the planet in Earth radii.
		dist:	(float, ufloat) -> Semimajor-axis of the planet's orbit in AU.
		lbol:	(float, ufloat) -> Bolometric luminosity of the host star in erg/s.
		env_fn:	(callable) -> Envelope structure formulation of signature:
				Parameters: (float) -> mass, fenv, fbol, age, rcore, dist
				Returns: 	(float) -> envelope_radius
				By default, it uses Chen & Rogers (2016).
		mr_fn:	(callable) -> Mass-radius relation for rocky cores of signature:
				Keyword parameters: mass (float)
				Returns: (ufloat) -> radius 
				By default, it uses Otegi et al. (2020).
		fenv_start: (float) -> Initial guess for the envelope mass fraction.
					By default, it is set to 0.05 (envelope_mass => 5% of planet_mass)
		no_errors:  (bool) -> If True, no uncertainties will be calculated for the final result

	Returns
	-------
		(dict) -> Solved internal structure of the planet as dict keys in relevant units:
			mass, radius, mcore, rcore, fenv, renv, age, dist, success.
			The `success` key is a boolean that specifies whether a solution was found.
	"""
	solver_success = True # Determines whether solver converged to a solution

	def diff_fn(fenv, mass, radius, age, dist, lbol):
		"""Returns difference in envelope radii from M-R relation and structure model"""
		# Calculate envelope radius from mass-radius relation
		mcore = (1.0 - fenv) * mass
		
		rcore = mr_fn(mass = mcore)[0].nominal_value
		renv1 = radius - rcore
		# Calculate envelope radius from envelope structure formulation
		fbol = flux(lbol, dist_au = dist)
		renv2 = env_fn(
			mass=mass, fenv=fenv, fbol=fbol,
			age=age, rcore=rcore, dist=dist
		)
		# print("Diff: ", renv1 - renv2)
		return renv1 - renv2
	
	def solver_fn(mass, radius, age, dist, lbol, fenv_start):
		"""Solves for the envelope mass fraction without uncertainties"""
		fenv, info, success, msg = scipy_fsolve(
			func = diff_fn,
			x0 = fenv_start, full_output = True,
			args = (mass, radius, age, dist, lbol)
		)
		if success != 1:
			print("[solve_structure] Failed to reach a solution") 
			solver_success = False
		return float(fenv)
	
	# wrap functions to accept or rule out uncertainties
	solver = uwrap(solver_fn) if no_errors is False else solver_fn
	mr     = mr_fn            if no_errors is False else lambda **kw: mr_fn(**kw).nominal_value

	# Solve for envelope and recalculate planet parameters
	fenv = solver(mass, radius, age, dist, lbol, fenv_start)
	mcore = (1.0 - fenv) * mass
	rcore = mr(mass = mcore)
	renv = radius - rcore
	
	params = dict(
		mass  = mass,  radius = radius,
		age   = age,   dist   = dist,
		mcore = mcore, rcore  = rcore,
		fenv  = fenv,  renv   = renv,
		success = solver_success
	)

	return params



class Planet:
	def __init__( \
			self,
			mp   : float = None, rp   : float = None,
			mcore: float = None, rcore: float = None,
			menv : float = None, renv : float = None,
			fenv : float = None, age  : float = None,
			dist : float = None, mrfunc: callable = None,
			**kwargs: dict[Any]
		) -> None:

		"""
		Stores a planet state at a point in time.

		Parameters
		----------
			mp:		float -> Planet mass (M_earth)
			rp:		float -> Planet radius (R_earth)
			mcore:	float -> Core mass (M_core)
			rcore:	float -> Core radius (R_core)
			menv:	float -> Envelope mass (M_earth)
			renv:	float -> Envelope radius (R_earth)
			fenv:	float -> Envelope mass fraction (M_env/M_planet)
			dist:	float -> Separation between the planet and the star (AU)
			age:	float -> Age of the planet (Myr)
			mrfunc:	callable -> Mass-Radius relation to use to calculate core radius.
            **kwargs: dict[Any] -> Extra arguments; these will be passed to the mass-radius relation function.

		Returns
		-------
			Planet

		Raises
		------
			ValueError -> if not enough parameters are provided or some have wrong values
		"""
		self.mp = mp
		self.rp = rp
		self.mcore = mcore
		self.rcore = rcore
		self.menv = menv
		self.renv = renv
		self.fenv = fenv
		self.dist = dist
		self.age = age
		self.mrfunc = mrfunc
		self.args = kwargs
		# --
		self.input_check()

	def input_check(self):
		if self.dist is None or self.age is None:
			raise ValueError("age or orbital distance undefined")

		if self.mrfunc is None or not callable(self.mrfunc):
			self.mrfunc = OtegiMassRadiusRelation
		
		if self.mcore is None and self.rcore is None:
			raise ValueError("core mass or radius undefined")
		elif self.mcore is None: self.mcore = self.mrfunc(radius=self.rcore, **self.args)
		elif self.rcore is None: self.rcore = self.mrfunc(mass=self.mcore, **self.args)

		if self.rcore is None:
			print(f" Warning: core radius undefined.", end='')
			print(f" Will be estimated from a mass-radius relation ", end='')
			print(f" -> {self.rcore:.3f} Earth radii")

		elif self.mcore is None:
			print(f" Warning: core mass undefined.", end='')
			print(f" Will be estimated from a mass-radius relation ", end='')
			print(f" -> {self.mcore:.3f} Earth masses")

		# Radii known (unknown menv, mp, fenv)
		if self.rp is not None:
			self.renv = self.rp - self.rcore if self.renv is None else self.renv
		elif self.renv is not None:
			self.rp = self.rcore + self.renv if self.rp is None else self.rp
		
		# Masses known (unknown renv, rp)
		elif self.mp is not None:
			self.menv = self.mp - self.mcore if self.menv is None else self.menv
			self.fenv = self.menv / self.mp
		elif self.fenv is not None:
			self.menv = self.fenv * self.mcore / (1 - self.fenv) if self.menv is None else self.menv
			self.mp   = self.menv + self.mcore		 if self.mp   is None else self.mp

		elif self.fenv is None and self.renv is None:
			raise ValueError(" Error: either envelope radius (renv) or mass fraction (fenv) must be defined")
		else: raise ValueError("not enough planet parameters defined")


	def __repr__(self):
		values = self.mp, self.rp, self.mcore, self.rcore, self.menv, self.renv, self.fenv, self.age, self.dist
		names =  "Mplanet", "Rplanet", "Mcore", "Rcore", "Menv", "Renv", "Fenv", "Age", "Dist" 
		units =  "Me", "Re", "Me", "Re", "Me", "Re", "%", "Myr", "AU"

		msg = f"<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}>\n"
		for n,v,u in zip(names,values,units):
			msg += f"{n}:".ljust(10)
			if n == "Fenv":
				msg += "TBD".ljust(10) if v is None else f"{v*100:.5f}".ljust(10)
			else:
				msg += "TBD".ljust(10) if v is None else f"{v:.2f}".ljust(10)
			msg += f"{u} \n"
		return msg

	def __str__(self):
		return self.__repr__()

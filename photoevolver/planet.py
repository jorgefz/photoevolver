

import numpy as np
from typing import Any, Union, List, Callable

from .owenwu17 import mass_to_radius as owen_radius, radius_to_mass as owen_mass


# Utils functions
def indexable(obj):
	return hasattr(obj, '__getitem__')

def is_mors_star(obj):
	try:
		import Mors
		return isinstance(obj, Mors.Star)
	except ImportError:
		return False


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


def OtegiMassRadiusRelation(**kwargs):
	"""
	Computes mass or radius based on planet's mass following the mass-radius
	relationships by Otegi et al. (2015).
	https://ui.adsabs.harvard.edu/abs/2020A%26A...634A..43O/abstract

	Parameters
	----------
	radius : float, optional*
			Radius in Earth radii
	mass : float, optional*
			Mass in Earth masses. Only specify EITHER mass OR radius!!
	mode : str, optional
			Choose: "rocky", "volatile"

	*Note: EITHER mass OR radius must be given.
	
	Returns
	-------
	mass/radius : float
			Radius if input mass, Mass if input radius.
	error : float
			Error in mass or radius, if error provided

	"""
	has_mass = 'mass' in kwargs.keys()
	has_radius = 'radius' in kwargs.keys()
	if (has_mass and has_radius) or (not has_mass and not has_radius):
		print(f" Error: specify either mass or radius")
		return None
	if has_mass: return otegi2020_radius(mass=kwargs['mass'])
	return otegi2020_mass(radius=kwargs['radius'])


def otegi2020_mass(radius, error=0.0, mode='rocky'):
	otegi_models = [
		dict(mode='rocky', const=1.03, c_err=0.02, pow=0.29, p_err=0.01),
		dict(mode='volatile', const=0.7, c_err=0.11, pow=0.63, p_err=0.04)
	]
	if(mode not in ['rocky','volatile']):
		print(f"Error: unknown mode '{mode}'. Setting to rocky.")
		mode = 'rocky'
	model = otegi_models[0] if mode=='rocky' else otegi_models[1]
	mass = (radius/model['const'])**(1/model['pow'])
	if error < 1e-3: return mass
	
	import uncertainties as uncert
	R = uncert.ufloat(radius, error)
	modelC = uncert.ufloat(model['const'], model['c_err'])
	modelP = uncert.ufloat(model['pow'], model['p_err'])
	mass = (R / modelC) ** (1 / modelP)
	return mass.nominal_value, mass.std_dev


def otegi2020_radius(mass, error=0.0, mode='rocky'):
	otegi_models = [
		dict(mode='rocky', const=1.03, c_err=0.02, pow=0.29, p_err=0.01),
		dict(mode='volatile', const=0.7, c_err=0.11, pow=0.63, p_err=0.04)
	]
	if(mode not in ['rocky','volatile']):
		print(f"Error: unknown mode '{mode}'. Setting to rocky.")
		mode = 'rocky'
	model = otegi_models[0] if mode=='rocky' else otegi_models[1]
	radius = model['const'] * mass ** model['pow']
	if error < 1e-3: return radius
	
	# Errors
	"""
	M = uncert.ufloat(mass, error)
	modelC = uncert.ufloat(model['const'], model['c_err'])
	modelP = uncert.ufloat(model['pow'], model['p_err'])
	"""
	const_term = model['c_err'] * mass ** model['pow']
	mass_term =  model['const'] \
				* model['pow'] \
				* error \
				* mass ** (model['pow']-1)
	pow_term = model['const'] \
				* mass ** model['pow'] \
				* np.log(mass) \
				* model['p_err']
	radius_err = np.sqrt( (const_term)**2 + (mass_term)**2 + (pow_term)**2 )
	return radius, radius_err




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

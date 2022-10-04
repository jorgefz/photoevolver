"""
[photoevolver.massloss]

Includes several atmosoheric mass loss formulations
"""

import numpy as np
import astropy.constants as const
from astropy import units

from .K18interp import K18Interpolator
 
def _keyword_check(keywords, params):
	for f in keywords:
		if f not in params:
			raise KeyError(f"model parameter '{f}' undefined")

def _bound_check(bounds, params):
	for f in bounds:
		if f not in params: continue
		if not (bounds[f][0] <= params[f] <= bounds[f][1]):
			raise ValueError(f"model parameter '{f}' out of safe bounds ({bounds[f][0]},{bounds[f][1]})")


def salz16_beta(**kwargs):
	"""
	Parameters
		fxuv: erg/cm2/s
		mp: Earth masses
		rp: Earth radii
	"""
	mp = kwargs['mp']
	rp = kwargs['rp']
	fxuv = kwargs['fxuv'] if 'fxuv' in kwargs else kwargs['Lxuv'] / (4*np.pi*(kwargs['dist']*units.au.to('cm'))**2)
	potential = const.G.to('erg*m/g^2').value * mp * const.M_earth.to('g').value / (rp * const.R_earth.to('m').value)
	log_beta = -0.185*np.log10(potential) + 0.021*np.log10(fxuv) + 2.42
	if log_beta < 0.0: log_beta = 0.0
	# upper limit to beta
	if 10**log_beta > 1.05 and potential < 1e12: log_beta = np.log10(1.05)
	return 10**(log_beta)

def salz16_eff(**kwargs):
	"""
	Parameters
		mp: Planet mass in Earth masses
		rp: Planet radius in Earth radii
	"""
	mp = kwargs['mp']
	rp = kwargs['rp']
	potential = const.G.to('erg*m/g^2').value * mp * const.M_earth.to('g').value / (rp * const.R_earth.to('m').value)
	v = np.log10(potential)
	if   ( v < 12.0):		   log_eff = np.log10(0.23) # constant
	if   (12.0  < v <= 13.11):  log_eff = -0.44*(v-12.0) - 0.5
	elif (13.11 < v <= 13.6):   log_eff = -7.29*(v-13.11) - 0.98
	elif (v > 13.6):			log_eff = -7.29*(13.6-13.11) - 0.98 # stable atmospheres, no evaporation (< 1e-5)
	return 10**(log_eff)*5/4 # Correction evaporation efficiency to heating efficiency


def EnergyLimited(**kwargs):
	"""
	Calculates the atmospheric mass loss rate driven by photoevaporation
	This is based on the energy balance between stellar influx and the potential of the planet.
	Sources: Watson et al (1981), Lecavelier des Etangs (2007), Erkaev (2007).

	Required keywords:
		mass: planet M_earth
		radius: planet R_earth
		Lxuv: XUV luminosity of the star in erg/s
		dist: planet-star separation in AU
		mstar: M_sun

	Optional keywords:
		safe: checks if the input parameters are within safe model bounds.
		eff: mass loss efficiency. Use value (e.g. 0.15 for 15%) or formulation: 'salz16'.
		beta: XUV radius to optical radius ratio.

	Returns:
		mloss: mass loss rate (M_earth per Myr)

	"""
	# --
	req_kw = ['radius', 'mass', 'Lxuv', 'dist', 'mstar']
	_keyword_check(req_kw, kwargs)
	# --
	bounds = {
			"radius": [0.5,   50.0],
			"mass":   [0.5, 20.0],
			"Lxuv":   [1.0, 1e38],
			"dist":   [0.01,   100.0],
			"mstar":  [0.5, 2.5]
	}
	if 'safe' in kwargs and kwargs['safe'] is True: _bound_check(bounds, kwargs)
	# --
	# Unit conversions
	kwargs['Lxuv']   *= 1e-7 # erg/s to Watt
	kwargs['mstar']  *= const.M_sun.value # M_sun to kg
	kwargs['mass']   *= const.M_earth.value # M_earth to kg
	kwargs['radius'] *= const.R_earth.value # R_earth to m
	kwargs['dist']   *= const.au.value # AU to m
	Fxuv = kwargs['Lxuv'] / ( 4 * np.pi * (kwargs['dist'])**2 )
	# Variable efficiency and Rxuv
	if 'eff' not in kwargs: kwargs['eff'] = 0.15
	elif kwargs['eff'] == 'salz16': kwargs['eff'] = salz16_eff(kwargs['mass']/const.M_earth.value, kwargs['radius']/const.R_earth.value)
	elif type(kwargs['eff']) is str: kwargs['eff'] = 0.15

	if 'beta' not in kwargs: kwargs['beta'] = 1.0
	elif kwargs['beta'] == 'salz16':
		kwargs['beta'] = salz16_beta(fxuv=Fxuv*1e3, mp=kwargs['mass']/const.M_earth.value, rp=kwargs['radius']/const.R_earth.value)
	elif type(kwargs['beta']) is str: kwargs['beta'] = 1.0
	# Energy-limited equation
	xi =( kwargs['dist'] / kwargs['radius'] ) * ( kwargs['mass'] / kwargs['mstar'] / 3)**(1/3)
	K_tide = 1 - 3/(2*xi) + 1/(2*(xi)**3) 
	mloss = kwargs['beta']**2 * kwargs['eff'] * np.pi * Fxuv * kwargs['radius']**3 / (const.G.value * K_tide * kwargs['mass'])
	return mloss * 5.28e-12 # Earth masses per Myr



def Kubyshkina18(**kwargs):
	"""
	Calculates the atmospheric mass loss rate driven by photoevaporation
	This is based on the hydrodynamic models by Kubyshkina et al (2018)

	Required keywords:
		mass: planet M_earth
		radius: planet R_earth
		Lxuv: XUV luminosity of the star in erg/s
		Lbol: bolometric luminosity in erg/s
		dist: planet-star separation in AU

	Optional keywords:
		safe: (bool) checks if the input parameters are within safe model bounds.

	Returns:
		mloss: mass loss rate (M_earth per Myr)

	"""
	# --
	req_kw = ['mass', 'radius', 'Lxuv', 'Lbol', 'dist']
	_keyword_check(req_kw, kwargs)
	# --
	bounds = {
			"radius": [1.0,  39.0],
			"mass":   [1.0,  10.0],
			"Lxuv":   [1e26, 5e30],
			"Lbol":   [1.0,  1e40],
			"dist":   [0.002, 1.3]
	}
	if 'safe' in kwargs and kwargs['safe'] is True: _bound_check(bounds, kwargs)
	# --
	# constants and parameters
	large_delta = {
		'beta':  16.4084,
		'alpha': [1.0, -3.2861, 2.75],
		'zeta':  -1.2978,
		'theta': 0.8846
	}
	small_delta = {
		'beta': 32.0199,
		'alpha': [0.4222, -1.7489, 3.7679],
		'zeta': -6.8618,
		'theta': 0.0095
	}

	def Epsilon(rp, Fxuv, dist):
		numerator = 15.611 - 0.578*np.log(Fxuv) + 1.537*np.log(dist) + 1.018*np.log(rp)
		denominator = 5.564 + 0.894*np.log(dist)
		return numerator / denominator

	mp = kwargs['mass']
	rp = kwargs['radius']
	Lxuv = kwargs['Lxuv']
	Lbol = kwargs['Lbol']
	dist = kwargs['dist']

	conv = (units.erg / units.cm**2 / units.s).to('W/m^2') # erg/cm^2/s to W/m^2
	Fxuv = Lxuv / (4 * np.pi * (dist*const.au.to('cm').value)**2 )
	Fbol = Lbol / (4 * np.pi * (dist*const.au.to('cm').value)**2 )
	Teq =  ( Fbol * conv / (4*const.sigma_sb.value) )**(1/4)
	mH = const.m_p.value +  const.m_e.value # H-atom mass (kg)

	Jeans_param = const.G.value * (mp*const.M_earth.value) * (mH) / (const.k_B.value * Teq * (rp*const.R_earth.value) )
	eps = Epsilon(rp, Fxuv, dist) 
	xp = small_delta if Jeans_param < np.exp(eps) else large_delta
	Kappa = xp['zeta'] + xp['theta']*np.log(dist)
	mloss = np.exp(xp['beta']) * (Fxuv)**xp['alpha'][0] * (dist)**xp['alpha'][1] * (rp)**xp['alpha'][2] * (Jeans_param)**Kappa

	return mloss * 5.28e-15 # g/s to M_earth/Myr



def EquilibriumTemperature(Lbol, a):
	"""Lbol(erg/s), a(AU), returns T(K)"""
	top = Lbol * units.erg.to('J')
	bottom = 16 * const.sigma_sb.value * np.pi * (a * units.au.to('m'))**2
	return (top/bottom)**0.25


def JeansParam(Mplanet, Rplanet, Lbol, Dist):
	Teq = EquilibriumTemperature(Lbol, Dist)
	Mplanet *= const.M_earth.value
	Rplanet *= const.R_earth.value
	m_H = const.m_p.value
	jeans = const.G.value * Mplanet * m_H / (const.k_B.value * Teq * Rplanet )
	return jeans


# TODO: Improve
def CorePowered(**kwargs):
	
	gparams = dict(
		gamma = 7/5, # adiabatic index
		mu = 2 * const.m_p.value , # kg, atmosphere molecular mass

	)

	def ModifiedBondiRadius(Mcore, Teq):
		"""
		Calculates modified Bondi radius
		Parameters:
			- Mcore: core mass (Earth masses)
			- Teq: equilibrium temperature (Kelvin)
		Returns:
			- Rbondi: Bondi radius (Earth radii)
		"""
		nonlocal gparams
		gamma = gparams['gamma']
		mu = gparams['mu']
		Mcore *= const.M_earth.value
		Rbondi = (gamma-1)/gamma * const.G.value * Mcore * mu / (const.k_B.value * Teq)
		return Rbondi / const.R_earth.value


	def DensityRcb(Menv, Renv, Mcore, Rcore, Teq):
		"""
		Calculates the density at the radiative-convective boundary of the atmosphere
		Parameters:
			...
		Returns:
			- RhoRcb: Rcb density (kg/m^3)
		"""
		nonlocal gparams
		gamma = gparams['gamma']
		mu = gparams['mu']
		Rbondi = ModifiedBondiRadius(Mcore, Teq)
		# Unit conversions
		Rbondi *= const.R_earth.value
		Renv *= const.R_earth.value
		Rcore *= const.R_earth.value
		Menv *= const.M_earth.value
		Mcore *= const.M_earth.value
		# Calculation
		RhoRcb = gamma / (gamma-1)
		RhoRcb *= Menv / (4 * np.pi * Rcore**2 * Renv)
		RhoRcb *= (Rbondi * Renv / Rcore**2)**(-1/(gamma-1))
		return RhoRcb


	def SoundSpeed(Teq):
		nonlocal gparams
		mu = gparams['mu']
		return np.sqrt( const.k_B.value * Teq / mu )


	def SonicPointRadius(Mplanet, Teq):
		Mplanet *= const.M_earth.value
		cs = SoundSpeed(Teq)
		Rs = const.G.value * Mplanet / (2 * cs**2)
		return Rs / const.R_earth.value


	def AtmosphericOpacity(RhoRcb, Zstar = 1.0):
		beta = 0.6
		cm = units.cm
		g = units.g
		opacity = 0.1 * Zstar * (RhoRcb)**beta # cm^2 / g
		opacity *= 0.01 # to m^2 / g
		return opacity

	def PlanetLuminosity(Teq, Rbondi, RhoRcb):
		Rbondi *= const.R_earth.value
		opacity = AtmosphericOpacity(RhoRcb)
		return 64*np.pi/3 * const.sigma_sb.value * Teq**4 * Rbondi / opacity / RhoRcb

	def PlanetEnergyLimitedMloss(mass, radius, Lplanet, Dist, Mstar):
		EL = EnergyLimited
		Lplanet = 1e7 * Lplanet # to erg/s
		mloss = EL(mass=mass, radius=radius, Lxuv=Lplanet, dist=Dist, mstar=Mstar)
		return mloss * const.M_earth.value / units.Myr.to('s')

	def CorePoweredMloss(mcore, rcore, menv, renv, Lbol, dist, verbose=False, **kwargs):
		
		Teq = EquilibriumTemperature(Lbol, dist)
		Rs = SonicPointRadius(mcore + menv, Teq)
		cs = SoundSpeed(Teq)
		RhoRcb = DensityRcb(menv, renv, mcore, rcore, Teq)

		# unit conversions
		menv *= const.M_earth.value
		mcore *= const.M_earth.value
		renv *= const.R_earth.value
		rcore *= const.R_earth.value
		Rs *= const.R_earth.value

		# calculation
		Rrcb = renv + rcore
		expn = (-1) * const.G.value * (menv + mcore) / (cs**2 * Rrcb)
		mloss = 4 * np.pi * Rs**2 * cs * RhoRcb * np.exp(expn)
		mloss *= 1e3 # to g/s 	

		if 'mstar' in kwargs:
			Lplanet = 1e7 * PlanetLuminosity(Teq, ModifiedBondiRadius(mcore,Teq), RhoRcb)
			mloss_e = 1e3 * PlanetEnergyLimitedMloss(mass=mcore+menv, radius=rcore+renv,
				Lplanet=Lplanet, Dist=dist, Mstar=kwargs['mstar'])
			#print(f"{Lplanet=:.3g} erg/s")
			if verbose: print(f"{mloss=:.3g}, {mloss_e=:.3g} g/s")
			return min(mloss, mloss_e)
		
		return mloss

	mp, rp, Lbol, dist = kwargs['mp'], kwargs['rp'], kwargs['Lbol'], kwargs['dist']
	if JeansParam(mp, rp, Lbol, dist) > 25: return 0.0  # negligible mass loss
	mloss = CorePoweredMloss(
		mcore=kwargs['mcore'], rcore=kwargs['rcore'],
		menv=kwargs['menv'],   renv=kwargs['renv'],
		Lbol=Lbol, dist=dist,
		mstar = kwargs['mstar'] if 'mstar' in kwargs else 0.74, # force star mass assuming K2-136
		verbose = False
	)
	mloss *= units.Myr.to('s') / const.M_earth.to('g').value # g/s to Me/Myr
	return mloss




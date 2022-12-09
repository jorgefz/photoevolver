from typing import Union
import sys
import os
import numpy as np
import astropy.units as units
import astropy.constants as const
from scipy.interpolate import griddata, LinearNDInterpolator

from .K18grid import interpolator as interpolator


_stdout = sys.stdout
def _DisablePrint():
	sys.stdout = open(os.devnull, 'w')
def _EnablePrint():
	sys.stdout = _stdout


"""
mss = input0[:,0]; #stellar mass [Msun]
teq = input0[:,1]; #equilibrium temperature [K]
sma = input0[:,2]; #orbital separation [AU]
euv = input0[:,3]; #EUV flux at the planetary orbit [erg/s/cm^2]
rpl = input0[:,4]; #planetary radius [Rearth]
mpl = input0[:,5]; #planetary mass [Mearth]
bt = input0[:,7]; # reduced Jeans escape parameter Lambda
lhy = input0[:,6]; #hydrodynamical mass loss [g/s]
mindex = input0[:,-1]; #technical index for extrapolation 0/1/2 []
"""


def LoadGrid():
	fields = ['Mstar', 'Teq', 'Dist', 'Feuv', 'Rp', 'Mp', 'Mloss', 'Jeans', 'Mass-index']
	txtdata = np.loadtxt('K18grid/input_interpol_26-03-2021.dat');
	data = {k: txtdata[:,i] for i,k in enumerate(fields)}
	return data


def EquilibriumTemperature(Lbol, a):
	""" Lbol in erg/s, a in AU. Returns temperature in Kelvin """
	numerator = Lbol * units.erg.to('J')
	denominator = 16 * const.sigma_sb.value * np.pi * (a * units.au.to('m'))**2
	return (numerator/denominator)**0.25


def LumToFlux(lum : Union[float,units.Quantity],
			  dist : Union[float,units.Quantity]):
	"""
	Converts luminosity to flux at given distance
	Default units if none provided:
		lum -> erg/s
		dist -> AU
	Returns:
		flux (units.Quantity)
	"""
	if not isinstance(lum, units.Quantity):
		lum *= units.erg / units.s
	if not isinstance(dist, units.Quantity):
		dist *= units.au
	return lum / (4.0 * np.pi * (dist)**2)


###########################
## Inteprolation Methods ##
###########################

"""
Interpolate with Daria Kubyshkina's script.

Parameters
----------
	mp		Planet mass (Mearth)
	rp		Planet radius (Rearth)
	Lbol	Star's bolometric luminosity (erg/s)
	dist	Orbital separation (AU)
	Leuv	Star's EUV luminosity (erg/s)
	mstar	Star's mass (Msun)

Returns
-------
	mass loss rate (M_e/Myr)
"""
def K18Interpolator(*args, **kwargs):
	nargs = 'mp','rp','Lbol','dist','Leuv', 'mstar'
	Mp, Rp, Lbol, Dist, Leuv, Mstar = [kwargs[n] for n in nargs]
	Teq = EquilibriumTemperature(Lbol, Dist)
	Feuv = LumToFlux(Leuv, Dist).to("erg * cm^-2 * s^-1").value

	_DisablePrint()
	mloss = interpolator.INTERPOL(Mstar, Feuv, Teq, Rp, Mp)
	_EnablePrint()
	
	gs_to_MeMyr = (units.g / units.s).to("M_earth/Myr")
	# print(f"{Leuv=}, {Feuv=}, {mloss=:.3e}")
	mloss = float(mloss) * gs_to_MeMyr
	return mloss 








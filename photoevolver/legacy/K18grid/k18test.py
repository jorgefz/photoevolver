import numpy as np
import pandas as pd
from astropy import constants, units as u
from scipy.interpolate import LinearNDInterpolator, griddata

def _prep_data():

	df = pd.read_csv(
		'input_interpol_26-03-2021.dat',
		names=['Mstar','Teq','Sep','Feuv','Rp','Mp','Jeans','Mloss','Regime'],
		delimiter='   ', header=None
	)
	s = df.sort_values(['Mstar','Teq','Sep','Feuv','Rp','Mp'])
	# points = s[['Mstar','Teq','Sep','Feuv','Rp','Mp']].to_numpy()

	print("Calculating Leuv...")
	s['Leuv'] = np.log10(s.apply(lambda r: r['Feuv']*4*np.pi*(r['Sep']*u.au.to('cm'))**2, axis=1))

	print("Calculating Fbol...")
	s['Fbol'] = s.apply(lambda r: ((r['Teq']*u.K)**4 * 4 * constants.sigma_sb).to('erg/s/cm^2').value, axis=1)
	s['Lbol'] = np.log10(s.apply(lambda r: r['Fbol']*4*np.pi*(r['Sep']*u.au.to('cm'))**2, axis=1))

	print("lbol=", s['Lbol'].min(), s['Lbol'].max())
	print("leuv=", s['Leuv'].min(), s['Leuv'].max())
	print("sep= ", s['Sep'].min(),  s['Sep'].max())
	print("mp=  ", s['Mp'].min(),   s['Mp'].max())
	print("rp=  ", s['Rp'].min(),   s['Rp'].max())

	return s

# _data = _prep_data()
# _interp_cache = {}

def _k18_mloss(mstar, fbol, sep, leuv, rp, mp):

	nearest_mstar = _data['Mstar'].unique()[(np.abs(_data['Mstar'].unique() - 1.1)).argmin()]
	
	if nearest_mstar in _interp_cache:
		return _interp_cache[nearest_mstar](fbol, sep, leuv, rp, mp)
	
	idx = (_data['Regime']==0.0) & (_data['Mstar'] == nearest_mstar)

	points = _data.loc[idx,['Fbol','Sep','Leuv','Rp','Mp']].to_numpy()
	values = _data.loc[idx,'Mloss'].to_numpy()

	interp = LinearNDInterpolator(points, values)
	_interp_cache[nearest_mstar] = interp
	return interp(fbol, sep, np.log10(leuv), rp, mp)




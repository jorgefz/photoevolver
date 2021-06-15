"""
[photoevolver.massloss]

Includes several atmosoheric mass loss formulations
"""

import numpy as np
import astropy.constants as Const
import astropy.units as Units


"""
class OwenWu17:
     
    bounds = {
            "mass": [1.0,   20.0],
            "fenv": [1e-4,  0.2],
            "fbol": [4.0,   400.0],
            "age":  [100.0, 10000.0]
    }

    req_fields = ['mass', 'fenv', 'age', 'fbol', 'mcore']
    def check_params(params):
        for f in LopezFortney14.req_fields:
            if f not in list(params.keys()):
                raise KeyError(f"Model 'OwenWu17' requires parameter '{f}'")

    def check_bounds(params):
        bounds = OwenWu17.bounds
        for f in list(bounds.keys()):
            if not (bounds[f][0] <= params[f] <= bounds[f][1]):
                raise ValueError(f" Input {pstr} out of safe bounds ({bmin},{bmax})")

    def env_radius(params = dict(), safe = True):
        OwenWu17.check_params(params)
        if safe is True: OwenWu17.check_bounds(params)
        # Code here
        return
"""


class EnergyLimited:

    bounds = None
    req_fields = ['rp', 'mp', 'Lxuv', 'dist', 'mstar', 'eff', 'beta']
    
    def check_bounds(params):
        pass

    def check_params(params):
        pass

    def salz16_beta(Fxuv, mp, rp):
        """
        Parameters
            Fxuv: erg/cm2/s
            mp: Earth masses
            rp: Earth radii
        """
        potential = Const.G.to('erg*m/g^2').value * mp * Const.M_earth.to('g').value / (rp * Const.R_earth.to('m').value)
        log_beta = -0.185*np.log10(potential) + 0.021*np.log10(Fxuv) + 2.42
        if log_beta < 0.0: log_beta = 0.0
        return 10**(log_beta)

    def salz16_eff(mp, rp):
        """
        Parameters
            mp: Planet mass in Earth masses
            rp: Planet radius in Earth radii
        """
        potential = Const.G.to('erg*m/g^2').value * mp * Const.M_earth.to('g').value / (rp * Const.R_earth.to('m').value)
        v = np.log10(potential)
        if   ( v < 12.0):           log_eff = np.log10(0.23) # constant
        if   (12.0  < v <= 13.11):  log_eff = -0.44*(v-12.0) - 0.5
        elif (13.11 < v <= 13.6):   log_eff = -7.29*(v-13.11) - 0.98
        elif (v > 13.6):            log_eff = -7.29*(13.6-13.11) - 0.98 # stable atmospheres, no evaporation (< 1e-5)
        return 10**(log_eff)*5/4 # Correction evaporation efficiency to heating efficiency

    def __init__(self, params = dict(), safe = True):
        """
        Applies the energy-limited mass loss equation to calculate
        the photoevaporative mass loss rate.
        Parameters:
            Lxuv:   (float) XUV luminosity from the star (erg/s)
            rp:     (float) Planet radius (Earth radii)
            mp:     (float) Planet mass (Earth masses)
            dist:   (float) Planet's orbital distance (AU)
            mstar:  (float) Mass of the host star (Msun)
            eff:    (float, default = 0.1) Efficiency
            beta:   (float, default = 1.0) Rxuv/Rp
        Returns
            mloss:  (float) Mass loss rate (Earth masses / Myr)
        """
	eff = params['eff'] if params and ('eff' in params) else eff = None
	beta = params['beta'] if params and ('beta' in params) else beta = None

        # Unit conversions
        Lxuv *= 1e-7 # erg/s to Watts
        mstar *= Const.M_sun.value # M_sun to kg
        mp *= Const.M_earth.value # M_earth to kg
        rp *= Const.R_earth.value # R_earth to m
        dist *= Const.au.value # AU to m

        Fxuv = Lxuv / ( 4 * np.pi * (dist)**2 )

        # Variable efficiency and Rxuv
        if eff  is None: eff  = salz16_eff(mp/Const.M_earth.value, rp/Const.R_earth.value)
        if beta is None: beta = salz16_beta(Fxuv * 1000, mp/Const.M_earth.value, rp/Const.R_earth.value) # Fxuv W/m2 to erg/cm2/s

        # Energy-limited equation
        xi = (dist / rp) * (mp / mstar / 3)**(1/3)
        K_tide = 1 - 3/(2*xi) + 1/(2*(xi)**3) 
        mloss = beta**2 * eff * np.pi * Fxuv * (rp)**3 / (Const.G.value * K_tide * mp)
        return mloss * 5.28e-12 # Earth masses per Myr


class Kubyshkina18:
    """
    Hydro-based approximation to mass loss.
    Source: Kubyshkina et al (2018)
    """
    large_delta = {
        'beta':  16.4084
        'alpha': [1.0, -3.2861, 2.75]
        'zeta':  -1.2978
        'theta': 0.8846
    }

    small_delta = {
        'beta': 32.0199
        'alpha': [0.4222, -1.7489, 3.7679]
        'zeta': -6.8618
        'theta': 0.0095
    }

    bounds = None
    req_fields = ['rp', 'mp', 'Lxuv', 'dist', 'Lbol']
 
    def check_bounds(params):
        pass

    def check_params(params):
        pass
    
    def Epsilon(rp, mp, Lxuv, dist):
        numerator = 15.611 - 0.578*np.ln(Fxuv) + 1.537*np.ln(dist) + 1.018*np.ln(rp)
        denominator = 5.564 + 0.894*np.ln(dist)
        return numerator / denominator

    def __init__(self, params = dict(), safe = True):
        """
        Lxuv: erg/s
        dist: AU
        mp: Earth masses
        rp: Earth masses
        """
	conv = (Units.erg / Units.cm**2 / Units.s).to('W/m^2') # erg/cm^2/s to W/m^2
	Fxuv = Lxuv / (4 * np.pi * (dist*Const.au.to('cm').value)**2 )
	Fbol = Lbol / (4 * np.pi * (dist*Const.au.to('cm').value)**2 )
	Teq =  ( Fbol * conv / (4*Const.sigma_sb.value) )**(1/4)
	mH = 1e3 # kg / mol 

        Delta = Const.G.value * (mp*Const.M_earth.value) * (mH) / (Const.k_B.value * Teq * (rp*Const.R_earth.value) )
        Epsilon = self.Epsilon(Fxuv, dist, rp) 

        xp = self.small_delta if Delta < np.exp(Epsilon) else self.large_delta
        
        Kappa = xp['zeta'] + xp['theta']*np.ln(dist)

        mloss = np.exp(xp['beta']) * (Fxuv)**xp['alpha'][0] * (dist)**xp['alpha'][1] * (rp)**xp[alpha][2] * (Delta)**Kappa
        return mloss # to Earth masses per Myr









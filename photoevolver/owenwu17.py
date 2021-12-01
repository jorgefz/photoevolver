
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as Const
import astropy.units as Units

from scipy.integrate import quad as QuadIntegrate
from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import fsolve as ScipyFsolve
from scipy.optimize import brentq as BrentRootFind


### Constants in cgs ##
k_B = Const.k_B.cgs.value # boltzmann's constant
mH = Units.M_p.to('g') + Units.M_e.to('g') # mass of Hydrogen atom in grams
sigma_sb = Const.sigma_sb.cgs.value # stefan-Boltzmann constant
G = Const.G.cgs.value # Gravitational constant

# opacity from form kappa = kappa0*P^alpha*T^beta
alpha   = 0.68 # pressure dependence of opacity
beta    = 0.45 # temperature dependence of opacity
kappa0  = 10**(-7.32) # opacity constant
mu = 2.35 * mH # solar metallicity gas
gamma = 5/3 # ratio of specific rho_earth_cgs
grad_ab = (gamma-1)/gamma



def Integrand1(x, gamma):
    return x * (1/x-1)**(1/(gamma-1))

def Integrand2(x, gamma):
    return x**2 * (1/x-1)**(1/(gamma-1))


def get_I2(DR_Rc, gamma):
    """ Calculates integral I2 """
    I2 = [0] * len(DR_Rc)
    for i in range(len(DR_Rc)):
        low_lim = 1 / (DR_Rc[i] + 1)
        up_lim = 1.0
        I2[i], _ = QuadIntegrate(func = Integrand2, a = low_lim, b = up_lim, args = gamma)
    return np.array(I2)


def get_I2_I1(DR_Rc, gamma):
    """ Calculates integral ratio I2 / I1 """
    ratio = [0] * len(DR_Rc)
    for i in range(len(DR_Rc)):
        low_lim = 1 / (DR_Rc[i] + 1)
        up_lim = 1.0
        I2, _ = QuadIntegrate(func = Integrand2, a = low_lim, b = up_lim, args = gamma)
        I1, _ = QuadIntegrate(func = Integrand1, a = low_lim, b = up_lim, args = gamma)
        ratio[i] = I2/I1
    return np.array(ratio)


def get_RhoRcb(log_D_Rrcb, xenv, mcore, rcore, Teq, Tkh, Xiron, Xice):
    """
    Evaluates the density at the radiative convective boundary.
    Equation 13 from Owen & Wu (2017)
    """
    Rrcb = 10**log_D_Rrcb + rcore
    Delta_R_Rc = 10**log_D_Rrcb / rcore
    I2_I1 = get_I2_I1(np.array([Delta_R_Rc]), gamma)

    TKh_sec = Tkh * Units.Myr.to('s')

    numerator = 64 * np.pi * I2_I1 * sigma_sb * Teq**(3-alpha-beta) * Rrcb * TKh_sec
    denominator = 3 * kappa0 * mcore * Units.M_earth.to('g') * xenv
    RhoRcb = (mu / k_B) * (numerator / denominator) ** (1 / (1+alpha))

    return RhoRcb



def XenvFunction(log_D_Rrcb, args : dict):
    """ Calculate Xenv and return difference with guessed value """
    # we combine equation 4 and 13 from Owen & Wu (2017) to produce a function to solve
    # first evaluate the density at the radiative convective boundary

    rcore = args['rcore']
    mcore = args['mcore']
    xenv = args['xenv']

    Rrcb = 10**log_D_Rrcb + rcore
 
    Delta_R_Rc = 10**log_D_Rrcb / rcore

    rho_core = mcore * Const.M_earth.to('g').value / (4/3 * np.pi * rcore**3)

    rho_rcb = get_RhoRcb(log_D_Rrcb, **args)

    I2 = get_I2( [Delta_R_Rc], gamma)

    cs2 = k_B * args['Teq'] / mu

    grad_ab = (gamma-1)/gamma

    Xguess=3.*(Rrcb/rcore)**3.*(rho_rcb/rho_core)*(grad_ab * (G * mcore * Units.M_earth.to('g'))/(cs2 * Rrcb))**(1./(gamma-1.))*I2

    return Xguess - xenv


def EquilibriumTemperature(fbol):
    """ Calculates Teq from Fbol in erg/cm^2/s """
    fbol *= (Units.erg / Units.cm**2 / Units.s).to("W m^-2")
    Teq = (fbol / 4 / Const.sigma_sb.value)**(1/4)
    return Teq


def RadiusOwenWu17(fenv, mass, rcore, age, fbol, dist, Xiron=1/3, Xice=0.0, **kwargs):

    mcore = mass * ( 1 - fenv)
    xenv = fenv / (1 - fenv)
    Teq = EquilibriumTemperature( fbol )
    Tkh = age if age > 100 else 100
    rcore *= Units.R_earth.to('cm')

    # Solving structure equation for radius
    # Rough guess with analytical formula

    Delta_R_guess = 2 * rcore * (xenv/0.027)**(1/1.31) * (mcore/5)**(-0.17)   

    # use log Delta_Rrcb_guess to prevent negative solutions
    args = dict(xenv=xenv, Teq=Teq, mcore=mcore, rcore=rcore, Tkh=Tkh, Xiron=Xiron, Xice=Xice)
    solution = ScipyFsolve(func = XenvFunction, x0 = np.log10(Delta_R_guess), args = args)
    log_D_Rrcb_sol = solution[0]

    Rrcb_sol = 10**log_D_Rrcb_sol + rcore

    # now find f-factor to compute planet radius
    rho_rcb = get_RhoRcb(log_D_Rrcb_sol, xenv, mcore, rcore, Teq, Tkh, Xiron, Xice)

    # now calculate the densities at the photosphere
    Pressure_phot = (2/3 * G * mcore * Const.M_earth.to('g').value / (Rrcb_sol**2 * kappa0 * Teq**beta) ) ** (1/(1+alpha))
    rho_phot_calc = (mu/k_B) * Pressure_phot / Teq

    # now find f factor
    H = k_B * Teq * Rrcb_sol ** 2 / ( mu * G * mcore * Const.M_earth.to('g').value)
    f = 1 + (H/Rrcb_sol) * np.log(rho_rcb/rho_phot_calc)
    Rplanet = f * Rrcb_sol

    #ret = Rrcb_sol, f, Rplanet, Rrcb_sol - rcore
    renv = Rplanet - rcore
    return float(renv) / Units.R_earth.to('cm')


###### functions for solid-cores from Fortney et al. 2007

def radius_to_mass(radius, Xiron, Xice):
    # convert a radius into mass using mass-radius relationship
    # use Fortney et al. 2007, mass-radius relationship

    if (Xiron > 0 ):
        # use iron function
        if (Xice > 0. ):
            print ("error, cannot use Iron and Ice fraction together")
            return -1
        else:
            mass = fsolve(iron_function,np.log10(5.),args=[Xiron,radius])
    else:
        mass = fsolve(ice_function,np.log10(5.),args=[Xice,radius])

    return 10**mass


def mass_to_radius(mass,Xiron,Xice):

    if (Xiron > 0. ):
        # use iron function
        if (Xice > 0. ):
            print ("error, cannot use Iron and Ice fraction together")

            return -1.

        else:
            rad = iron_function(np.log10(mass),[Xiron,0.])
    else:
        rad = ice_function(np.log10(mass),[Xice,0.])

    return rad


def ice_function(lg_mass,inputs):

    X=inputs[0]
    radius = inputs[1]

    R = (0.0912*X + 0.1603) * (lg_mass)**2.
    R += (0.3330*X + 0.7387) * (lg_mass)
    R += (0.4639*X + 1.1193)

    return R-radius

def iron_function(lg_mass,inputs):

    Xiron=inputs[0]
    radius = inputs[1]

    X = 1. - Xiron # rock mass fraction

    R = (0.0592*X + 0.0975) * (lg_mass)**2.
    R += (0.2337*X + 0.4938) * (lg_mass)
    R += (0.3102*X + 0.7932)

    return R-radius




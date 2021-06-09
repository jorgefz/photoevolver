import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.constants as Const
from scipy.interpolate import interp1d, UnivariateSpline
import Mors as mors


# Constants
Lbol_sun = Const.iau2015.L_sun.value
Fbol_earth = Const.iau2015.L_sun.to('erg/s').value / (4 * np.pi * (Const.au.to('cm').value)**2)
R_earth = Const.R_earth.value
M_earth = Const.M_earth.value
kb = Const.k_B.value # Boltzmann constant
sb = Const.sigma_sb.value # Stefan-Boltzmann constant
G = Const.G.value # Gravitational constant
au_cm = Const.au.to('cm').value

# Units
Units = {
    'Mass':   "M_earth",
    'Radius': "R_earth",
    'Age':    "Myr",
    'Lbol':   "erg/s",
    'Fbol':   "erg/cm^2/s",
    'Lxuv':   "erg/s",
    'Fxuv':   "erg/cm^2/s",
}


def radiative_radius(mp , rp, Lbol, au):
    """
    Estimates radius of the radiative atmosphere.
    """
    # Parameters
    Fbol = Lbol * Const.L_sun.value / (4 * np.pi * (au * Const.au.value)**2 )
    Teq = (Fbol / (4*sb) )**(1/4)
    surf_grav = G * (mp * M_earth) / (rp * R_earth)**2
    #molec_weight = 0.75 + 0.25/4 # H:75%, He:25%
    molec_weight = 3.5e3 # kg/mol

    # Equation
    ratm = 9 * (kb * Teq / surf_grav / molec_weight) / R_earth
    
    return ratm



def envelope_radius(mp, rp, Lbol, au, age, fenv, verbose=True):
    """
    Estimates the envelope radius of a planet.
    Parameters:
        mp:    (float) Planet mass (Earth masses)
        rp:    (float) Planet radius (Earth radii)
        Lbol:  (float) Star's bolometric luminosity (Lsun)
        au:    (float) Orbital semi-major axis of the planet (AU)
        age:   (float) Age of the system (Myr)
        
        fenv:  (float, default=0.01) Envelope mass fraction
        verbose: (bool, default=True) Verbosity

    Returns:
        R_env  (float) The envelope radius (Earth radii)
        R_core (float) The core radius (Earth radii)
    """

    # Constants
    Fbol_earth = Const.L_sun.value / (4 * np.pi * (Const.au.value)**2)
    R_earth = Const.R_earth.value
    M_earth = Const.M_earth.value
    
    # Parameters
    Fbol = Lbol * Const.L_sun.value / (4 * np.pi * (au * Const.au.value)**2 )

    mass_term = 2.06 * (mp)**(-0.21)
    flux_term = (Fbol/Fbol_earth)**(0.044)
    age_term  = (age/5000)**(-0.18)

    fenv_term = (fenv/0.05)**(0.59)
    renv = mass_term * fenv_term * flux_term * age_term

    rcore = rp - renv - radiative_radius(mp=mp, rp=rp, Lbol=Lbol, au=au)
    if verbose:
        print(f" Envelope radius estimate from simple power-law fit to their models:")
        print(f" Renv = {renv:.2g} Earth radii ({100*renv/rp:.0f}% of its total radius)")
    return renv, rcore
    

def mass_fraction(mp, rp, Fbol, age, rcore, verbose=True):
    """
    Parameters:
        mp:     (float) Planet's mass (Earth masses)
        rp:     (float) Planet's radius (Earth radii)
        Fbol:   (float) Bolometric flux at the planet (erg/cm2/s)
        age:    (float) Age of the system (Myr)
        rcore:  (float) Planet's core radius
    """
    mass_term = 2.06 * (mp)**(-0.21)
    flux_term = (Fbol/Fbol_earth)**(0.044)
    age_term  = (age/5000)**(-0.18)

    #ratm = radiative_radius(mp, rp, Lbol, au)
    renv = rp - rcore

    if verbose:
        print("")
        print(f" Envelope radius estimate:")
        print(f" - Radius = {rp:.2f}")
        print(f" - Rcore  = {rcore:.2f}")
        print(f" - Renv   = {renv:.2f}")
        print("")

    fenv = (renv / (mass_term * flux_term * age_term) )**(1/0.59) * 0.05

    planet_dens = mp * M_earth * 1e3 / ( 4/3 * np.pi * (rp * R_earth * 1e2)**3)
    core_dens = mp * (1-fenv) * M_earth * 1e3 / ( 4/3 * np.pi * (rcore * R_earth * 1e2)**3)
    envelope_shell_volume = 4/3 * np.pi * ((rcore+renv)**3 - renv**3) * (R_earth*1e2)**3 # g/cm3
    envelope_dens = mp * fenv * M_earth * 1e3 / envelope_shell_volume

    if verbose:
        print(f" Estimated envelope mass fraction: {fenv*100:.2g}% ({mp*fenv:.2g} Earth masses)")
        print(f" Planet density:   {planet_dens:.3f} g/cm3 ")
        print(f" Core density:     {core_dens:.3f} g/cm3 ")
        print(f" Envelope density: {envelope_dens:.3g} g/cm3 ") 

    return fenv


def planet_mass(mcore, renv, Lbol, dist, age, verbose=True):
    """
    Calculate a planet's mass using its envelope radius and envelope mass fraction
    """

    Fbol = Lbol * Const.L_sun.to('erg/s').value / (4 * np.pi * (dist * Const.au.to('cm').value)**2 )
    flux_term = (Fbol/Fbol_earth)**(0.044)
    age_term  = (age/5000)**(-0.18)
    C = flux_term * age_term

    def f(mp, *args):
        mcore = args[0]
        renv = args[1]
        C = args[2]
        if mp <= mcore: return 0
        #y = 2.06*(mp)**(-0.21) * ((mp-mcore)/mp/0.05)**(0.59) * C - renv
        y = 2.06*C/(0.05**0.59) * mp**(-0.8) * (mp-mcore)**0.59 - renv
        return y

    if verbose:
        print( "Solving equation with arguments:")
        print(f" - Mcore: {mcore}")
        print(f" - Renv:  {renv}")
        print(f" - Const: {C}")

    from scipy.optimize import fsolve, least_squares
    
    # Solve Lopez & Fortney '14 equation for the planet mass
    # First estimate is core mass plus envelope mass with a density of 0.028 g/cm3
    estimate = mcore + (4/3) * np.pi * (renv)**3 * (0.028*0.04345)
    if verbose: print(f" - Estim: {estimate}")

    res = least_squares( f, estimate, bounds = (mcore, 100.0), args=(mcore,renv,C))
    if verbose:
        print(res)
        if res.success: print(f" === Solution found! Mp = {res.x} ===")
        else:
            print(" Error! Unable to converge.")
            return 0
    return res.x[0]
    
def salz16_beta(Fxuv, mp, rp):
    """
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
    mp: Earth radii
    rp: Earth radii
    """
    potential = Const.G.to('erg*m/g^2').value * mp * Const.M_earth.to('g').value / (rp * Const.R_earth.to('m').value)
    v = np.log10(potential)
    if   ( v < 12.0):           log_eff = np.log10(0.23) # constant
    if   (12.0  < v <= 13.11):  log_eff = -0.44*(v-12.0) - 0.5
    elif (13.11 < v <= 13.6):   log_eff = -7.29*(v-13.11) - 0.98
    elif (v > 13.6):            log_eff = -7.29*(13.6-13.11) - 0.98 # stable atmospheres, no evaporation (< 1e-5)
    return 10**(log_eff)*5/4 # Correction evaporation efficiency to heating efficiency

def mass_loss(Lxuv, rp, mp, dist, mstar, eff=None, beta=None):
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


class Planet:
    def __init__(self, dist, age, mstar=None, radius = None, mass = None,
                 core_radius = None, core_mass = None,
                 env_radius = None, env_mass = None, env_mass_fraction = None,
                 eff = None, beta = None, Star = None, timestep = 1.0): 
        """
        Planet class defines the structure of a planet and evolves it in time.
        On initialization, define the initial parameters:

        Parameters:
            dist:   (float) Distance between the planet and the star (AU).
            age:    (float) Initial age of the planet/star (Myr).
            
            radius:
            core_radius:
            env_radius:

            mass:
            core_mass:
            env_mass:
            env_mass_fraction:

            eff:
            beta:
            Star:
            timestep: (float, default = 1 Myr)
        """

        # Initial parameters
        self.rp    = radius
        self.mp    = mass
        self.rcore = core_radius
        self.mcore = core_mass
        self.renv  = env_radius 
        self.menv  = env_mass
        self.fenv  = env_mass_fraction
        self.ratm  = 0.0
        self.mloss = 0.0

        self.age  = age
        self.dist = dist
        self.mstar = mstar
        self.fbol = 0.0
        self.fxuv = 0.0
        self.eff  = eff
        self.beta = beta

        # Tracks
        self.Star = Star
        self.init_index = -1  if Star is None else np.transpose(np.nonzero(age < Star.AgeTrack))[0][0]
        self.AgeTrack  = None if Star is None else Star.AgeTrack
        self.LbolTrack = None if Star is None else Star.LbolTrack
        self.LxTrack   = None if Star is None else Star.LxTrack
        self.LeuvTrack = None if Star is None else Star.LeuvTrack
        self.LxuvTrack = None if Star is None else np.array([Star.LxTrack[i] + Star.LeuvTrack[i] for i in range(len(Star.LxTrack))])

        self.RadiusTrack          = None if Star is None else np.zeros_like(Star.AgeTrack)
        self.MassTrack            = None if Star is None else np.zeros_like(Star.AgeTrack)
        self.CoreRadiusTrack      = None if Star is None else np.zeros_like(Star.AgeTrack)
        self.CoreMassTrack        = None if Star is None else np.zeros_like(Star.AgeTrack)
        self.EnvRadiusTrack       = None if Star is None else np.zeros_like(Star.AgeTrack)
        self.EnvMassTrack         = None if Star is None else np.zeros_like(Star.AgeTrack)
        self.EnvMassFractionTrack = None if Star is None else np.zeros_like(Star.AgeTrack)
        self.DensityTrack         = None if Star is None else np.zeros_like(Star.AgeTrack)
        self.MassLossTrack        = None if Star is None else np.zeros_like(Star.AgeTrack)

        # Interpolation functions
        self.LbolInterp       = None if Star is None else UnivariateSpline(Star.AgeTrack, Star.LbolTrack, k=4, s=0) 
        self.LxInterp         = None if Star is None else UnivariateSpline(Star.AgeTrack, Star.LxTrack, k=4, s=0)
        self.LeuvInterp       = None if Star is None else UnivariateSpline(Star.AgeTrack, Star.LeuvTrack, k=4, s=0)
        """
        self.RadiusInterp     = None if Star is None else UnivariateSpline(Star.AgeTrack, Star.MassTrack, k=4, s=0)
        self.MassInterp       = None if Star is None else UnivariateSpline(Star.AgeTrack, Star.RadiusTrack, k=4, s=0)
        self.CoreRadiusInterp = None if Star is None else UnivariateSpline(Star.AgeTrack, Star.CoreRadiusTrack, k=4, s=0)
        self.CoreMassInterp   = None if Star is None else UnivariateSpline(Star.AgeTrack, Star.CoreMassTrack, k=4, s=0)
        self.EnvRadiusInterp  = None if Star is None else UnivariateSpline(Star.AgeTrack, Star.EnvRadiusTrack, k=4, s=0)
        self.EnvMassInterp    = None if Star is None else UnivariateSpline(Star.AgeTrack, Star.EnvMassTrack, k=4, s=0)
        self.EnvMassFInterp   = None if Star is None else UnivariateSpline(Star.AgeTrack, Star.EnvMassFunctionTrack, k=4, s=0)
        self.DensityInterp    = None if Star is None else UnivariateSpline(Star.AgeTrack, Star.DensityTrack, k=4, s=0)
        self.MassLossInterp   = None if Star is None else UnivariateSpline(Star.AgeTrack, Star.MassLossTrack, k=4, s=0)
        """

        # Derivatives of interps
        self.LbolDeriv = None if Star is None else self.LbolInterp.derivative()

        #if Star: self.generateTracks(self.AgeTrack, self.LxTrack, self.LeuvTrack, self.LxuvTrack)


    def __repr__(self):
        msg =  f"<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}>\n\n"\
            + f"".ljust(10) + f"Radius".ljust(10) + f"Mass".ljust(10) + "\n"\
            + f"Core".ljust(10) + f"{self.rcore:.2f}".ljust(10) + f"{self.mcore:.2f}".ljust(10) + "\n"\
            + f"Envelope".ljust(10) + f"{self.renv:.2f}".ljust(10) + f"{self.menv:.2f}".ljust(10) + "\n"\
            + f"Total".ljust(10) + f"{self.rp:.2f}".ljust(10) + f"{self.mp:.2f}".ljust(10) + "\n"
        return msg

    def __str__(self):
        return self.__repr__()
    
    # Getters
    def mass(self):
        return self.mp

    def radius(self):
        return self.rp

    # Interpolations
    def Lbol(self, age):
        return self.LbolInterp(age)
    def gradLbol(self, age):
        return self.LbolDeriv(age)
    def Lx(self, age):
        return self.LxInterp(age)
    def Leuv(self, age):
        return self.LeuvInterp(age)
    def Lxuv(self, age):
        return self.Lx(age) + self.Leuv(age)
    """
    def Radius(age):
        return self.RadiusInterp(age)
    def Mass(age):
        return self.MassInterp(age)
    def CoreRadius(age):
        return self.CoreRadiusInterp(age)
    def CoreMass(age):
        return self.CoreMassInterp(age)
    def EnvRadius(age):
        return self.EnvRadiusInterp(age)
    def EnvMass(age):
        return self.EnvMassInterp(age)
    def EnvMassFraction(age):
        return self.EnvMassFInterp(age)
    def Density(age):
        return self.DensityInterp(age)
    def MassLoss(age):
        return self.MassLossInterp(age)
    """
    # Equations
    def density(self):
        return self.mass() * Const.M_earth.value / (4/3 * np.pi * (self.radius() * Const.R_earth.value)**3)
    def temperature(self):
        return (self.fbol*1e-3 / (4*Const.sigma_sb.value) )**(1/4)
    def surface_gravity(self):
        return Const.G.value * (self.mass() * Const.M_earth.value) / (self.radius() * Const.R_earth.value)**2
    def radiative_radius(self):
        Teq = self.temperature()
        sgravity = self.surface_gravity()
        #molec_weight = 0.75 + 0.25/4 # H:75%, He:25%, they use 3.5 g/mol
        molec_weight = 3.5e3 # g/mol
        self.ratm = 9 * Const.k_B.value * Teq / (sgravity * molec_weight) * Const.R_earth.value
        return self.ratm
    def envelope_radius(self):
        mass_term = 2.06 * (self.mp)**(-0.21)
        flux_term = (self.fbol/Fbol_earth)**(0.044)
        age_term  = (self.age/5000)**(-0.18)
        fenv_term = ((self.menv/self.mp)/0.05)**(0.59)
        self.renv = mass_term * fenv_term * flux_term * age_term
        #self.renv += self.radiative_radius()
        return self.renv
    
    
    def grad_envelope_radius(self):
        # What if no initial envelope??
        if (self.mp <= self.mcore):
            return 0.0
        # Derivative wrt planet mass
        mass_term = 2.06 * (self.mp)**(-0.21)
        grad_mass_term = 2.06 * (-0.21) * (self.mp)**(-0.21) * self.mloss

        fenv_term = (( (self.mp - self.mcore) /self.mp)/0.05)**(0.59)
        grad_fenv_term = (0.05)**(-0.59) * 0.59 * fenv_term**((0.59-1)/0.59) * self.mcore * self.mloss / (self.mp)**2

        fbol = self.Lbol(self.age) / (4 * np.pi * (self.dist*Const.au.to('cm').value)**2)
        fbol_deriv = self.gradLbol(self.age) / (4 * np.pi * (self.dist*Const.au.to('cm').value)**2)

        flux_term = (Fbol_earth)**(-0.044) * (fbol)**(0.044)
        grad_flux_term =  (Fbol_earth)**(-0.044) * (fbol)**(0.044-1) * 0.044 * fbol_deriv

        age_term  = (self.age/5000)**(-0.18)
        grad_age_term = (5000)**(0.18) * (-0.18) * (self.age)**(-1.18)
        
        terms = [mass_term, fenv_term, flux_term, age_term]
        derivs = [grad_mass_term, grad_fenv_term, grad_flux_term, grad_age_term]
        
        grad_renv = 0.0
        for i in range(len(terms)):
            deriv_term = derivs[i]
            for j in range(len(terms)):
                if j==i: continue
                deriv_term *= terms[j]
            grad_renv += deriv_term
        
        if not (np.isfinite(grad_renv)):
            print(f"fenv_term({self.mp},{self.mcore}) = {fenv_term}")
            print(f"grad_fenv_term({self.mp},{self.mcore}) = {grad_fenv_term}")
            print(f"{grad_renv}={age_term}*{flux_term}*({mass_term}*{grad_fenv_term} + {fenv_term}*{grad_mass_term})")
            raise ValueError("Envelope Radius derivative is not a number")
        return grad_renv
        
    
    def evaporate(self, mlost):
        self.menv -= mlost
        if self.menv < 0: self.menv = 0.0
        return self.menv
    
    def mloss_rate(self, i):
        self.mloss = mass_loss(self.LxuvTrack[i], self.rp, self.mp, self.dist, mstar=self.mstar, eff=self.eff, beta=self.beta)
        return self.mloss

    def updateTracks(self, i):
        self.RadiusTrack[i] = self.rp
        self.MassTrack[i] = self.mp
        self.CoreRadiusTrack[i] = self.rcore
        self.CoreMassTrack[i] = self.mcore
        self.EnvRadiusTrack[i] = self.renv
        self.EnvMassTrack[i] = self.menv
        self.EnvMassFractionTrack[i] = self.fenv
        self.DensityTrack[i] = self.mp * Const.M_earth.to('g').value / ( 4/3 * np.pi * (self.rp*Const.R_earth.to('cm').value)**3)
        self.MassLossTrack[i] = self.mloss

    def step(self, i, devolve=False, legacy=True):
        ni = i-1 if devolve else i+1 # next index
        if ni < 0 or ni >= len(self.AgeTrack):
            #self.updateTracks(i)
            return

        # Draw new fluxes from stellar tracks
        self.fbol = self.LbolTrack[i] / ( 4 * np.pi * (self.dist*Const.au.to('cm').value)**2 )
        self.fxuv = self.LxuvTrack[i] / ( 4 * np.pi * (self.dist*Const.au.to('cm').value)**2 )
        
        # Update temporal parameters
        self.age = self.AgeTrack[i]
        time_step = np.abs( self.AgeTrack[ni] - self.AgeTrack[i] )
 
        # Apply mass loss
        mloss = self.mloss_rate(i)
        if devolve: mloss = -mloss
        elif self.rp <= self.rcore or self.menv <= 0:
            self.mloss = 0.0
            mloss = 0.0
        self.menv = self.evaporate(mloss*time_step)
        self.mp = self.mcore + self.menv

        # Update radii
        """
        # In case grad_envelope_radius stops working

        self.renv = self.envelope_radius()
        self.rp = self.rcore + self.renv + self.radiative_radius() # Risky!
        """
        
        if legacy:
            old_renv = self.renv
            self.renv = self.envelope_radius()
            renv_step = self.grad_envelope_radius() * time_step
            #print(i,') ',self.renv-old_renv,' ',renv_step)
            self.rp = self.rcore + self.renv #+ self.radiative_radius() # Risky!

        else:
            renv_step = self.grad_envelope_radius() * time_step
            if devolve: renv_step *= -1
            #print(i, renv_step)
            self.renv = self.renv + renv_step if self.renv > 0 else self.renv
            self.rp   = self.rp   + renv_step if self.renv > 0 else self.rp

        self.fenv = self.menv / self.mp

        
        # Update tracks
        self.updateTracks(ni)

    def parameterValidation(self, age_ind):
        
        if (self.fenv is not None) and not (0.0 <= self.fenv <= 1.0):
            raise ValueError("Error: envelope mass fraction must be between 0 and 1")
        if (self.mcore is not None and self.mp is not None) and (self.mcore > self.mp):
            raise ValueError("Error: the core mass can't be greater than the total mass")
        if (self.menv is not None and self.mp is not None) and (self.menv > self.mp):
            raise ValueError("Error: the envelope mass can't be greater than the total mass")
        if (self.rcore is not None and self.rp is not None) and (self.rcore > self.rp):
            raise ValueError("Error: the core radius can't be greater than the total radius")
        if (self.renv is not None and self.rp is not None) and (self.renv > self.rp):
            raise ValueError("Error: the envelope radius can't be greater than the total radius")
        if (self.mstar is None): raise ValueError("Error: undefined stellar mass")

        Lbol = self.LbolTrack[self.init_index]

        # -- Scenarios
        # I start out with minimal parameters. What else do I need?
        # Solution: you can estimate all parameters from just any three of them.
        # To make things easier, I constrain the combination of allowed initiap parameters
        # to a set of predefined scenarios. 

        # 1) Observational: you have mass and radius
        if (self.mp is not None and self.rp is not None):
            if (self.fenv is not None):
                if self.menv  is None: self.menv  = self.mp * self.fenv
                if self.mcore is None: self.mcore = self.mp - self.menv
                if self.renv  is None: self.renv  = self.envelope_radius()
                if self.rcore is None: self.rcore = self.rp - self.renv #- self.radiative_radius()
            elif (self.rcore is not None):
                if self.renv  is None: self.renv  = self.rp - self.rcore #- self.radiative_radius()
                if self.fenv  is None:
                    self.fenv = mass_fraction(self.mp, self.rp, self.fbol, self.age, self.rcore, verbose=True)
                    envelope_radius(mp=self.mp, rp=self.rp, Lbol=Lbol/Const.L_sun.to('erg/s').value, au=self.dist, age=self.age, fenv=self.fenv, verbose=True)
                if self.menv  is None: self.menv = self.mp * self.fenv
                if self.mcore is None: self.mcore = self.mp - self.menv
            else:
                raise ValueError("Error: not enough parameters defined. ")

        # 2) Simulation: you define core mass and radius
        elif (self.mcore is not None and self.rcore is not None):
            if (self.menv is not None):
                # To obtain: rp, mp, renv, fenv
                if self.mp   is None: self.mp   = self.mcore + self.menv
                if self.fenv is None: self.fenv = self.menv / self.mp
                if self.renv is None: self.renv = self.envelope_radius()
                if self.rp   is None:
                    self.rp = self.rcore + self.renv
                    #self.rp += self.radiative_radius()
            elif (self.fenv is not None):
                if self.menv is None: self.menv = self.fenv * self.mcore / (1 - self.fenv)
                if self.mp   is None: self.mp   = self.menv + self.mcore
                if self.renv is None: self.renv = self.envelope_radius()
                if self.rp  is None:
                    self.rp = self.rcore + self.renv
                    #self.rp += self.radiative_radius()
            elif (self.renv is not None):
                warnings.warn("This method requires analytical non-linear equation solving to obtain the planet mass.")
                if self.mp is None:   self.mp = planet_mass(self.mcore, self.renv, Lbol, self.dist, self.age, verbose=0)
                if self.menv is None: self.menv = self.mp - self.mcore
                if self.fenv is None: self.fenv = self.menv / self.mp
                if self.rp is None:
                    self.rp = self.rcore + self.renv
                    #self.rp += self.radiative_radius()
            else:
                raise NotImplementedError("Error: WIP")

        else:
            raise ValueError("Error: not enough parameters defined. ")
 
    def reset_to_initial(self):
        # Reset initial values and input age  
        self.rp    = self.RadiusTrack[self.init_index]
        self.mp    = self.MassTrack[self.init_index]
        self.rcore = self.CoreRadiusTrack[self.init_index]
        self.mcore = self.CoreMassTrack[self.init_index]
        self.renv  = self.EnvRadiusTrack[self.init_index]
        self.menv  = self.EnvMassTrack[self.init_index]
        self.fenv  = self.EnvMassFractionTrack[self.init_index]
        self.ratm  = 0.0
        self.mloss = self.MassLossTrack[self.init_index]
        self.age   = self.AgeTrack[self.init_index]
        self.fbol  = self.LbolTrack[self.init_index] / ( 4 * np.pi * (self.dist*Const.au.to('cm').value)**2 )
        self.fxuv  = self.LxuvTrack[self.init_index] / ( 4 * np.pi * (self.dist*Const.au.to('cm').value)**2 )

    def generateTracks(self, AgeTrack=None, LbolTrack=None, LxTrack=None, LeuvTrack=None, legacy=True): 

        if AgeTrack is not None:
            self.AgeTrack = AgeTrack
            self.LbolTrack = LbolTrack
            self.LxTrack = LxTrack
            self.LeuvTrack = LeuvTrack
            self.LxuvTrack = LxTrack + LeuvTrack

        # Find irradiation levels at given age
        self.init_index = np.transpose(np.nonzero(self.age < self.AgeTrack))[0][0]
        self.fbol = self.LbolTrack[self.init_index] / ( 4 * np.pi * (self.dist*Const.au.to('cm').value)**2 )
        self.fxuv = self.LxuvTrack[self.init_index] / ( 4 * np.pi * (self.dist*Const.au.to('cm').value)**2 )
   
        self.parameterValidation(self.init_index)

        # Initialize custom tracks
        self.RadiusTrack = np.zeros_like(self.AgeTrack)
        self.MassTrack = np.zeros_like(self.AgeTrack)
        self.CoreRadiusTrack = np.zeros_like(self.AgeTrack)
        self.CoreMassTrack = np.zeros_like(self.AgeTrack)
        self.EnvRadiusTrack = np.zeros_like(self.AgeTrack)
        self.EnvMassTrack = np.zeros_like(self.AgeTrack)
        self.EnvMassFractionTrack = np.zeros_like(self.AgeTrack)
        self.DensityTrack = np.zeros_like(self.AgeTrack)
        self.MassLossTrack = np.zeros_like(self.AgeTrack)
 
        self.updateTracks(self.init_index)

        # Evolve forwards & backwards from its current age
        forward_steps = len(self.AgeTrack[self.init_index:])
        backward_steps = len(self.AgeTrack[:self.init_index])

        for i in range( forward_steps ):
            self.step(self.init_index + i, legacy=legacy)

        self.reset_to_initial()

        # Fix for mass loss rate at starting age
        self.MassLossTrack[self.init_index] = self.mloss_rate(self.init_index)
        
        for i in range( backward_steps ):
            self.step(self.init_index - i, devolve=True, legacy=legacy)
        
        # Fix missing initial track values
        self.RadiusTrack[0]     = self.RadiusTrack[1]
        self.MassTrack[0]       = self.MassTrack[1]
        self.CoreRadiusTrack[0] = self.CoreRadiusTrack[1]
        self.CoreMassTrack[0]   = self.CoreMassTrack[1]
        self.EnvRadiusTrack[0]  = self.EnvRadiusTrack[1]
        self.EnvMassTrack[0]    = self.EnvMassTrack[1]
        self.EnvMassFractionTrack[0] = self.EnvMassFractionTrack[1]
        self.DensityTrack[0]    = self.DensityTrack[1]
        self.MassLossTrack[0]   = self.MassLossTrack[1]

        self.reset_to_initial()

        def at_age(self, lookup_age):
            """
            Returns Planet object with initial parameters set to
            the input lookup_age.
            """
            # Lookup age index
            # Make planet object
            # return p
            pass



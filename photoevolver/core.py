"""

[photoevolver.core]

"""

from copy import deepcopy
import numpy as np
import astropy.constants as Const
import Mors as mors

from .EvapMass.planet_structure import mass_to_radius_solid as owen_radius
from .EvapMass.planet_structure import solid_radius_to_mass as owen_mass

# Wrapper for EvapMass M-R relation
def OwenMassRadiusRelation(**kwargs):
    """
    Input:
        mass
        radius
        Xice
        Xiron
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
    Input:
            radius      :(float) Radius in Earth radii
            mass 	:(float) Mass in Earth masses. Only specify EITHER mass OR radius!!
            mode 	:(str) "rocky", "volatile"
    Returns:
            [0] 	:(float) Radius if input mass, Mass if input radius.
            [1] 	:(float) Error in mass or radius, if error provided
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
    return mass

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
    if error == 0.0: return radius
    # Errors
    const_term = model['c_err'] * mass ** model['pow']
    mass_term =  model['const'] * model['pow'] * error * mass ** (model['pow']-1)
    pow_term =   model['const'] * mass ** model['pow'] * np.log(mass) * model['p_err']
    radius_err = np.sqrt( (const_term)**2 + (mass_term)**2 + (pow_term)**2 )
    return radius, radius_err



def evolve_forward(planet, mloss, struct, star, time_step=1.0, age_end=1e4, **kwargs):
    """
    Evolves a planet forward in time.
    Parameters:
        planet:     (object) photoevolve.core.Planet object previously initialised
        mloss:      (class) Mass loss formulation from photoevolve.massloss
        struct:     (class) Function that relates the envelope radius to the envelope mass fraction
        star:       (object OR dict) Description of X-ray evolution of the host star. Three options:
                        > Mors.Star object
                        > Dict of two arrays: 'Lxuv' and 'Lbol' luminosity tracks of length (age_end - planet.age + 1)/time_step
        time_step:  (float) Time step of the simulation in Myr
        age_end:    (float) Oldest age to which to run simulation
        **kwargs:   Specific parameters to tailor your simulation:
                        'eff': mass loss efficiency if using the EnergyLimited formulation.
                        'beta': Rxuv / Rp
                        'mstar': Mass of the host star in M_sun.
    """
    pl = deepcopy(planet)
    if type(pl) != Planet:
        raise ValueError("the planet must be an instance of the photoevolve.core.Planet class")
    if not callable(struct):
        raise ValueError("struct must be a function")
    if age_end <= pl.age:
        raise ValueError("The maximum age must be greater than the planet's age")
    if type(star) != mors.Star:
        raise NotImplementedError("The star must be a Mors.Star instance")

    tracks = _init_tracks()
    params = kwargs

    # Initial definition of Renv and Rp
    params = Evolve.update_params(params, pl, star)
    pl.renv = struct(**params)
    pl.rp = pl.rcore + pl.renv
    # Update input planet unknowns
    planet.renv = pl.renv
    planet.rp = pl.rp

    while(pl.age < age_end):
        # Parameters
        params = Evolve.update_params(params, pl, star)
        # Evolution
        # - mass loss
        if mloss is not None and pl.fenv > 1e-4:
            pl.menv -= mloss(**params)
            pl.mp = pl.mcore + pl.menv 
            pl.fenv = pl.menv / pl.mp
            params = _update_params(params, pl, star)
        # - envelope radius
        pl.renv = struct(**params)
        pl.rp = pl.rcore + pl.renv
        # Update Tracks
        tracks = _update_tracks(tracks, pl, star)
        # Jump
        pl.age += time_step
    return tracks


def evolve_back(planet, mloss, struct, star, time_step=1.0, age_end=1.0, **kwargs):
    """
    Evolves a planet backwards in time.
    Parameters:
        planet:     (object) photoevolve.core.Planet object previously initialised
        mloss:      (class) Mass loss formulation from photoevolve.massloss
        struct:     (class) Function that relates the envelope radius to the envelope mass fraction
        star:       (object OR dict) Description of X-ray evolution of the host star. Three options:
                        > Mors.Star object
                        > Dict of two arrays: 'Lxuv' and 'Lbol' luminosity tracks of length (age_end - planet.age + 1)/time_step
        time_step:  (float) Time step of the simulation in Myr
        age_end:    (float) Earliest age to which to run simulation
        **kwargs:   Specific parameters to tailor your simulation:
                        'eff': mass loss efficiency if using the EnergyLimited formulation.
                        'beta': Rxuv / Rp
                        'mstar': Mass of the host star in M_sun.
    """
    pl = deepcopy(planet)
    if type(pl) != Planet:
        raise ValueError("the planet must be an instance of the photoevolve.core.Planet class")
    if not callable(struct):
        raise ValueError("struct must be a function")
    if age_end >= pl.age:
        raise ValueError("The miminum age must be lower than the planet's age")
    if type(star) != mors.Star:
        raise NotImplementedError("The star must be a Mors.Star instance")

    tracks = _init_tracks()
    params = kwargs

    # Initial definition of Renv and Rp
    params = Evolve.update_params(params, pl, star)
    pl.renv = struct(**params)
    pl.rp = pl.rcore + pl.renv
    # Update input planet unknowns
    planet.renv = pl.renv
    planet.rp = pl.rp
    if pl.fenv < 1e-4: pl.fenv = 1e-4

    while(pl.age > age_end):
        # Parameters
        params = Evolve.update_params(params, pl, star)
        # Evolution
        # - mass loss
        if mloss is not None and pl.fenv > 1e-4:
            pl.menv += mloss(**params)
            pl.mp = pl.mcore + pl.menv 
            pl.fenv = pl.menv / pl.mp
            params = _update_params(params, pl, star)
        # - envelope radius
        pl.renv = struct(**params)
        pl.rp = pl.rcore + pl.renv
        # Update Tracks
        tracks = _update_tracks(tracks, pl, star)
        # Jump
        pl.age -= time_step
    tracks = _reverse_tracks(tracks)
    return tracks


def _init_tracks():
    fields = ('Age', 'Lbol', 'Lx', 'Rp', 'Mp', 'Menv', 'Renv', 'Fenv', 'Mloss', 'Dens')
    tracks = dict( Age=[], Lbol=[], Rp=[], Mp=[] )
    return tracks

def _update_tracks(tracks, planet, star):
    tracks['Age'].append(planet.age)
    tracks['Lbol'].append(star.Lbol(planet.age))
    tracks['Rp'].append(planet.rp)
    tracks['Mp'].append(planet.mp)
    return tracks

def _reverse_tracks(tracks):
    for k in tracks.keys():
        tracks[k].reverse()
    return tracks

def _update_params(params, planet, star):
    fbol = star.Lbol(planet.age) / (planet.dist*Const.au.to('cm').value)**2 / (4*np.pi)
    params.update(planet.__dict__)
    params['fbol'] = fbol
    params['mass'] = planet.mp
    params['radius'] = planet.rp
    params['Lxuv'] = star.Lx(planet.age) + star.Leuv(planet.age)
    params['mstar'] = star.Mstar
    return params

# LEGACY
class Evolve:
    def forward(planet, mloss, struct, star, time_step=1.0, age_end=1e4, **kwargs):
        """
        Evolves a planet forward and backwards.
        Parameters:
            planet:     (object) photoevolve.core.Planet object previously initialised
            mloss:      (class) Mass loss formulation from photoevolve.massloss
            struct:     (class) Function that relates the envelope radius to the envelope mass fraction
            star:       (object OR dict) Description of X-ray evolution of the host star. Three options:
                            > Mors.Star object
                            > Dict of two arrays: 'Lxuv' and 'Lbol' luminosity tracks of length (age_end - planet.age + 1)/time_step
            time_step:  (float) Time step of the simulation in Myr
            age_end:    (float) Oldest age to which to run simulation
            **kwargs:   Specific parameters to tailor your simulation:
                            'eff': mass loss efficiency if using the EnergyLimited formulation.
                            'beta': Rxuv / Rp
                            'mstar': Mass of the host star in M_sun.
        """
        pl = deepcopy(planet)
        print(type(pl))
        if type(pl) != Planet:
            raise ValueError("the planet must be an instance of the photoevolve.core.Planet class")
        if not callable(struct):
            raise ValueError("struct must be a function")
        if age_end <= pl.age:
            raise ValueError("The miminum and maximum ages must be lower/greater than the defined pl age")
        if type(star) != mors.Star:
            raise NotImplementedError("The star must be a Mors.Star instance")

        tracks = dict( Age=[], Lbol=[], Rp=[], Mp=[] )
        params = kwargs

        # Initial definition of Renv and Rp
        params = Evolve.update_params(params, pl, star)
        pl.renv = struct(**params)
        pl.rp = pl.rcore + pl.renv
        # Update input planet unknowns
        planet.renv = pl.renv
        planet.rp = pl.rp

        while(pl.age < age_end):
            # Parameters
            params = Evolve.update_params(params, pl, star)
            # Evolution
            # - mass loss
            if mloss is not None and pl.fenv > 1e-4:
                pl.menv -= mloss(**params)
                pl.mp = pl.mcore + pl.menv 
                pl.fenv = pl.menv / pl.mp
                params = Evolve.update_params(params, pl, star)
            # - envelope radius
            pl.renv = struct(**params)
            pl.rp = pl.rcore + pl.renv
            # Update Tracks
            tracks = Evolve.update_tracks(tracks, pl, star)
            # Jump
            pl.age += time_step
        return tracks

    def update_tracks(tracks, planet, star):
        tracks['Age'].append(planet.age)
        tracks['Lbol'].append(star.Lbol(planet.age))
        tracks['Rp'].append(planet.rp)
        tracks['Mp'].append(planet.mp)
        return tracks

    def update_params(params, planet, star):
        fbol = star.Lbol(planet.age) / (planet.dist*Const.au.to('cm').value)**2 / (4*np.pi)
        params.update(planet.__dict__)
        params['fbol'] = fbol
        params['mass'] = planet.mp
        params['radius'] = planet.rp
        params['Lxuv'] = star.Lx(planet.age) + star.Leuv(planet.age)
        params['mstar'] = star.Mstar
        return params


class Planet:
    def __init__(self, mp=None, rp=None, mcore=None, rcore=None, menv=None, renv=None, fenv=None, dist=None, age=None, comp=None, mr=None):
        """
        Parameters:
            mp:     Planet mass (M_earth)
            rp:     Planet radius (R_earth)
            mcore:  Core mass (M_core)
            rcore:  Core radius (R_core)
            menv:   Envelope mass (M_earth)
            renv:   Envelope radius (R_earth)
            fenv:   Envelope mass fraction (M_env/M_planet)
            dist:   Separation between the planet and the star (AU)
            age:    Age of the planet (Myr)
            comp:   Dictionary with keys [Xice, Xrock, Xiron] (mass abundances)
            mr:     Mass-Radius relation to use to calculate core radius
        """
        self.comp = comp
        self.mp = mp
        self.rp = rp
        self.mcore = mcore
        self.rcore = rcore
        self.menv = menv
        self.renv = renv
        self.fenv = fenv
        self.dist = dist
        self.age = age
        self.mr = mr
        # --
        self.input_check()

    def input_check(self):
        if self.dist is None or self.age is None:
            raise ValueError("age or orbital distance undefined")

        if self.mr is None or not callable(self.mr):
            self.mr = OtegiMassRadiusRelation

        if type(self.comp) != dict or \
           (type(self.comp) == dict and tuple(self.comp.keys()) != ['Xiron','Xice']) or \
           sum( list(self.comp.values()) ) > 1.0:
                self.comp = {'Xiron':1/3, 'Xice':0.0}
        
        if self.mcore is None and self.rcore is None:
            raise ValueError("core mass or radius undefined")
        elif self.mcore is None: self.mcore = self.mr(radius=self.rcore, **self.comp)
        elif self.rcore is None: self.rcore = self.mr(mass=self.mcore, **self.comp)

        if self.rcore is None:
            print(f" Warning: core radius undefined.", end='')
            print(f" Will be estimated from a mass-radius relation ", end='')
            print(f" -> {self.rcore:.3f} Earth radii")

        elif self.mcore is None:
            print(f" Warning: core mass undefined.", end='')
            print(f" Will be estimated from a mass-radius relation ", end='')
            print(f" -> {self.mcore:.3f} Earth masses")

        # Temporal solution until solving for fenv is implemented
        if self.fenv is None:
            raise NotImplementedError("fenv must be defined to calculate envelope radius")

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
            self.mp   = self.menv + self.mcore       if self.mp   is None else self.mp    

        else: raise ValueError("not enough planet parameters defined") 
        
    def __repr__(self): 
        mcore_str = f"{self.mcore:.2f}"      if self.mcore is not None else "TBD"
        rcore_str = f"{self.rcore:.2f}"      if self.rcore is not None else "TBD"
        menv_str = f"{self.menv:.2f}"        if self.menv is not None else "TBD"
        renv_str = f"{self.renv:.2f}"        if self.renv is not None else "TBD"
        mp_str = f"{self.mp:.2f}"            if self.mp is not None else "TBD"
        rp_str = f"{self.rp:.2f}"            if self.rp is not None else "TBD"
        fenv_str = f"[{self.fenv*100:.2f}%]" if self.fenv is not None else "TBD"
        comp_str = f"[ice:{self.comp['Xice']*100:.0f}% iron:{self.comp['Xiron']*100:.0f}%]"

        msg =  f"<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}>\n\n" \
            + f"".ljust(10)         + f"Radius".ljust(10)   + f"Mass".ljust(10)               + "\n" \
            + f"Core".ljust(10)     + rcore_str.ljust(10)   + mcore_str.ljust(10)  + comp_str + "\n" \
            + f"Envelope".ljust(10) + renv_str.ljust(10)    + menv_str.ljust(10)   + fenv_str + "\n" \
            + f"Total".ljust(10)    + rp_str.ljust(10)      + mp_str                          + "\n"
        return msg

    def __str__(self):
        return self.__repr__()





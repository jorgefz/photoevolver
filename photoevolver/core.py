"""

[photoevolver.core]

"""

from copy import deepcopy
import numpy as np
import astropy.constants as Const
import matplotlib.pyplot as plt
import Mors as mors

from .structure import fenv_solve
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

def indexable(obj):
    try:
        obj[:]
    except TypeError:
        return False
    return True

"""
def get_time_track(time_step, age_end):
    if indexable(time_step):
        return time_step
    time_track = np.


def get_star_track(star, steps):
    try:
        import Mors as mors
        star
    except ImportError:
        # check if is dict
"""

"""

GetItem
Tracks['Age'], Tracks['Radius']

SetItem
Tracks['Fenv'][0] = 0.05

Interpolation
Tracks.Fenv(100) # fenv @ 100 Myr

"""

class Tracks:
    def __init__(self, data : dict):
        self.tracks = data
        self.interp()
    
    def keys(self):
        return self.tracks.keys()
    
    def __getitem__(self, field):
        return self.tracks[field]

    def __len__(self):
        return len(self.tracks)

    def __str__(self):
        return "photoevolver.core.Tracks instance"

    def __repr__(self):
        return __str__()

    def __add__(self, t2):
        if type(t2) != Tracks: raise TypeError(f" Tracks can only concatenate with another Tracks instance, not '{type(t2)}'")
        elif max(self['Age']) <= min(t2['Age']):
            # Track 2 is to the right (older ages)
            new = Tracks(self.tracks)
            for f in new.tracks.keys():
                new.tracks[f] += t2.tracks[f]
            new.interp()
            return new
        elif min(self['Age']) >= max(t2['Age']):
            # Track 2 is to the left (younger ages)
            new = Tracks(t2.tracks)
            for f in new.tracks.keys():
                new.tracks[f] += self.tracks[f]
            new.interp()
            return new
        else:
            raise ValueError(f"the ages of Tracks to concatenate must not overlap: ({min(self['Age'])},{max(self['Age'])}) + ({min(t2['Age'])},{max(t2['Age'])}) Myr")

    def append(self, t2):
        self = self + t2
        return self

    def interp(self):
        from scipy.interpolate import interp1d
        for key in self.keys():
            if key == 'Age': continue
            func = interp1d(x=self.tracks['Age'], y=self.tracks[key])
            setattr(self, key, func)
    
    def planet(self, age, p):
        return Planet(mp=self.Mp(age), rp=self.Rp(age), mcore=p.mcore, rcore=p.rcore, menv=self.Menv(age), renv=self.Renv(age), fenv=self.Fenv(age), dist=p.dist, age=age)


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
                        'fenv_min':
                        'renv_min': 
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
    if 'fenv_min' not in kwargs: kwargs['fenv_min'] = 1e-9
    if 'renv_min' not in kwargs: kwargs['renv_min'] = 0.001

    tracks = _init_tracks()
    params = kwargs

    # Update planet parameters with structure equation
    params = _update_params(params, pl, star)
    # - radii
    if pl.fenv is not None:
        if pl.fenv > kwargs['fenv_min']:
            pl.renv = struct(**params) if pl.renv is None else pl.renv
        else: pl.renv = 0.0
        pl.rp = pl.rcore + pl.renv if pl.rp is None else pl.rp
    # - masses
    elif pl.renv is not None:
        if pl.renv > kwargs['renv_min']: 
            pl.fenv = fenv_solve(fstruct=struct, **params) if pl.fenv is None else pl.fenv
        else: pl.fenv = 0.0
        pl.mp = pl.mcore / (1 - pl.fenv) if pl.mp is None else pl.mp
        pl.menv = pl.mp * pl.fenv if pl.menv is None else pl.menv 
    params = _update_params(params, pl, star)

    while(pl.age < age_end):
        # Parameters
        params = _update_params(params, pl, star)
        # Evolution
        # - mass loss
        if mloss is not None and pl.fenv > kwargs['fenv_min']:
            pl.menv -= mloss(**params)
            pl.mp = pl.mcore + pl.menv 
            pl.fenv = pl.menv / pl.mp
            params = _update_params(params, pl, star)
        elif pl.fenv < kwargs['fenv_min']:
            pl.fenv = 0.0
            pl.renv = 0.0
        # - envelope radius
        if pl.renv > kwargs['renv_min']:
            pl.renv = struct(**params)
            if pl.renv <= kwargs['renv_min']: pl.renv = 0.0
            pl.rp = pl.rcore + pl.renv
        # Update Tracks
        tracks = _update_tracks(tracks, pl, star)
        # Jump
        pl.age += time_step
    return Tracks(tracks)


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
    if 'fenv_min' not in kwargs: kwargs['fenv_min'] = 1e-9
    if 'renv_min' not in kwargs: kwargs['renv_min'] = 0.001
 
    # Zero envelope fix
    if pl.fenv is not None and pl.fenv < kwargs['fenv_min']: pl.fenv = kwargs['fenv_min']
    if pl.renv is not None and pl.renv < kwargs['renv_min']: pl.renv = kwargs['renv_min']
    
    tracks = _init_tracks()
    params = kwargs

    # Update planet parameters with structure equation
    params = _update_params(params, pl, star)
    # - radii 
    pl.renv = struct(**params) if pl.renv is None else pl.renv
    pl.rp = pl.rcore + pl.renv if pl.rp is None else pl.rp
    # - masses
    pl.fenv = fenv_solve(fstruct=struct, **params) if pl.fenv is None else pl.fenv
    pl.mp = pl.mcore / (1 - pl.fenv) if pl.mp is None else pl.mp
    pl.menv = pl.mp * pl.fenv if pl.menv is None else pl.menv
    # Update with newly calculated values
    params = _update_params(params, pl, star)

    while(pl.age > age_end):
        # Parameters
        params = _update_params(params, pl, star)
        # Evolution
        # - mass loss
        #if mloss is not None and pl.fenv > kwargs['fenv_min']:
        if mloss is not None:
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
    return Tracks(tracks)



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
        

def _init_tracks():
    fields = ('Age', 'Lbol', 'Lxuv', 'Rp', 'Mp', 'Menv', 'Renv', 'Fenv', 'Dens')
    #tracks = dict( Age=[], Lbol=[], Rp=[], Mp=[] )
    tracks = {f:[] for f in fields}
    return tracks

def _planet_density(mp, rp):
    """
    mp: mass in Earth masses
    rp: radius in Earth radii
    Returns: density in g/cm^3
    """
    return Const.M_earth.to('g').value * mp / (4*np.pi/3 * (Const.R_earth.to('cm').value * rp)**3)

def _update_tracks(tracks, planet, star):
    tracks['Age'].append(planet.age)
    tracks['Lbol'].append(star.Lbol(planet.age))
    tracks['Lxuv'].append(star.Lx(planet.age)+star.Leuv(planet.age))
    tracks['Rp'].append(planet.rp)
    tracks['Mp'].append(planet.mp)
    tracks['Menv'].append(planet.menv)
    tracks['Renv'].append(planet.renv)
    tracks['Fenv'].append(planet.fenv)
    tracks['Dens'].append( _planet_density(mp=planet.mp, rp=planet.rp) )
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
    params['Lbol'] = star.Lbol(planet.age)
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

        elif self.fenv is None and self.renv is None:
            raise ValueError(" Error: either envelope radius (renv) or mass fraction (fenv) must be defined")
        else: raise ValueError("not enough planet parameters defined")


        
    def __repr__(self): 
        mcore_str = f"{self.mcore:.2f}"      if self.mcore is not None else "TBD"
        rcore_str = f"{self.rcore:.2f}"      if self.rcore is not None else "TBD"
        menv_str = f"{self.menv:.2f}"        if self.menv is not None else "TBD"
        renv_str = f"{self.renv:.2f}"        if self.renv is not None else "TBD"
        mp_str = f"{self.mp:.2f}"            if self.mp is not None else "TBD"
        rp_str = f"{self.rp:.2f}"            if self.rp is not None else "TBD"
        fenv_str = f"[{self.fenv*100:.5f}%]" if self.fenv is not None else "TBD"
        comp_str = f"[ice:{self.comp['Xice']*100:.0f}% iron:{self.comp['Xiron']*100:.0f}%]"

        msg =  f"<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}>\n\n" \
            + f"".ljust(10)         + f"Radius".ljust(10)   + f"Mass".ljust(10)               + "\n" \
            + f"Core".ljust(10)     + rcore_str.ljust(10)   + mcore_str.ljust(10)  + comp_str + "\n" \
            + f"Envelope".ljust(10) + renv_str.ljust(10)    + menv_str.ljust(10)   + fenv_str + "\n" \
            + f"Total".ljust(10)    + rp_str.ljust(10)      + mp_str                          + "\n"
        return msg

    def __str__(self):
        return self.__repr__()





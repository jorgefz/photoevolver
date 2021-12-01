"""

[photoevolver.core]

"""

from copy import deepcopy
import numpy as np
import astropy.constants as Const
import matplotlib.pyplot as plt
import pickle

from .structure import fenv_solve
#from .EvapMass.planet_structure import mass_to_radius_solid as owen_radius
#from .EvapMass.planet_structure import solid_radius_to_mass as owen_mass

from .owenwu17 import mass_to_radius as owen_radius
from .owenwu17 import radius_to_mass as owen_mass


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

    *Note: EITHER mass OR radius must be input.
    
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


def load_tracks(filename): 
    with open(filename, 'rb') as handle:
        tracks = pickle.load(handle)
    return tracks

class Tracks:
    def __init__(self, data : dict, base_pl):
        for k in data.keys():
            if not indexable(data[k]): continue
            data[k] = np.array(data[k])
        self.tracks = data
        self.pl = base_pl
        self.interp()
    
    def keys(self):
        return self.tracks.keys()
    
    def save(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, field):
        return self.tracks[field]

    def __len__(self):
        return len(self.tracks)

    def __str__(self):
        return "photoevolver.core.Tracks instance"

    def __repr__(self):
        return self.__str__()

    def __add__(self, t2):
        if type(t2) != Tracks:
            raise TypeError(f"Tracks can only be concatenated \
                    with another Tracks instance, not '{type(t2)}'")
        elif max(self['Age']) <= min(t2['Age']):
            # Track 2 is to the right (older ages)
            new = Tracks(self.tracks, t2.pl)
            for f in new.tracks.keys():
                # new.tracks[f] += t2.tracks[f]
                new.tracks[f] = np.append( new.tracks[f], t2.tracks[f] )
            new.interp()
            return new
        elif min(self['Age']) >= max(t2['Age']):
            # Track 2 is to the left (younger ages)
            new = Tracks(t2.tracks, t2.pl)
            for f in new.tracks.keys():
                new.tracks[f] += self.tracks[f]
            new.interp()
            return new
        else:
            raise ValueError(f"The ages of Tracks to concatenate must not \
                    overlap: ({min(self['Age'])},{max(self['Age'])}) \
                    + ({min(t2['Age'])},{max(t2['Age'])}) Myr")

    def append(self, t2):
        self = self + t2
        return self

    def interp(self):
        from scipy.interpolate import interp1d
        for key in self.keys():
            if key == 'Age': continue
            func = interp1d(x=self.tracks['Age'], y=self.tracks[key])
            setattr(self, key, func)
    
    def planet(self, age, *args):
        p = self.pl
        q = Planet(mp=self.Mp(age), rp=self.Rp(age), mcore=p.mcore, \
                rcore=p.rcore, menv=self.Menv(age), renv=self.Renv(age), \
                fenv=self.Fenv(age), dist=p.dist, age=age)
        return q


def indexable(obj):
    try:
        obj[:]
    except TypeError:
        return False
    return True


def is_mors_star(obj):
    try:
        import Mors
        return isinstance(obj, Mors.Star)
    except ImportError:
        return False


def GenerateStarTrack(star, ages):
    if isinstance(star, dict):
        if star.keys() != dict(Lxuv=None,Lbol=None).keys():
            raise KeyError("Star dict must only have keys 'Lxuv' and 'Lbol'.")
        if not indexable(star['Lxuv']) and not indexable(star['Lbol']):
            raise ValueError("Star dict values must be array_like")
        if not (len(star['Lxuv']) == len(star['Lbol']) == len(ages)):
            print(len(star['Lxuv']), len(star['Lbol']), len(ages))
            raise ValueError("Star tracks and ages must have the same length")
        return star
    elif is_mors_star(star):
        if max(ages) > max(star.AgeTrack):
            raise ValueError("Max age out of bounds for star object ")
        LxuvTrack = np.array([star.Lx(a)+star.Leuv(a) for a in ages])
        LbolTrack = np.array([star.Lbol(a) for a in ages])
        return dict(Lxuv=LxuvTrack, Lbol=LbolTrack)
    elif callable(star): raise NotImplementedError()
    raise ValueError("Star must be dictionary, Mors.Star instance, \
                    or callable")


def evolve_forward(planet, mloss, struct, star, time_step=1.0, age_end=1e4,\
    ages=None, **kwargs):
    """
    Evolves a planet's gaseous envelope forward in time, taking into account
    stellar high energy emission, natural cooling over time, and
    photoevaporative mass loss.

    Parameters
    ----------
    planet : photoevolve.core.Planet instance
            Planet to be evolved. Its data will not be modified.
    mloss : callable
            Mass loss formulation, preferably the functions defined in the
            'photoevolver.massloss' module.
    struct : callable
            Function that relates the envelope radius to the envelope mass
            fraction and vice-versa, preferably the functions defined in
            the 'photoevolver.structure' module.
    star : Mors.Star object | dict | callable
            Description of X-ray evolution of the host star. Defines the
            XUV and bolometric luminosities of the star at any given age.
             - Mors.Star instance: the luminosities will be drawn from
            the Lx, Leuv, and Lbol tracks.
             - dict: must have keys 'Lxuv' and 'Lbol', each being an array
            of appropriate length containing the luminosity at each age.
            Must be sorted from young to old.
             - callable: NOT IMPLEMENTED. Must be a function that takes 
            the age in Myr as its single argument, and returns an array of two 
            values, Lxuv and Lbol, in erg/s. E.g. f(age) -> [Lxuv, Lbol]
    time_step : float, optional
            Time step of the simulation in Myr. Default is 1 Myr.
    age_end : float, optional
            Age at which to end the simulation. Must be greater than the
            planet age if evolving forward, and smaller if evolving back.
            Default is 10 000 Myr.
    ages : array_like, optional 
            Array of ages at which to update planet parameters.
            Must be sorted from young to old.
            If None (default), it is ignored and time_step and age_end
            are used instead.
            If array_like, the age of each time step is drawn from it,
            and the simulation will run for N-1 steps, where N is the
            length of the array.
    **eff : float, optional
            Mass loss efficiency (0.0 to 1.0).
            Only used if using 'photoevolver.massloss.EnergyLimited'
            formulation as the mass loss.
    **beta : float, optional
            XUV radius of the planet as a factor of its optical radius
            (Rxuv / Rp).
            Only used if using 'photoevolver.massloss.EnergyLimited'
            formulation as the mass loss.
    **mstar : float, optional
            Mass of the host star in solar masses.
            Required if using ...
            If the input star is a Mors.Star instance, the star mass is
            retrieved from it.
    **fenv_min : float, optional
            Minimum envelope mass fraction at which to consider the planet
            to be completely rocky. Default is ...
    **renv_min : float, optional
            Minimum envelope radius at which to consider the planet
            to be completely rocky. Default is ...

    Note: parameters starting with '**' are keyword arguments (**kwargs)
        and should be used without the double asterisks.

    Returns
    -------
    tracks : photoevolver.core.Tracks instance
            The evolutionary tracks that describe how each of planet's
            parameters evolves in the time range given.

    """
    # input params check
    pl = deepcopy(planet)
    if not isinstance(pl, Planet):
        raise ValueError("The planet must be an instance of the \
                photoevolve.core.Planet class")
    if not callable(struct):
        raise ValueError("struct must be a function")
    if ages is None and age_end <= pl.age:
        raise ValueError("The maximum age must be greater than the planet's age")
    if ages is not None and ages[0] < pl.age:
        raise ValueError("The starting age must be greater or equal \
                to the planet's age")

    # kwargs check
    if 'fenv_min' not in kwargs: kwargs['fenv_min'] = 1e-5
    if 'renv_min' not in kwargs: kwargs['renv_min'] = 0.05
    if 'mstar' not in kwargs and is_mors_star(star):
        kwargs['mstar'] = star.Mstar  
    
    if ages is None:
        length = int(abs(pl.age - age_end)/time_step) + 1
        ages = np.linspace(pl.age, age_end+time_step, length)
    star = GenerateStarTrack(star, ages)

    tracks = _init_tracks()
    params = kwargs
    params = _update_params(params, pl, star, i=0)

    # Calculate current envelope properties
    # - radii
    if pl.fenv is not None:
        if pl.fenv > kwargs['fenv_min']:
            pl.renv = struct(**params)
        else: pl.renv = 0.0
        pl.rp = pl.rcore + pl.renv
    # - masses
    elif pl.renv is not None:
        if pl.renv > kwargs['renv_min']: 
            pl.fenv = fenv_solve(fstruct=struct, **params)
        else: pl.fenv = 0.0
        pl.mp = pl.mcore / (1 - pl.fenv)
        pl.menv = pl.mp * pl.fenv
    
    # MAIN EVOLUTION LOOP
    for i,a in enumerate(ages):
        if (i >= len(ages)-1): break # skip last age
        pl.age = a
        tstep = abs(pl.age - ages[i+1])
        params = _update_params(params, pl, star, i=i)
        if mloss is not None and pl.fenv > kwargs['fenv_min']:
            pl.menv -= mloss(**params) * tstep
            pl.mp = pl.mcore + pl.menv 
            pl.fenv = pl.menv / pl.mp
            params = _update_params(params, pl, star, i=i)
        elif pl.fenv < kwargs['fenv_min'] or pl.rp < pl.rcore:
            pl.fenv = 0.0
            pl.renv = 0.0
            pl.rp = pl.rcore
        if pl.renv > kwargs['renv_min']:
            pl.renv = struct(**params)
            if pl.renv <= kwargs['renv_min']: pl.renv = 0.0
            pl.rp = pl.rcore + pl.renv
        tracks = _update_tracks(tracks, pl, star, i=i)
    return Tracks(tracks, pl)


def evolve_back(planet, mloss, struct, star, time_step=1.0, age_end=1.0, ages = None, **kwargs):
    """
    Evolves a planet's gaseous envelope backwards in time, taking into account
    stellar high energy emission, natural cooling over time, and
    photoevaporative mass loss.

    Parameters
    ----------
    planet : photoevolve.core.Planet instance
            Planet to be evolved. Its data will not be modified.
    mloss : callable
            Mass loss formulation, preferably the functions defined in the
            'photoevolver.massloss' module.
    struct : callable
            Function that relates the envelope radius to the envelope mass
            fraction and vice-versa, preferably the functions defined in
            the 'photoevolver.structure' module.
    star : Mors.Star object | dict | callable
            Description of X-ray evolution of the host star. Defines the
            XUV and bolometric luminosities of the star at any given age.
             - Mors.Star instance: the luminosities will be drawn from
            the Lx, Leuv, and Lbol tracks.
             - dict: must have keys 'Lxuv' and 'Lbol', each being an array
            of appropriate length containing the luminosity at each age.
            Must be sorted from young to old.
             - callable: NOT IMPLEMENTED. Must be a function that takes 
            the age in Myr as its single argument, and returns an array of two 
            values, Lxuv and Lbol, in erg/s. E.g. f(age) -> [Lxuv, Lbol]
    time_step : float, optional
            Time step of the simulation in Myr. Default is 1 Myr.
    age_end : float, optional
            Age at which to end the simulation. Must be greater than the
            planet age if evolving forward, and smaller if evolving back.
            Default is 10 000 Myr.
    ages : array_like, optional 
            Array of ages at which to update planet parameters.
            Must be sorted from min to max.
            If None (default), it is ignored and time_step and age_end
            are used instead.
            If array_like, the age of each time step is drawn from it,
            and the simulation will run for N-1 steps, where N is the
            length of the array.
    **eff : float, optional
            Mass loss efficiency (0.0 to 1.0).
            Only used if using 'photoevolver.massloss.EnergyLimited'
            formulation as the mass loss.
    **beta : float, optional
            XUV radius of the planet as a factor of its optical radius
            (Rxuv / Rp).
            Only used if using 'photoevolver.massloss.EnergyLimited'
            formulation as the mass loss.
    **mstar : float, optional
            Mass of the host star in solar masses.
            Required if using ...
            If the input star is a Mors.Star instance, the star mass is
            retrieved from it.
    **fenv_min : float, optional
            Minimum envelope mass fraction at which to consider the planet
            to be completely rocky. Default is ...
    **renv_min : float, optional
            Minimum envelope radius at which to consider the planet
            to be completely rocky. Default is ...

    Note: parameters starting with '**' are keyword arguments (**kwargs)
        and should be called without the double asterisks.

    Returns
    -------
    tracks : photoevolver.core.Tracks instance
            The evolutionary tracks that describe how each of planet's
            parameters evolves in the time range given.

    """
    # input parameters check
    pl = deepcopy(planet)
    if type(pl) != Planet:
        raise ValueError("the planet must be an instance of the \
                photoevolve.core.Planet class")
    if not callable(struct):
        raise ValueError("struct must be a function")
    if ages is None and age_end >= pl.age:
        raise ValueError("The miminum age must be lower than the planet's age")
    if ages is not None and ages[-1] > pl.age:
        raise ValueError("The oldest age must be lower or equal \
                to the planet's age")

    # kwargs check
    if 'fenv_min' not in kwargs: kwargs['fenv_min'] = 1e-9
    if 'renv_min' not in kwargs: kwargs['renv_min'] = 0.001
    if 'mstar' not in kwargs and is_mors_star(star):
        kwargs['mstar'] = star.Mstar 

    if ages is None:
        length = int(abs(pl.age - age_end)/time_step) + 1
        ages = np.linspace(pl.age, age_end+time_step, length)
    star = GenerateStarTrack(star, ages)

    # Zero envelope fix
    if pl.fenv is not None and pl.fenv < kwargs['fenv_min']:
        pl.fenv = kwargs['fenv_min']
    if pl.renv is not None and pl.renv < kwargs['renv_min']:
        pl.renv = kwargs['renv_min']
    
    tracks = _init_tracks()
    params = kwargs
    params = _update_params(params, pl, star, i=0)
   
    # Update current planet parameter, i=0s
    # - radii 
    if pl.renv is None: pl.renv = struct(**params)
    pl.rp = pl.rcore + pl.renv
    # - masses
    if pl.fenv is None: pl.fenv = fenv_solve(fstruct=struct, **params)
    pl.mp = pl.mcore / (1 - pl.fenv)
    pl.menv = pl.mp * pl.fenv

    for i,a in enumerate(ages):
        if (i == len(ages)-1): break
        pl.age = a
        tstep = abs(a - ages[i+1])
        params = _update_params(params, pl, star, i)
        if mloss is not None:
            pl.menv += mloss(**params) * tstep
            pl.mp = pl.mcore + pl.menv 
            pl.fenv = pl.menv / pl.mp
            params = _update_params(params, pl, star, i)
        pl.renv = struct(**params)
        pl.rp = pl.rcore + pl.renv
        tracks = _update_tracks(tracks, pl, star, i)
    
    tracks = _reverse_tracks(tracks)
    return Tracks(tracks, pl)


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
    tracks = {f:[] for f in fields}
    return tracks

def _planet_density(mp, rp):
    """
    mp: mass in Earth masses
    rp: radius in Earth radii
    Returns: density in g/cm^3
    """
    mass_g = Const.M_earth.to('g').value * mp
    denominator = 4*np.pi/3 * (Const.R_earth.to('cm').value * rp)**3
    return mass_g / denominator

def _update_tracks(tracks, planet, star, i=0):
    tracks['Age'].append(planet.age)
    #tracks['Lbol'].append(star.Lbol(planet.age))
    tracks['Lbol'].append(star['Lbol'][i])
    #tracks['Lxuv'].append(star.Lx(planet.age)+star.Leuv(planet.age))
    tracks['Lxuv'].append(star['Lxuv'][i])
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

def _update_params(params, planet, star, i=0):
    fbol = star['Lbol'][i] / (planet.dist*Const.au.to('cm').value)**2 / (4*np.pi)
    params.update(planet.__dict__)
    params['fbol'] = fbol
    params['mass'] = planet.mp
    params['radius'] = planet.rp
    #params['Lxuv'] = star.Lx(planet.age) + star.Leuv(planet.age)
    #params['Lbol'] = star.Lbol(planet.age)
    params['Lxuv'] = star['Lxuv'][i]
    params['Lbol'] = star['Lbol'][i]
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





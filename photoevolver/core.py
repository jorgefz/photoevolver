"""

[photoevolver.core]

"""


class Evolve:
    def __init__(planet, mloss, struct, tracks, time_step=1.0, agemin=10, agemax=10000, params=None):
        """
        Evolves a planet forward and backwards.
        Parameters:
            planet:     (object) photoevolve.core.Planet object previously initialised
            mloss:      (class) Mass loss formulation from photoevolve.massloss
            struct:     (class) Planet structure formulation that relates the envelope radius to the envelope mass fraction from photoevolve.structure.
            tracks:     (object OR dict) Description of X-ray evolution of the host star. Two options:
                            Mors.Star object
                            Dictionary with keys 'Rx_sat' for the saturation activity, 't_sat' in Myr, and 'power' for the power-law index.
            params:     (dict) Specific parameters to tailor your simulation:
                            'eff': mass loss efficiency if using the EnergyLimited formulation.
                            'Rxuv': XUV radius of the planet if using the EnergyLimited formulation in R_earth.
                            'mstar': Mass of the host star in M_sun.
                            ''
            time_step:  (float) Time step of the simulation in Myr
            agemin:     (float) Earliest age to which to run simulation
            agemax:     (float) Oldest age to which to run simulation
        """
        if type(planet) != Planet:
            raise ValueError("the planet must be an instance of the photoevolve.core.Planet class")
        if (agemin > planet.age) or (agemax < planet.age):
            raise ValueError("The miminum and maximum ages mist me lower/greater than the defined planet age")
    
        # Calculating remaining planet parameters
        if (planet.fenv is None or planet.menv is None or planet.mp is None):
            pass
        else if (planet.renv is None or planet.rp is None):
            pass



    def step():
        


class Planet:
    def __init__(self, mp=None, rp=None, mcore=None, rcore=None, menv=None, renv=None, fenv=None, dist=None, age=None, comp=None):
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

        self._gen_mass = False
        self._gen_radius = False
        
        self.input_check()

    def input_check(self):
        if self.dist is None or self.age is None:
            raise ValueError("age or orbital distance undefined")
        
        if self.mcore is None or self.rcore is None:
            raise ValueError("core mass or radius undefined")
        
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
        
        if type(self.comp) != dict or \
           (type(self.comp) == dict and tuple(self.comp.keys()) != ['Xice','Xrock','Xiron']) or \
           sum( list(self.comp.values()) ) > 1.0:
                self.comp = {'Xice':0.0, 'Xrock':2/3, 'Xiron':1/3}
        
    def __repr__(self): 
        mcore_str = f"{self.mcore:.2f}" if self.mcore is not None else str(None)
        rcore_str = f"{self.rcore:.2f}" if self.rcore is not None else str(None)
        menv_str = f"{self.menv:.2f}" if self.menv is not None else str(None)
        renv_str = f"{self.renv:.2f}" if self.renv is not None else str(None)
        mp_str = f"{self.mp:.2f}" if self.mp is not None else str(None)
        rp_str = f"{self.rp:.2f}" if self.rp is not None else str(None)
        fenv_str = f"[{self.fenv*100:.2f}%]" if self.fenv is None else str(None)+'%'
        comp_str = f"[ice:{self.comp['Xice']*100:.0f}% rock:{self.comp['Xrock']*100:.0f}% iron:{self.comp['Xiron']*100:.0f}%]"

        msg =  f"<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}>\n\n" \
            + f"".ljust(10)         + f"Radius".ljust(10)   + f"Mass".ljust(10)               + "\n" \
            + f"Core".ljust(10)     + rcore_str.ljust(10)   + mcore_str.ljust(10)  + comp_str + "\n" \
            + f"Envelope".ljust(10) + renv_str.ljust(10)    + menv_str.ljust(10)   + fenv_str + "\n" \
            + f"Total".ljust(10)    + rp_str.ljust(10)      + mp_str                          + "\n"
        return msg

    def __str__(self):
        return self.__repr__()





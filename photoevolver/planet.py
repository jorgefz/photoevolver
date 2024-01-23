
import pandas as pd
import dataclasses as dataclass
import numpy as np
import typing
import copy
import abc
from astropy import units, constants
import uncertainties as uncert
from uncertainties import ufloat
from uncertainties import wrap as uwrap
from scipy.optimize import (
    least_squares as scipy_leastsq,
    fsolve as scipy_fsolve
)
import tqdm

from . import physics, utils
from .integrator import EulerIntegrator, RK45Integrator
from .evostate import EvoState, wrap_callback


def solve_core_model(
        core_model :typing.Callable[[EvoState,dict],float],
        state      :EvoState,
        guess      :float,
        model_kw   :dict = None
    ) -> list[float,bool]:
    """
    Given a model for the planet's core, which takes a mass to calculate
    its radius, it solves the model function to return the core mass
    from a given radius instead.
    For documentation, see `solve_envelope_model`.
    """
    def wrapper(x, *args, **kwargs):
        state, model_kw = kwargs['state'], kwargs['model_kw']
        core_model = kwargs['core_model']
        state.mcore = x[0]
        result_rcore = core_model(state, model_kw)
        assert ~np.isnan(result_rcore), "Failed to find solution"
        return result_rcore - state.rcore
    
    if model_kw is None: model_kw = dict()
    solution = scipy_leastsq(
        fun = wrapper,  x0 = guess, # bounds = [0.0, 100.0],
        kwargs = dict(state=state, model_kw=model_kw, core_model=core_model)
    )
    return solution.x[0], solution.success


def solve_envelope_model(
        env_model :typing.Callable[[EvoState,dict],float],
        state     :EvoState,
        guess     :float,
        model_kw  :dict = None
    ) -> list[float,bool]:
    """
    Given a model for the planet's envelope, which takes a mass fraction
    to calculate its thickness, it solves the model function
    to return the envelope mass fraction from a given thickness instead.
    It requires the core mass to be defined in the state.
    
    Parameters
    ----------
    env_model : Callable[[EvoState,dict],float]
        Function that takes the simulation state and extra arguments
        in a dict, and returns the envelope thickess.
    guess     : float
        Starting guess for the envelope mass fraction.
    state     : EvoState
        Simulation state - modified within the function.
    model_kw  : dict, optional
        Keyword parameters to pass to the envelope model.
    
    Returns
    -------
    solution : float
        Solved envelope mass fraction.
    success  : bool
        True if the solution converged, False otherwise.
    """
    def wrapper(x :list[float], *args, **kwargs):
        state, model_kw = kwargs['state'], kwargs['model_kw']
        env_model = kwargs['env_model']
        state.fenv = x[0]
        # state.mass = state.mcore / (1 - state.fenv)
        state.mass = state.mcore * (1 + state.fenv)
        result_renv = env_model(state, model_kw)
        return result_renv - state.renv

    if model_kw is None:
        model_kw = dict()
    solution = scipy_leastsq(
        fun = wrapper,  x0 = guess, bounds = [0.0, 1.0],
        kwargs = dict(state=state, model_kw=model_kw, env_model=env_model)
    )
    return solution.x[0], solution.success


def solve_planet_from_mass_radius(
        state      :EvoState,
        env_model  :typing.Callable[[EvoState,dict],float],
        core_model :typing.Callable[[EvoState,dict],float],
        fenv_guess :float = 0.01,
        model_kw   :dict  = None,
        errors     :bool  = False
        ) -> list[EvoState,bool]:
    """
    Solves for the structure of a planet given its mass and radius,
    as well as orbital and stellar parameters.
    Assumes the mass and radius are defined, but does not require
    core or envelope information.
    It can also accept variables with uncertainties and return planet parameters
    with calculated uncertainties as well, as long as the provided models
    can handle uncertainties.
    
    Parameters
    ----------
    state      : EvoState
        Simulation state - must have mass and radius defined.
        This instance is modified in the function.
    env_model  : Callable[[EvoState,dict], float]
        Function that calculates the envelope thickness
        from the simulation state. For documentation on
        signature, see `solve_envelope_model`.
    core_model : callable
        Function that calculates the core radius from the
        simulation state. For documentation on
        signature, see `solve_core_model`.
    fenv_guess : float, optional
        Initial guess for the envelope mass fraction.
    model_kw   : dict, optional
        Keyword arguments passed to envelope and core models.
    errors     : bool, optional
        If True, enables propagation of uncertainties.

    Returns
    -------
    state :EvoState
        Solved internal structure of the planet.
    success :bool
        Whether the solution converged.
    """

    def diff_fn(x :list[float], *args, **kwargs) -> float:
        """Returns the difference in envelope thicknesses
        calculated using the core and envelope models"""
        state, model_kw = kwargs['state'], kwargs['model_kw']
        core_model, env_model = kwargs['core_model'], kwargs['env_model']
        state.fenv = x[0]
        # Calculate envelope radius from core mass-radius relation
        state.mcore = state.mass / (1.0 + state.fenv)
        state.rcore = core_model(state, model_kw)
        renv1 = state.radius - state.rcore
        # Calculate envelope thickness from envelope structure model
        renv2 = env_model(state, model_kw)
        return renv1 - renv2
    
    # Objects on function parameters are initialised only once!
    if model_kw is None:
        model_kw = {}
    
    # Least squares model more accurate than fsolve
    solution = scipy_leastsq(
        fun = diff_fn, x0 = fenv_guess, bounds = [0.0, 1.0],
        kwargs = dict(
            state = state, model_kw = model_kw,
            core_model = core_model, env_model = env_model
        )
    )
    
    state.fenv = solution.x[0]
    state.mcore = state.mass / (1.0 + state.fenv)
    state.rcore = core_model(state, model_kw)
    state.renv = state.radius - state.rcore
    return state, solution.success



def solve_with_errors(
        planet     :'Planet',
        age        :float,
        fenv_guess :float = 0.01,
        model_kw   :dict  = None,
        error_kw   :dict  = None,
        ) -> list[EvoState,bool]:
    """
    Solves for the structure of a planet given its mass and radius,
    as well as orbital and stellar parameters,
    allowing the state to have quantities with uncertainties.
    Assumes the mass and radius are defined, but does not require
    core or envelope information.

    Parameters
    ----------
    planet     : Planet
        Planet instance with mass, radius, and separation defined,
        as well as core, envelope, and stellar models set.
    age        : float
        Age of the system in Myr.
    fenv_guess : float, optional
        Initial guess for the envelope mass fraction.
    model_kw   : dict, optional
        Keyword arguments passed to envelope and core models.
    error_kw  : dict, optional
        Keyword arguments to signal the core model to
        return quantities with uncertainties.

    Returns
    -------
    solved  :dict
        Solved internal structure of the planet.
    success :bool
        Whether the solution converged.
    """
    solver_success :bool = True
    
    @uwrap
    def usolve(**kwargs):
        state = EvoState.from_dict(kwargs)
        state.lx = planet.star_model['lx'](state, model_kw)
        state.leuv = planet.star_model['leuv'](state, model_kw)
        state.lbol = planet.star_model['lbol'](state, model_kw)
        state.mstar = planet.star_model['mass']

        solved_state, success = solve_planet_from_mass_radius(
            state      = state,
            env_model  = planet.core_model,
            core_model = planet.envelope_model
        )
        nonlocal solver_success
        solver_success |= success
        return solved_state.fenv

    if model_kw is None: model_kw = {}
    if error_kw is None: error_kw = {}
    
    state = planet.initial_state.copy()
    state.age    = age
    state.mstar  = planet.star_model['mass']
    if state.sep:
        state.period = uwrap(physics.keplers_third_law)(
            state.mstar, state.mass, sep = state.sep
        )
    else:
        state.sep = uwrap(physics.keplers_third_law)(
            state.mstar, state.mass, period = state.period
        )
    state.fenv   = usolve(**state.asdict())
    state.mcore  = state.mass / (1.0 + state.fenv)
    state.rcore  = planet.core_model(state, error_kw)
    state.renv   = state.radius - state.rcore
    return state.asdict(), solver_success



class Planet:

    ###################
    # Class Variables #
    ###################

    debug :bool = False
    """ If set to True, debug messages are printed """

    ###################
    # Dunder Methods  #
    ###################
    
    def __init__(self, **kwargs):
        """
        Initialises the starting conditions of a planet:
            mass    :float, planet mass in Earth masses
            radius  :float, planet radius in Earth radii
            mcore   :float, core mass in Earth masses
            rcore   :float, core radius in Earth radii
            fenv    :float, envelope mass fraction = (mass - mcore) / mcore
            period  :float, orbital period in days
            sep     :float, orbital separation in AU (assuming circular orbit)
        Only certain combinations of starting parameters are accepted:
            - mass and radius
            - mcore and fenv
            - rcore and fenv
        Either the period or separation must also be provided.
        Examples:
            ```
            planet = ph.Planet(mass = 5.0, radius = 2.5, sep = 1.0)
            planet = ph.Planet(mcore = 2.0, fenv = 0.01, sep = 1.0)
            planet = ph.Planet(rcore = 1.5, fenv = 0.02, period = 15.0)
            ```
        """
        
        # Parse starting planet parameters
        self.initial_state :EvoState = EvoState.from_dict(kwargs)
        Planet._debug_print(f"Created instance with parameters: {kwargs}")

        # Models
        self.star_model      :dict     = None   # :dict(mass :float, lx :callable, leuv :callable, lbol :callable)
        self.envelope_model  :callable = None   # :Callable[params:float] -> float
        self.mass_loss_model :callable = None   # :Callable[params:float] -> float
        self.core_model      :callable = None   # :Callable[params:float] -> float
        self.model_args      :dict     = None   # :dict(Any)
    
    ###################
    # Public Methods  #
    ###################

    def set_models(self,
            star_model      :dict|typing.Any             = None,
            envelope_model  :typing.Callable[..., float] = None,
            core_model      :typing.Callable[..., float] = None,
            mass_loss_model :typing.Callable[..., float] = None,
            model_args      :dict = None
        ) -> 'Planet':
        """
        Provides the model callbacks necessary to calculate the planet's internal structure
        and to simulate its evaporation history.
        If an input model is left unset (with None value),
        its value in the Planet instance will not be updated.

        Parameters
        ----------
        star    :dict | mors.Star, optional
            Host star parameters and emission history.
            Must be one of the following:
            - Dict with the following keys:
                mass    :float, star mass in solar masses
                lx      :callable(EvoState,dict) -> float,
                        function that takes state and model arguments,
                        and returns X-ray luminosity in erg/s.
                leuv    :callable(EvoState,dict) -> float,
                        function that takes state and model arguments,
                        and returns EUV luminosity in erg/s.
                lbol    :callable(EvoState,dict) -> float,
                        function that takes state and model arguments,
                        and returns bolometric luminosity in erg/s.
            
            - Mors.Star object,
                see https://github.com/ColinPhilipJohnstone/Mors
        
        envelope  :Callable(EvoState,dict) -> float, optional
            Envelope structure model. Calculates and returns the envelope thickness.

        core_model :Callable(EvoState,dict) -> float, optional
            Calculates and returns the core radius from its mass.

        mass_loss :Callable(EvoState,dict) -> float, optional
            Calculates and returns the mass loss rate in grams/sec.
            Default is function that returns a mass loss of zero.

        model_args :dict, optional
            Additional parameters passed to the models above.
        """
        self.model_args = {} if model_args is None else model_args
        
        if star_model is not None:
            self.star_model = self._parse_star(star_model)
            Planet._debug_print(
                f"Using star model with {self.star_model['mass']:.2f} solar masses",
            )
        
        if envelope_model is not None:
            assert callable(envelope_model), "Envelope model must be a function"
            self.envelope_model  = wrap_callback(envelope_model)
            if hasattr(envelope_model, "__name__"):
                Planet._debug_print(f"Using envelope model {envelope_model.__name__}")

        if core_model is not None:
            assert callable(core_model), "Core model must be a function"
            self.core_model  = wrap_callback(core_model)
            if hasattr(core_model, "__name__"):
                Planet._debug_print(f"Using core model {core_model.__name__}")

        if mass_loss_model is not None:
            assert callable(mass_loss_model), "Mass loss model must be a function"
            self.mass_loss_model  = wrap_callback(mass_loss_model)
            if hasattr(mass_loss_model, "__name__"):
                Planet._debug_print(f"Using mass loss model {mass_loss_model.__name__}")

        return self

    def use_models(self, other: 'Planet') -> 'Planet':
        """ Copies planet and stellar models from another Planet instance"""
        self.star_model      = other.star_model
        self.envelope_model  = other.envelope_model
        self.mass_loss_model = other.mass_loss_model
        self.core_model      = other.core_model
        self.model_args      = other.model_args
        return self
    
    def solve_structure(
            self,
            age    :float,
            errors :bool = False
        ) -> EvoState:
        """
        Solves for the internal structure of the planet
        assuming a core and envelope as distinct layers,
        using either the planet's mass and radius,
        or the core (mass or radius) and the envelope mass.
        """
        # TODO: Should work with ufloats
        assert self.star_model and self.envelope_model and self.core_model, \
            "Models have not been set!"

        state       = self.initial_state.copy()
        state.mstar = self.star_model['mass']
        state.age   = age
        state.lx    = self.star_model['lx'](state,self.model_args)
        state.leuv  = self.star_model['leuv'](state,self.model_args)
        state.lbol  = self.star_model['lbol'](state,self.model_args)
        
        # Calculate period or separation
        if state.period:
            assert state.period > 0.0, "Period must be greater than zero"
            state.sep = physics.keplers_third_law(
                state.mstar, period = state.period)
        elif state.sep:
            assert state.sep > 0.0, "Semi-major axis must be over zero"
            state.period = physics.keplers_third_law(
                state.mstar, sep = state.sep)
        else:
            raise ValueError("Specify either orbital period or separation")

        Planet._debug_print(
            f"Orbital parameters:",
            f"period={state.period:.2f} days and",
            f"separation={state.sep:.4f} AU"
        )

        # Mass and radius scenario
        if state.mass and state.radius:
            assert state.mass   > 0.0, "Planet mass must be over zero"
            assert state.radius > 0.0, "Planet radius must be over zero"
            self._solve_from_mass_radius(state, errors=errors)
            return state
        
        # Core and fenv scenario
        elif (state.mcore or state.rcore) and state.fenv:
            # Calculate core mass from core radius, or viceversa
            if state.mcore:
                assert state.mcore > 0.0, "Core mass must be over zero"
                state.rcore = self.core_model(state, self.model_args)
            elif state.rcore:
                assert state.rcore > 0.0, "Core radius must be over zero"
                guess = state.rcore**3
                state.mcore, success = solve_core_model(
                    self.core_model, state, guess, self.model_args)
                assert success, "Failed to solve for the core mass"
            assert state.fenv > 0.0, "Envelope mass must be over zero"
            state = self._solve_from_core_fenv(state)
            return state
        
        else:
            raise ValueError(
                "Either mass and radius,"
                "or core and envelope mass must be provided"
            )
    
    def evolve(
            self,
            start  :float,
            end    :float,
            step   :float = 0.01,
            method :str   = None,
            progressbar : bool = False,
            **kwargs
        ) -> pd.DataFrame:
        """
        Evolves the state of the planet in time.
        The planet parameters and models must have been set.

        Parameters
        ----------
        start, end  : float
            Initial and final age bounds for the simulation in Myr.
        method      : str | IntegratorBase, optional
            Integration method: "euler", or "rk45".
            'euler' : forward Euler method with a fixed step size.
            'rk45'  : 4th order Runge-Kutta method.
        step        : float
            Step size for the integration.
            For the euler method, thi is the fixed step size.
            For the RK45 method, this is the initial step size.
        progressbar : bool
            Displays a progress bar of the simulation using the `tqdm` module.

        Returns
        -------
        pandas.DataFrame
            Dataframe with a column for each simulation parameter,
            and a row for the values at each simulation step.
        """
        assert start > 0.0 and end > 0.0, "Invalid initial or final ages"

        # Default evaporation model always returns a mass loss rate of zero
        if self.mass_loss_model is None:
            self.mass_loss_model = lambda *args, **kwargs: 0.0
            self.mass_loss_model.__name__ = "zero_mass_loss"
        
        # Solve state at starting age
        self.initial_state = self.solve_structure(age=start)
        state = self.initial_state.copy()
        state.tstep = step # initial step size

        evo_states :list[EvoState] = []
        evo_states.append(state.copy())
        env_limit = 1 + 1e-5

        Planet._debug_print(
            f"Integrating from t={start:.2f} to t={end:.2f} Myr",
            f"with method '{method}'"
        )

        # Choose integration method
        integration_methods = {
            'euler': [EulerIntegrator, {'step_size': step}],
            'rk45' : [RK45Integrator,  {'first_step': step, 'max_step':step}],
        }
        
        if method is None:
            method = "euler"
        assert method in integration_methods, \
              f"Unknown integration method. " \
            + f"Available methods: {list(integration_methods.keys())}"
        integ_cls, integ_args = integration_methods[method]

        # Run integration
        step_fn = lambda t,s,**kw: self._integration_step(t,s,**kw)
        integrator :IntegratorBase = integ_cls(
            func     = step_fn,
            y_start  = state,
            t_start  = start,
            t_end    = end,
            progress = progressbar,
            func_kw  = {'env_limit': env_limit},
            **integ_args
        )
        
        while(integrator.running()):
            state = integrator.step()
            evo_states.append(state.copy())

        Planet._debug_print(f"Finished integration")
        df = pd.DataFrame([state.asdict() for state in evo_states])
        return df


    ###################
    # Private Methods #
    ###################

    def _parse_star(self, star :dict|typing.Any) -> dict:
        """
        Collects stellar parameters and evolution tracks
        from input arguments.
        """
        if isinstance(star, dict):
            # Stellar tracks from functions
            if set(star) == set(['mass','lx','leuv','lbol']):
                valid_format = all([
                    callable(star['lx']),   callable(star['leuv']),
                    callable(star['lbol']), star['mass'] > 0.0
                ])
                assert valid_format, "Invalid format for stellar model"
                return star
            # Stellar tracks from arrays
            raise ValueError("Invalid format for stellar model")
        
        elif utils.is_mors_star(star):
            return dict(
                mass = star.Mstar,
                lx   = lambda state,kw: star.Lx(state.age),
                leuv = lambda state,kw: star.Leuv(state.age),
                lbol = lambda state,kw: star.Lbol(state.age),
            )
        
        raise ValueError("Invalid format for stellar model")
    
    def _solve_from_mass_radius(
            self,
            state  :EvoState,
            errors :bool=False
        ) -> EvoState:
        """ Internal helper to solve the planet's structure from its
        measured mass and radius alone """
        # Assume mass and radius are provided
        state, success = solve_planet_from_mass_radius(
            state      = state,
            env_model  = self.envelope_model,
            core_model = self.core_model,
            fenv_guess = 0.01,
            model_kw   = self.model_args,
            errors     = errors
        )
        Planet._debug_print(f"Mass-radius solver converged... {success}")
        return state
    
    def _solve_from_core_fenv(self, state :EvoState) -> EvoState:
        """
        Uses the envelope model to calculate the envelope thickness
        from the core mass and radius, and the envelope mass fraction.
        Parameters:
            state   :EvoState, simulation state
        """
        state.mass = state.mcore * (1 + state.fenv)
        state.renv = self.envelope_model(state, self.model_args)
        state.radius = state.rcore + state.renv
        return state

    def _integration_step(
            self,
            age       :float,
            old_state :EvoState,
            **kw
        ) -> EvoState:
        """
        Computes a planet's state at a given age, based on a previous state.
        """
        kw['env_limit'] = kw.get('env_limit', 1+1e-5)
        state = old_state.copy()
        direction = 1 if (age > state.age) else -1
        state.age = age

        # Update X-ray and bolometric luminosities
        state.lx   = self.star_model['lx'](state, self.model_args)
        state.leuv = self.star_model['leuv'](state, self.model_args)
        state.lbol = self.star_model['lbol'](state, self.model_args)

        # Calculate mass loss rate
        mloss_conv = (units.g / units.s).to("M_earth/Myr")
        mloss = mloss_conv * self.mass_loss_model(state, self.model_args)

        # Update envelope mass
        new_mass = state.mass - direction * mloss * state.tstep
        if(new_mass <= state.mcore * kw['env_limit']):
            state.radius = state.rcore
            state.mass = state.mcore
            return state
        state.mass = new_mass
        state.fenv = (state.mass - state.mcore) / state.mcore

        # Calculate envelope thickness
        state.renv = self.envelope_model(state, self.model_args)
        state.radius = state.rcore + state.renv

        return state

    ###################
    # Static Methods  #
    ###################

    @staticmethod
    def _debug_print(*args : list[str]) -> typing.NoReturn:
        """Prints message in debug mode"""
        if Planet.debug is True:
            print(f"[photoevolver.Planet] ", *args)



import pandas as pd
import dataclasses as dataclass
import numpy as np
import typing
import copy
import abc
from astropy import units, constants
import uncertainties as uncert
from scipy.optimize import (
    least_squares as scipy_leastsq,
    fsolve as scipy_fsolve
)
import tqdm

from . import physics, utils
from .integrator import *
from .evostate import EvoState


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
        env_model  :callable,
        core_model :callable,
        fenv_guess :float = 0.01,
        model_kw   :dict = None,
        errors     :bool = False
        ) -> list[EvoState,bool]:
    """
    Solves for the structure of a planet given its mass and radius,
    as well as orbital and stellar parameters.
    Assumes the mass and radius are defined, but does not require
    core or envelope information.
    It can also accept variables with uncertainties and return planet parameters
    with calculated uncertainties as well.
    
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
        Specifies whether the solution converged.

    """

    def diff_fn(x :list[float], *args, **kwargs) -> float:
        """Returns the difference in envelope thicknesses
        predicted by the core and envelope models"""
        state, model_kw = kwargs['state'], kwargs['model_kw']
        core_model, env_model = kwargs['core_model'], kwargs['env_model']
        state.fenv = x[0]
        # Calculate envelope radius from mass-radius relation
        # state.mcore = (1.0 - state.fenv) * state.mass
        state.mcore = state.mass / (1.0 + state.fenv)
        state.rcore = core_model(state, model_kw)
        renv1 = state.radius - state.rcore
        # Calculate envelope radius from envelope structure formulation
        renv2 = env_model(state, model_kw)
        return renv1 - renv2
    
    if model_kw is None:
        model_kw = {}
    solution = scipy_leastsq(
        fun = diff_fn,  x0 = fenv_guess, bounds = [0.0, 1.0],
        kwargs = dict(
            state = state, model_kw = model_kw,
            core_model = core_model, env_model = env_model
        )
    )
    state.fenv = solution.x[0]
    # state.mcore = (1.0 - state.fenv) * state.mass
    state.mcore = state.mass / (1.0 + state.fenv)
    state.rcore = core_model(state, model_kw)
    state.renv = state.radius - state.rcore
    return state, solution.success


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
        self.model_args      :dict     = None
    
    ###################
    # Public Methods  #
    ###################

    def set_models(self,
            star_model      :dict|typing.Any,
            envelope_model  :typing.Callable[[EvoState,dict],float],
            mass_loss_model :typing.Callable[[EvoState,dict],float],
            core_model      :typing.Callable[[EvoState,dict],float],
            model_args      :dict = None
        ) -> typing.NoReturn:
        """
        Provides the models necessary to run the simulation:

        Parameters
        ----------
            star    :dict | mors.Star
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
        
        envelope  :Callable(EvoState,dict) -> float
            Envelope structure model. Calculates and returns the envelope thickness.

        mass_loss :Callable(EvoState,dict) -> float
            Calculates and returns the mass loss rate in grams/sec.

        core_mr   :Callable(EvoState,dict) -> float
            Calculates and returns the core radius from its mass.

        model_args :dict
            Additional parameters passed to the models above.
        """
        # Validate stellar model
        if model_args is None: model_args = dict()
        self.star_model = self._parse_star(star_model)
        Planet._debug_print(
            f"Using star model with {self.star_model['mass']} solar masses",
        )
        
        # Validate planet models
        models_valid = all([
            callable(envelope_model),
            callable(mass_loss_model),
            callable(core_model)
        ])
        assert models_valid, "Invalid model"
        self.envelope_model  = envelope_model
        self.mass_loss_model = mass_loss_model
        self.core_model      = core_model
        self.model_args      = model_args
        Planet._debug_print(
            "Using models",
            f"'{envelope_model.__name__}' for the envelope,",
            f"'{mass_loss_model.__name__}' for mass loss, and",
            f"'{core_model.__name__}' for the core",
        )
    
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
            else:
                raise ValueError("Specify either core mass or core radius")
            assert state.fenv > 0.0, "Envelope mass must be over zero"
            self._solve_from_core_fenv(state)
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
            method :str = "auto",
            step   :str = 0.01,
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
            Integration method: "linear", "RK45" or "auto".
        step        : float
            Step size for the integration, passed to the integrator.
        progressbar : bool
            Displays a progress bar of the simulation using the `tqdm` module.

        Returns
        -------
        pandas.DataFrame
            Dataframe with a column for each simulation parameter,
            and a row for the values at each simulation step.
        """
        assert start > 0.0 and end > 0.0, "Invalid initial or final ages"
        
        # Solve state at starting age
        self.initial_state = self.solve_structure(age=start)
        state = self.initial_state.copy()
        state.tstep = step # initial step size

        evo_states :list[EvoState] = []
        evo_states.append(state.copy())
        mloss_conv = (units.g / units.s).to("M_earth/Myr")
        env_limit_factor = (1 + 1e-5)

        def step_fn(age :float, prev_radius :float, **kw) -> float:
            # Update age, lx, leuv, lbol
            direction = 1 if (age > state.age) else -1
            state.age = age
            state.lx   = self.star_model['lx'](state, self.model_args)
            state.leuv = self.star_model['leuv'](state, self.model_args)
            state.lbol = self.star_model['lbol'](state, self.model_args)
            # Calculate mloss
            mloss = mloss_conv * self.mass_loss_model(state, self.model_args)
            # Update mass and fenv
            new_mass = state.mass - direction * mloss * state.tstep
            if(new_mass <= state.mcore * env_limit_factor):
                state.radius = state.rcore
                state.mass = state.mcore
                evo_states.append(state.copy())
                return 0.0
            state.mass = new_mass
            # state.fenv = (state.mass - state.mcore) / state.mass
            state.fenv = (state.mass - state.mcore) / state.mcore
            # Calculate renv
            state.renv = self.envelope_model(state, self.model_args)
            state.radius = state.rcore + state.renv
            # Save state
            evo_states.append(state.copy())
            drdt = (state.radius - prev_radius) / state.tstep
            return drdt

        Planet._debug_print(
            f"Integrating from t={start:.2f} to t={end:.2f} Myr",
            f"with method '{method}'"
        )
        
        assert method in ["auto","linear"], "Only Euler's method is implemented!"        
        integration_methods = dict(
            linear = [EulerIntegrator, dict(step_size=step)],
            auto   = [EulerIntegrator, dict(step_size=step)],
        )

        integ_cls, integ_args = integration_methods[method]

        integrator :IntegratorBase = integ_cls(
            fun = step_fn, y_start = state.radius,
            t_start = start, t_end = end,
            progress = progressbar, **integ_args
        )
        
        while(integrator.running()):
            integrator.step()
            state.tstep = integrator.step_size()

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
            raise NotImplementedError("Invalid format for stellar model")
        
        elif utils.is_mors_star(star):
            return dict(
                mass = star.Mstar,
                lx   = lambda state,kw: star.Lx(state.age),
                leuv = lambda state,kw: star.Leuv(state.age),
                lbol = lambda state,kw: star.Lbol(state.age),
            )
        
        raise NotImplementedError("Invalid format for stellar model")
    
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
        state.mass = state.mcore / (1 - state.fenv)
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


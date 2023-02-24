import pytest
import numpy as np

import Mors as mors
import photoevolver as ph
from photoevolver import models, Planet
from photoevolver.evostate import EvoState, wrap_callback
from uncertainties import ufloat, wrap as uwrap
from scipy.optimize import least_squares as scipy_leastsq


def model_unwrapper(fn):
    """ model(state,kw) -> model(**kw) """
    def wrapper(**kw):
        state_keys = list(EvoState().asdict().keys())
        state_dict = {k:v for k,v in kw.items() if k in state_keys}
        state = EvoState.from_dict(state_dict)
        model_kw = {k:v for k,v in kw.items() if k not in state_keys}
        return fn(state, model_kw)
    return wrapper

def super_wrapper(fn):
    return wrap_callback(uwrap(model_unwrapper(fn)))

def _state_model_to_kw(state, model_kw):
    return state.asdict().update(model_kw)


"""
PROBLEMS

State has ucnertainties.
State is passed everywhere, including model functions.
Model functions cant accept uncertainties.
Solver can't accept uncertainties.
Uncertainties wrapper wont work if state already has ucnertainties.

SOLUTIONS
Make models accept arbitrary keywords.
Wrap models with ucnertainty wrappers.
Convert state to dict before passing to models.

"""


def _solve_planet_from_mass_radius(
        state      :EvoState,
        env_model  :callable,
        core_model :callable,
        fenv_guess :float = 0.01,
        model_kw   :dict = None,
        errors     :bool = False
        ) -> list[EvoState,bool]:
    
    def diff_fn(x :list[float], *args, **kwargs) -> float:
        """Returns the difference in envelope thicknesses
        predicted by the core and envelope models"""
        core_model, env_model = kwargs['core_model'], kwargs['env_model']
        kwargs['fenv'] = x[0]
        # Calculate envelope radius from mass-radius relation
        kwargs['mcore'] = state.mass / (1.0 + kwargs['fenv'])
        kwargs['rcore'] = core_model(**kwargs)
        renv1 = kwargs['radius'] - kwargs['rcore']
        # Calculate envelope radius from envelope structure formulation
        renv2 = env_model(**kwargs)
        return renv1 - renv2
    
    @uwrap
    def solver_fn(**kwargs):
        solution = scipy_leastsq(
            fun = diff_fn,  x0 = fenv_guess, bounds = [0.0, 2.0],
            kwargs = kwargs
        )
        return solution.x[0]

    if model_kw is None:
        model_kw = {}
    
    solution = solver_fn(**state.asdict(), **model_kw,
        env_model=env_model, core_model=core_model)
    state.fenv = solution
    state.mcore = state.mass / (1.0 + state.fenv)
    state.rcore = core_model(**state.asdict(), **model_kw, ot20_errors=True)
    state.renv = state.radius - state.rcore
    # state.renv = uwrap(env_model)(**state.asdict(), **model_kw)
    return state, True


def test_solve_planet_from_mass_radius():
    

    star = mors.Star(Mstar=1.0, percentile=50.0)
    planet = Planet(
        mass   = ufloat(5.0,1.0),
        radius = ufloat(2.0,0.1),
        sep = 0.1
    )
    
    planet.set_models(
        star_model = star,
        core_model      = model_unwrapper(models.core_otegi20),
        envelope_model  = model_unwrapper(models.envelope_chen16),
        mass_loss_model = model_unwrapper(models.massloss_energy_limited),
    )

    planet.initial_state.age = ufloat(1000, 100)
    planet.initial_state.lx   = planet.star_model['lx'](EvoState(age=1000), {})
    planet.initial_state.leuv = planet.star_model['leuv'](EvoState(age=1000), {})
    planet.initial_state.lbol = planet.star_model['lbol'](EvoState(age=1000), {})
    state, success = _solve_planet_from_mass_radius(
        state = planet.initial_state,
        model_kw = planet.model_args,
        env_model = planet.envelope_model,
        core_model = planet.core_model
    )

    print(success, state)
    assert False




def test_planet_solve_structure_errors():
    
    star = mors.Star(Mstar=1.0, percentile=50.0)

    planet = Planet(
        mass   = ufloat(5.0,1.0),
        radius = ufloat(2.0,0.1),
        period = 25.0
    )

    planet.set_models(
        star_model      = {
            'mass': star.Mstar,
            'lx'  : wrap_callback(uwrap(lambda **kw: star.Lx(   kw['age'] ))),
            'leuv': wrap_callback(uwrap(lambda **kw: star.Leuv( kw['age'] ))),
            'lbol': wrap_callback(uwrap(lambda **kw: star.Lbol( kw['age'] ))),
        },
        core_model      = super_wrapper(models.core_otegi20),
        envelope_model  = super_wrapper(models.envelope_chen16),
        mass_loss_model = super_wrapper(models.massloss_energy_limited)
    )

    # state = planet.solve_structure(age=ufloat(100,10), errors=True)
    # print(state)
    
    # star_model
    # keplers_third_law
    # solve_from_mass_radius


def test_planet_solve_structure():
    star = mors.Star(Mstar=1.0, percentile=50.0)
    planet = Planet(
        mass   = 5.0,
        radius = 2.0,
        period = 25.0
    )
    planet.set_models(
        star_model      = star,
        core_model      = models.core_otegi20,
        envelope_model  = models.envelope_chen16,
        mass_loss_model = models.massloss_energy_limited
    )
    state = planet.solve_structure(age=100)
    print(state)
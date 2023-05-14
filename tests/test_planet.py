import pytest
import numpy as np

import Mors as mors
import photoevolver as ph
from photoevolver import models, Planet
from photoevolver.evostate import EvoState, wrap_callback
from uncertainties import ufloat, wrap as uwrap
from scipy.optimize import least_squares as scipy_leastsq

# Generate sun-like stellar model once
STAR = mors.Star(Mstar=1.0, percentile=50.0)


"""

COVERAGE
photoevolver/planet.py 75%
    432-433
    441
    447
    486-533
    547-555
    565
    602-629
    639

PROBLEMS

State has uncertainties.
State is passed everywhere, including model functions.
Model functions cant accept uncertainties.
Solver can't accept uncertainties.
Uncertainties wrapper wont work if state already has ucnertainties.

SOLUTIONS
Make models accept arbitrary keywords - DONE
Wrap models with uncertainty wrappers.
Convert state to dict before passing to models.

"""


def test_solve_core_model():
    rcore = 1.0
    guess = 1.0
    mcore, success = ph.planet.solve_core_model(
        core_model = wrap_callback(models.core_otegi20),
        state = EvoState(rcore = rcore),
        guess = guess
    )
    assert success, "Core model did not converge"
    # Re-run core model with solution to ensure input and output are consistent
    assert np.isclose(rcore, models.core_otegi20(mcore=mcore)), \
        "Model output and model solution are not consistent"


def test_solve_envelope_model():
    # Chose envelope that roughly doubles planet size
    guess = 1.0
    state = EvoState(mcore = 1.0, renv = 1.0, lbol = 1e33, sep = 0.1, age = 100)
    fenv, success = ph.planet.solve_envelope_model(
        env_model = wrap_callback(models.envelope_chen16),
        state = state,
        guess = guess
    )
    assert success, "Envelope model failed to converge"
    # Re-run core model with solution to ensure input and output are consistent
    assert np.isclose(state.renv, models.envelope_chen16(
            mass=state.mass, fenv=fenv, lbol=state.lbol, sep=state.sep, age=state.age
        )), "Model output and model solution are not consistent"


def test_solve_planet_from_mass_radius():
    state = EvoState(mass=5.0, radius=2.0, sep=0.1, lbol=1e33, age=100)
    solved, success = ph.planet.solve_planet_from_mass_radius(
        state = state,
        env_model  = wrap_callback(models.envelope_chen16),
        core_model = wrap_callback(models.core_otegi20)
    )

    assert success, "Models did not converge to a solution"
    assert solved.mass == 5.0 and solved.radius == 2.0 and solved.sep == 0.1,\
        "Preset parameters changed"
    assert np.isclose(solved.radius, solved.rcore + solved.renv), \
        "Core and envelope sizes are not consistent with given radius"
    assert np.isclose(solved.mass, solved.mcore * (1 + solved.fenv)), \
        "Core and envelope masses are not consistent with given mass"


def test_solve_planet_from_mass_radius_uncert():

    ustate = EvoState(
        mass   = ufloat(5.0, 0.5),
        radius = ufloat(2.0, 0.1),
        sep    = 0.1,
        lbol   = 1e33,
        age    = ufloat(100.0, 5.0)
    )

    solved, success = ph.planet.solve_planet_from_mass_radius_uncert(
        state = ustate,
        env_model  = wrap_callback(models.envelope_chen16),
        core_model = wrap_callback(models.core_otegi20),
        error_kw = {'ot20_errors': True},
    )
    assert success, "Models did not converge to a solution"
    assert solved.mass    == ustate.mass   \
        and solved.radius == ustate.radius \
        and solved.sep    == ustate.sep,   \
        "Preset parameters changed"
    assert np.isclose(solved.radius.n, solved.rcore.n + solved.renv.n), \
        "Core and envelope sizes are not consistent with given radius"
    assert np.isclose(solved.mass.n, solved.mcore.n * (1 + solved.fenv.n)), \
        "Core and envelope masses are not consistent with given mass"


def test_planet_init():

    mass = 5.0
    radius = 2.0
    planet = Planet(mass = mass, radius = radius)

    assert planet.initial_state.mass == mass, \
        "Mass parameter no saved on planet state"
    assert planet.initial_state.radius == radius, \
        "Radius parameter no saved on planet state"


def test_planet_set_models():
    planet = Planet(mass = 5.0, radius = 2.0)
    planet.set_models(
        star_model     = STAR,
        envelope_model = models.envelope_chen16,
        core_model     = models.core_otegi20,
        mass_loss_model = models.massloss_energy_limited,
        model_args = None
    )

    assert planet.envelope_model and planet.core_model, \
        "Models not set on planet"
    assert planet.star_model and planet.mass_loss_model, \
        "Models not set on planet"
    assert planet.model_args == {}, \
        "Model args not set to empty dict by default"

    # Check mass loss model defaults to function returning zero
    planet.set_models(
        star_model     = STAR,
        envelope_model = models.envelope_chen16,
        core_model     = models.core_otegi20,
        mass_loss_model = None,
    )

    assert planet.mass_loss_model, \
        "Default mass loss model not set"
    assert planet.mass_loss_model(EvoState(), {}) == 0.0, \
        "Default mass loss model does not return zero"


def test_planet_use_models():
    planet = Planet(mass = 5.0, radius = 2.0)
    planet.set_models(
        star_model     = STAR,
        envelope_model = models.envelope_chen16,
        core_model     = models.core_otegi20,
        mass_loss_model = models.massloss_energy_limited,
    )

    planet2 = Planet(mass = 10.0, radius = 4.0)
    planet2.use_models(planet)

    assert planet2.star_model == planet.star_model \
        and planet2.core_model == planet.core_model \
        and planet2.envelope_model == planet.envelope_model \
        and planet2.mass_loss_model == planet.mass_loss_model \
        and planet2.model_args == planet.model_args, \
        "Models not copied over"

def test_planet_solve_structure():
    # Solve with mass and radius
    planet = Planet(mass = 10.0, radius = 4.0, sep = 0.1)
    planet.set_models(
        star_model     = STAR,
        envelope_model = models.envelope_chen16,
        core_model     = models.core_otegi20,
    )
    solved = planet.solve_structure(age = 100)
    assert solved.mass == planet.initial_state.mass \
        and solved.radius == planet.initial_state.radius, \
        "Original parameters have changed on the solution"
    assert np.isclose(solved.mass, solved.mcore * (1 + solved.fenv)), \
        "Mass in solution not consistent with core and envelope masses"
    assert np.isclose(solved.radius, solved.rcore + solved.renv), \
        "Radius in solution not consistent with core and envelope sizes"

    # Solve with rcore and fenv
    planet = Planet(rcore = 1.5, fenv = 0.01, sep = 0.1)
    planet.set_models(
        star_model     = STAR,
        envelope_model = models.envelope_chen16,
        core_model     = models.core_otegi20,
    )
    solved = planet.solve_structure(age = 100)

    print(solved.mass, solved.mcore * (1 + solved.fenv))

    assert solved.rcore == planet.initial_state.rcore \
        and solved.fenv == planet.initial_state.fenv, \
        "Original parameters have changed on the solution"
    assert np.isclose(solved.mass, solved.mcore * (1 + solved.fenv)), \
        "Mass in solution not consistent with core and envelope masses"
    assert np.isclose(solved.radius, solved.rcore + solved.renv), \
        "Radius in solution not consistent with core and envelope sizes"

    # Solve with mcore and fenv
    planet = Planet(mcore = 4.0, fenv = 0.01, sep = 0.1)
    planet.set_models(
        star_model     = STAR,
        envelope_model = models.envelope_chen16,
        core_model     = models.core_otegi20,
    )
    solved = planet.solve_structure(age = 100)

    print(solved.mass, solved.mcore * (1 + solved.fenv))

    assert solved.mcore == planet.initial_state.mcore \
        and solved.fenv == planet.initial_state.fenv, \
        "Original parameters have changed on the solution"
    assert np.isclose(solved.mass, solved.mcore * (1 + solved.fenv)), \
        "Mass in solution not consistent with core and envelope masses"
    assert np.isclose(solved.radius, solved.rcore + solved.renv), \
        "Radius in solution not consistent with core and envelope sizes"

    # Calculate separation from period
    planet2 = Planet(mass = 10.0, radius = 4.0, period = 5.0)
    planet2.use_models(planet)
    solved = planet2.solve_structure(age = 100)
    assert np.isclose(solved.sep, ph.physics.keplers_third_law(
        big_mass=STAR.Mstar, period=5.0)), \
        "Separation not consistent with period"

    # Error when neither sep nor period given
    planet2 = Planet(mass = 10.0, radius = 4.0)
    planet2.use_models(planet)
    with pytest.raises(ValueError):
        planet2.solve_structure(age = 100)

    # Error when not enough structure parameters given
    planet2 = Planet(sep = 0.1)
    planet2.use_models(planet)
    with pytest.raises(ValueError):
        planet2.solve_structure(age = 100)
    

def test_planet_parse_star():
    # Parse dict with functions and star mass
    planet = Planet(mass = 10.0, radius = 4.0, sep = 0.1)
    stellar_model = dict(
        mass = 1.0,
        lx   = lambda _:0.0,
        leuv = lambda _:0.0,
        lbol = lambda _:0.0
    )
    parsed_model = planet._parse_star(stellar_model)
    assert parsed_model == stellar_model

    # Parse invalid model
    stellar_model = dict(
        mass = 1.0,
        # lx   = lambda _:0.0,
        leuv = lambda _:0.0,
        lbol = lambda _:0.0
    )
    with pytest.raises(ValueError):
        parsed_model = planet._parse_star(stellar_model)
    
    with pytest.raises(ValueError):
        parsed_model = planet._parse_star(int())
    
    # Parse Mors object
    parsed_model = planet._parse_star(STAR)
    assert isinstance(parsed_model, dict)
    assert set(parsed_model) == set(['lx', 'leuv', 'lbol', 'mass'])
    assert parsed_model['mass'] == STAR.Mstar
    assert callable(parsed_model['lx']) \
        and callable(parsed_model['leuv']) \
        and callable(parsed_model['leuv'])
    assert parsed_model['lx'](EvoState(age=100),{}) == STAR.Lx(100)
    assert parsed_model['leuv'](EvoState(age=100),{}) == STAR.Leuv(100)
    assert parsed_model['lbol'](EvoState(age=100),{}) == STAR.Lbol(100)


def test_planet_integration_step():
    pass

def test_planet_evolve():
    pass

def test_planet_debug_print(capfd):
    # Capture debug message
    msg = "test"
    header = "[photoevolver.Planet]  "
    Planet.debug = True
    Planet._debug_print(msg)
    out, err = capfd.readouterr()
    print(out)
    assert out == header + msg + "\n"
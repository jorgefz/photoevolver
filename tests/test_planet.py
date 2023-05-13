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
photoevolver/planet.py 50%
    297-301
    315-369
    408-455
    477-487
    497-506
    515-518
    524-551
    561

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

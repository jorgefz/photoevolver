import pytest
import numpy as np
import uncertainties as uncert

from photoevolver import models
from photoevolver.evostate import EvoState

"""
Coverage: 52%

284-292
311-321
351-354
368-381
411-452
480-499
"""


def test_core_otegi20():
    
    result = models.core_otegi20(mcore = 1.0)
    assert np.isclose(result, 1.0, rtol=0.03)

    bad_states = {}, {'mcore':np.nan}, {'mcore':-1}
    for b in bad_states:
        with pytest.raises((ValueError, TypeError)):
            models.core_otegi20(**b)
    
    uncert_state = {'mcore': uncert.ufloat(1.0,0.1), 'ot20_errors':True}
    result_err   = models.core_otegi20(**uncert_state)
    assert result == result_err.nominal_value


def test_envelope_models():
    env_models = [
        models.envelope_lopez14,
        models.envelope_chen16,
        # models.envelope_owen17
    ]
    params = {'mass':5.0, 'fenv':0.01, 'lbol':1e33, 'sep':0.1, 'age':100}
    for model in env_models:
        result = model(**params)
        assert isinstance(result, float) \
            and np.isfinite(result) \
            and result > 0.0

    # Test unique features of models
    
    # --> lopez14 - opaque atmosphere
    result = models.envelope_lopez14(**params, lf14_opaque=True)
    assert isinstance(result, float) \
        and np.isfinite(result) \
        and result > 0.0

    # --> chen16 - high water content core
    result = models.envelope_chen16(**params, cr16_water=True)
    assert isinstance(result, float) \
        and np.isfinite(result) \
        and result > 0.0

    # --> owen17 - not implemented!
    with pytest.raises(NotImplementedError):
        result = models.envelope_owen17(**params)


def test_salz16_models():
    params = {'mass':1.0, 'radius':1.0, 'lx':1e29, 'leuv':1e29, 'sep':0.1}
    result = models.rxuv_salz16(**params)
    assert isinstance(result, float) \
        and np.isfinite(result) \
        and result > 0.0

    result = models.efficiency_salz16(mass=1.0, radius=1.0)
    assert isinstance(result, float) \
        and np.isfinite(result) \
        and result > 0.0


def test_massloss_models():
    """
    def massloss_salz16(
        lx     :float,
        leuv   :float,
        mstar  :float,
        mass   :float,
        radius :float,
        sep    :float,
        **kwargs
    ) -> float:

    def massloss_kubyshkina18(
        mass   :float,
        radius :float,
        leuv   :float,
        lbol   :float,
        sep    :float,
        mstar  :float,
        **kwargs
    ) -> float:

    def massloss_kubyshkina18_approx(
        mass   :float,
        radius :float,
        lx     :float,
        leuv   :float,
        lbol   :float,
        sep    :float,
        **kwargs
    ) -> float:
    """
    params = {
        'mass': 5.0, 'radius': 2.0, 'mstar': 1.0,
        'sep': 0.1, 'lx': 1e29, 'leuv': 1e29, 'lbol': 1e33
    }

    mloss_models = [
        models.massloss_salz16,
        models.massloss_kubyshkina18,
        models.massloss_kubyshkina18_approx
    ]

    for model in mloss_models:
        result = model(**params)
        assert isinstance(result, float) \
            and np.isfinite(result) \
            and result > 0.0


def test_euv_king18_model():
    """
    def euv_king18(
        lx     : float,
        rstar  : float,
        energy : str = "0.1-2.4"
    ):
    """
    result = models.euv_king18(lx=1e28, rstar=1.0)

    assert isinstance(result, float) \
        and np.isfinite(result) \
        and result > 0.0

    with pytest.raises(ValueError):
        models.euv_king18(lx=1e30, rstar=1.0, energy="invalid")


    
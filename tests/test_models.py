import pytest
import numpy as np
import uncertainties as uncert

from photoevolver import models
from photoevolver.evostate import EvoState


def test_core_otegi20():
    
    valid_state = EvoState(mcore = 1.0)
    result = models.core_otegi20(valid_state, dict())
    assert np.isclose(result, 1.0, rtol=0.03)

    bad_states = EvoState(), EvoState(mcore=np.nan), EvoState(mcore=-1)
    for b in bad_states:
        with pytest.raises(ValueError):
            models.core_otegi20(b, dict())
    
    uncert_state = EvoState(mcore = uncert.ufloat(1.0,0.1))
    result_err   = models.core_otegi20(uncert_state, dict(ot20_errors=True))
    assert result == result_err.nominal_value

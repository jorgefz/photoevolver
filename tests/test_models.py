import pytest
import numpy as np
import uncertainties as uncert

from photoevolver import models
from photoevolver.evostate import EvoState


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

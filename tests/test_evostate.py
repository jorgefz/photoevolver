import numpy as np
import pytest

from photoevolver.evostate import EvoState, wrap_callback

def test_evostate():
    some_params = dict(mass=1.0, radius=1.0)
    assert EvoState(**some_params)

    wrong_params = dict(mass=1.0, mock=1.0)
    with pytest.raises(Exception):
        EvoState(**wrong_params)

def test_evostate_asdict():
    state_dict = EvoState().asdict()
    assert isinstance(state_dict, dict) and len(state_dict)==14
    assert not all(state_dict.values()) # all None

def test_evostate_fromdict():
    keys = list(EvoState().asdict())
    expected = {k:1.0 for k in keys}
    state = EvoState.from_dict(expected)
    assert state.asdict() == expected

def test_evostate_copy():
    state_1 = EvoState()
    state_2 = state_1.copy()
    assert state_1.asdict() == state_2.asdict()
    assert state_1 is not state_2

def test_evostate_validate():
    state = EvoState(mass = 1.0)
    assert not state.validate()

    keys = list(EvoState().asdict())
    filled_args = {k:1.0 for k in keys}
    state = EvoState(**filled_args)
    assert state.validate()

def test_wrap_callback():
    state_args = dict(mass=1.0, radius=1.0)
    model_args = dict(mock=True)
    fn = lambda **kw: 1.0
    result_fn = wrap_callback(fn)

    with pytest.raises(TypeError):
        result_fn(**state_args, **model_args)
    
    state = EvoState(**state_args)
    assert result_fn(state, model_args) == 1.0
    assert result_fn.__name__ == fn.__name__
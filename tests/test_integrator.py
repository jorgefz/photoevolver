import numpy as np
import pytest
from photoevolver.integrator import IntegratorBase, EulerIntegrator

def test_integrator_base():
    fun = lambda t,y,**kw: -y
    solution_y = lambda t: np.exp(-t)
    args = dict(fun=fun, t_start=0.0, t_end=1.0, y_start=1.0)

    # Instatiate abstract class
    with pytest.raises(TypeError):
        IntegratorBase(**args)

    # Instantiate derived class with no overried abstract methods
    class Deriv(IntegratorBase): pass
    with pytest.raises(TypeError):
        Deriv(**args)

    # Instantiate derived class with overriden abstract methods
    step_size = 0.1
    euler = EulerIntegrator(**args, step_size = step_size)

    # Check base members
    assert euler.fun == args['fun'] and euler.y_start == args['y_start']
    assert euler.t_start == args['t_start'] and euler.t_end == args['t_end']
    assert euler.direction == 1 and euler.do_pbar == False

    # Check derived members
    assert isinstance(euler.fun_kwargs,dict) and euler.y == euler.y_start 
    assert euler.dt == step_size and euler.age == euler.t_start
    assert not euler.dydt and euler.run and euler.step_size() == euler.dt

    # Progress bar
    euler = EulerIntegrator(**args, step_size = step_size, progress=False)
    assert not euler.do_pbar
    assert not euler.pbar
    assert not euler._init_pbar()
    euler = EulerIntegrator(**args, step_size = step_size, progress=True)
    assert euler._init_pbar()
    assert euler.pbar
    assert euler._update_pbar()
    euler._end_pbar()
    assert not euler.pbar
    assert not euler._update_pbar()

    # Integration step
    euler._init_pbar()
    y = euler.y
    euler.step()
    assert euler.dydt == -y
    assert euler.y == y + euler.dydt * euler.dt
    assert euler.age == euler.direction * euler.dt

    # Reach end of integration
    # One step already taken, nine left
    for i in range(9):
        assert euler.run and euler.running()
        euler.step()
    assert not euler.run and not euler.running()
    assert np.isclose(euler.age, args['t_end'])
    assert np.isclose(solution_y(1.0), euler.y, rtol=0.1)
    




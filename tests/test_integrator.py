import numpy as np
import pytest
from photoevolver.integrator import (
    IntegratorBase, EulerIntegrator, RK45Integrator
)
from photoevolver.evostate import EvoState



def test_base_integrator():
    # Instatiate abstract class
    func = lambda t,state: state.update({'radius': state.radius * (1 - state.tstep)})
    args = dict(func=func, t_start=0.0, t_end=1.0, y_start=EvoState(radius=1.0))
    with pytest.raises(TypeError):
        IntegratorBase(**args)

    # Instantiate derived class with no overried abstract methods
    class Deriv(IntegratorBase): pass
    with pytest.raises(TypeError):
        Deriv(**args)



def test_euler_integrator():
    # Use the state radius as a proxy to integrate the function y(x) where dy/dx = -y
    # Note that the function must return the next state after one time step
    func = lambda t,state: state.update({'radius': state.radius * (1 - state.tstep)})
    func_solution = lambda t: np.exp(-t)
    args = dict(func=func, t_start=0.0, t_end=1.0, y_start=EvoState(radius=1.0))

    # Instantiate derived class with overriden abstract methods
    step_size = 0.1
    euler = EulerIntegrator(**args, step_size = step_size)

    # Check base members
    assert euler.func == args['func'] and euler.y_start == args['y_start']
    assert euler.t_start == args['t_start'] and euler.t_end == args['t_end']
    assert euler.direction == 1 and euler.do_pbar == False

    # Check derived members
    assert isinstance(euler.func_kw,dict) and euler.state == euler.y_start 
    assert euler.dt == step_size and euler.t == euler.t_start
    assert euler.run and euler.step_size() == euler.dt

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
    euler._init_pbar()
    
    # Reach end of integration
    # One step already taken, nine left
    for i in range(10):
        assert euler.run and euler.running()
        euler.step()
    assert not euler.run and not euler.running()
    assert np.isclose(euler.t, args['t_end'])
    assert np.isclose(func_solution(1.0), euler.state.radius, rtol=0.1)
    

def test_rk45_intergator():
    # Use the state radius as a proxy to integrate the function y(x) where dy/dx = -y
    # Note that the function must return the next state after one time step
    func = lambda t,state: state.update({'radius': state.radius * (1 - state.tstep)}).copy()
    func_solution = lambda t: np.exp(-t)
    args = dict(func=func, t_start=0.0, t_end=1.0, y_start=EvoState(radius=1.0))

    step_size = 0.01
    rk45 = RK45Integrator(**args, first_step = step_size, max_step = step_size)

    # Check base members
    assert rk45.func == args['func'] and rk45.y_start == args['y_start']
    assert rk45.t_start == args['t_start'] and rk45.t_end == args['t_end']
    assert rk45.direction == 1 and rk45.do_pbar == False

    # Check derived members
    assert isinstance(rk45.func_kw,dict) and rk45.state == rk45.y_start 
    assert rk45.dt == step_size
    assert hasattr(rk45, "rk45")

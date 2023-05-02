

import abc
import typing
import tqdm
import numpy as np
from scipy.integrate import RK45 as ScipyRK45
from .evostate import EvoState

class IntegratorBase(abc.ABC):
    """ ODE solver base class.
    Integrates the evaporation history of a planet
    using different methods.
    Contains functionality to print a progress bar
    to measure the progress of the integration.

    Parameters
    ----------
    func    : callable[[float,EvoState], EvoState]
        Function to integrate.
        It should have signature func(t, state)->state,
        where
            t     :float,    the current age of the planet.
            state :EvoState, the planet state at the previous age.
        and returns the new state at time `t`.
    y_start : EvoState
        Initial state of the planet
    t_start, t_end : float
        Starting and ending values for the age.
    """
    def __init__(
            self,
            func     :typing.Callable[[float,EvoState], float],
            y_start  :list,
            t_start  :float,
            t_end    :float,
            progress :bool = False,
            func_kw  :dict = None,
            **kwargs
        ):
        """Initialises integrator"""
        self.func      :typing.Callable = func
        self.y_start   :EvoState = y_start
        self.t_start   :float    = t_start
        self.t_end     :float    = t_end
        self.direction :int  = 1 if t_start < t_end else -1
        self.do_pbar   :bool = progress
        self.pbar :tqdm.tqdm = None
        self.func_kw = func_kw if func_kw else dict()

    def _init_pbar(self) -> bool:
        """Initialises progress bar"""
        if not self.do_pbar: return False
        fmt = (r"{desc}: {percentage:.3f}%|{bar}| "
                + r"{n:.3f}/{total_fmt} [{elapsed}<{remaining}]")
        self.pbar = tqdm.tqdm(
            total = abs(self.t_start-self.t_end),
            bar_format = fmt
        )
        return True
    
    def _update_pbar(self) -> bool:
        """Advances progress bar by one time step"""
        if not self.pbar: return False
        self.pbar.update(self.step_size())
        return True

    def _end_pbar(self):
        """Destroy progress bar object"""
        if self.pbar:
            self.pbar.close()
            self.pbar = None

    @abc.abstractmethod
    def step(self):
        """Executes one integration step"""
        pass
    
    @abc.abstractmethod
    def running(self):
        """Returns True if the integration is running"""
        pass

    @abc.abstractmethod
    def step_size(self):
        """Returns the current step size of the integration"""
        pass


class EulerIntegrator(IntegratorBase):
    """ First order integrator that uses Euler's method and a fixed step size
    to solve the ODE """
    def __init__(
            self,
            step_size  :float,
            **base_kw
        ):
        """ Initialises linear fixed-step integrator"""
        super().__init__(**base_kw)
        self.state :list  = self.y_start
        self.dt    :float = step_size
        self.t     :float = self.t_start
        self.run   :bool  = True
        self._init_pbar()

    def step(self) -> EvoState:
        dt = self.step_size()
        self.state.tstep = dt
        self.state = self.func(self.t, self.state, **self.func_kw)
        self.t   += self.direction * dt
        if self.direction * (self.t - self.t_end + self.step_size()) > 0:
            self.run = False
            self._end_pbar()
            return self.state.copy()
        self._update_pbar()
        return self.state.copy()

    def step_size(self) -> float:
        return self.dt

    def running(self) -> bool:
        return self.run


class RK45Integrator(IntegratorBase):
    """ 4th order Runge-Kutta integrator """
    def __init__(
            self,
            first_step :float = 0.01,
            max_step   :float = 1.0,
            **base_kw
        ):
        """
        Initialises the RK45 integrator with an initial time step 'first_step'
        and maximum time step 'max_step', both in Myr.
        """
        super().__init__(**base_kw)
        self.state = self.y_start
        self.dt = first_step
        self.integ_params = [
            "mass", "radius", "fenv", "renv",
            "lx", "leuv", "lbol"
        ]

        y = [self.state.asdict()[k] for k in self.integ_params]
        self.rk45 = ScipyRK45(
            fun     = self._wrap_integrand(),
            y0      = y,
            t0      = self.t_start,
            t_bound = self.t_end,
            first_step = first_step,
            max_step   = max_step
        )
        self._init_pbar()

    def _wrap_integrand(self):
        """
        Wraps the function to integrate, which takes a planet state
        and advances it by one time step, into the format used
        by scipy.integrate.RK45, which expects a list of
        variables to integrate.
        """
        def wrapped(t :float, y :list) -> list:
            pstate = self.state.update(dict(zip(self.integ_params, y)))
            nstate  = self.func(t, pstate, **self.func_kw)
            # Calculate differentials (dy/dt)
            pvalues = [pstate.asdict()[k] for k in self.integ_params]
            nvalues = [nstate.asdict()[k] for k in self.integ_params]
            return (np.array(nvalues) - np.array(pvalues)) * self.direction
        
        return wrapped

    def step(self) -> EvoState:
        """ Advances the integration by one step and returns the new state"""
        result = self.rk45.step()
        self.dt = self.step_size()
        self.state.tstep = self.dt
        self.state.age = self.rk45.t
        self.state = self.state.update(dict(zip(self.integ_params, self.rk45.y)))
        
        # if result:
        #     raise ValueError(result)
        if not self.running():
            self._end_pbar()
            return self.state.copy()
        self._update_pbar()
        return self.state.copy()

    def step_size(self) -> float:
        return self.rk45.step_size

    def running(self) -> bool:
        return self.rk45.status == "running"

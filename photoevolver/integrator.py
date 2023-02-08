

import abc
import typing
import tqdm
from scipy.integrate import RK45 as scipy_RK45

class IntegratorBase(abc.ABC):
    """ ODE solver base class.
    Contains functionality to run a progressbar
    to measure the progress of the integration.
    """
    def __init__(
            self,
            fun      :typing.Callable[[float,float,...],float],
            y_start  :float,
            t_start  :float,
            t_end    :float,
            progress :bool = False,
            **kwargs
        ):
        """Initialises integrator"""
        self.fun     = fun
        self.y_start = y_start
        self.t_start = t_start
        self.t_end   = t_end
        self.direction = 1 if t_start < t_end else -1
        self.do_pbar = progress
        self.pbar :tqdm.tqdm = None
    
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
            fun_kwargs :dict = None,
            **base_kwargs
        ):
        """ Initialises linear fixed-step integrator"""
        super().__init__(**base_kwargs)
        self.fun_kwargs = fun_kwargs
        if self.fun_kwargs is None:
            self.fun_kwargs = dict()
        self.y    :float = self.y_start
        self.dt   :float = step_size
        self.age  :float = self.t_start
        self.dydt :float = None
        self.run  :bool = True
        self._init_pbar()

    def step(self) -> typing.NoReturn:
        dt = self.step_size()
        self.dydt = self.fun(self.age, self.y, **self.fun_kwargs)
        self.y   += self.dydt * dt
        self.age += self.direction * dt
        if self.direction * (self.age - self.t_end + self.step_size()) > 0:
            self.run = False
            self._end_pbar()
            return
        self._update_pbar()

    def step_size(self) -> float:
        return self.dt

    def running(self) -> bool:
        return self.run



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
    
    def _init_pbar(self):
        """Initialises progress bar"""
        if not self.do_pbar: return
        fmt = (r"{desc}: {percentage:.3f}%|{bar}| "
                + r"{n:.3f}/{total_fmt} [{elapsed}<{remaining}]")
        self.pbar = tqdm.tqdm(
            total = abs(self.t_start-self.t_end),
            bar_format = fmt
        )
    
    def _update_pbar(self):
        """Advances progress bar by one time step"""
        if not self.pbar: return
        self.pbar.update(self.step_size())

    def _end_pbar(self):
        """Destroy progress bar object"""
        if not self.pbar: return
        self.pbar.close()

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
        if self.direction * (self.age - self.t_end) > 0:
            self.run = False
            self._end_pbar()
            return
        self._update_pbar()

    def step_size(self) -> float:
        return self.dt

    def running(self) -> bool:
        return self.run



class AdaptiveIntegrator(IntegratorBase):
    """ Second order integrator that uses Heun's method and
    and adaptive step size
    """
    def __init__(
            self,
            step_size  :float,
            fun_kwargs :dict = None,
            tol        :float = 0.01,
            **base_kwargs
        ):
        raise NotImplementedError("Adaptive integrator is not available")
        super().__init__(**base_kwargs)
        self.fun_kwargs = fun_kwargs
        if self.fun_kwargs is None:
            self.fun_kwargs = dict()
        self.y    :float = self.y_start
        self.dt   :float = step_size
        self.age  :float = self.t_start
        self.run  :bool  = True
        self.k1   :float = None
        self.k2   :float = None
        self.tol  :float = tol
        self._init_pbar()

    def step(self) -> typing.NoReturn:
        if self.k1 is None:
            self.k1 = self.fun(self.age, self.y, **self.fun_kwargs)
            return
        
        self.k2 = self.fun(
            self.age + self.dt * self.direction,
            self.y + self.dt * self.k1,
            **self.fun_kwargs
        )

        err = 0.5 * abs(self.k1 - self.k2)
        fac = self.tol / err
        if fac > 1.0:
            self.y += self.dt * self.k1
            self.k1 = self.k2
            self.age += self.direction * self.dt
        
        self.dt *= 0.9 * fac

        print(f"k1={self.k1}, k2={self.k2}, {err=}, {fac=}, {self.dt=}")

        if self.direction * (self.age - self.t_end) > 0:
            self.run = False
            self._end_pbar()
            return
        self._update_pbar()
    
    def step_size(self) -> float:
        return self.dt

    def running(self) -> bool:
        return self.run


class RK45Integrator(IntegratorBase):
    def __init__(self, max_step :float, **kw):
        super().__init__(**kw)
        self.handle = scipy_RK45(
            fun = self.fun, t0 = self.t_start, y0 = [self.y_start],
            t_bound = self.t_end, max_step = max_step
        )
        self.dt     :float = self.handle.step_size
        self.info   :str = None
        self.status :str = None
        self._init_pbar()

    def step(self):
        self.info = self.handle.step()
        self.dt = self.handle.step_size
        self._update_pbar()
    
    def running(self) -> bool:
        self.status = self.handle.status
        run = (self.status == "running")
        if not run:
            self._end_pbar()
        return run

    def step_size(self) -> float:
        return self.dt


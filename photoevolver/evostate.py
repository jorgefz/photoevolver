
import dataclasses as dataclass
import numpy as np
import typing
import copy
import functools

@dataclass.dataclass(slots=True)
class EvoState:
    """
    Dataclass that stores the state of the simulation at a given time.
    This can be passed to model functions to calculate
    further parameters.
    """
    # Planet
    mass   :float = None # Planet mass   (M_earth)
    radius :float = None # Planet radius (R_earth)
    mcore  :float = None # Core mass     (M_earth)
    rcore  :float = None # Core radius   (R_earth)
    fenv   :float = None # = (mass-mcore)/mcore
    renv   :float = None # Envelope thickness (R_earth)
    period :float = None # Orbital period  (days)
    sep    :float = None # Semi-major axis (AU)

    # Star
    mstar  :float = None # Host star mass (M_sun)
    lx     :float = None # Host stat X-ray luminosity (erg/s)
    leuv   :float = None # Host stat EUV luminosity (erg/s)
    lbol   :float = None # Host stat bolometric luminosity (erg/s)

    # Simulation
    age    :float = None # Current age of the system (Myr)
    tstep  :float = None # Time step of the simulation (Myr)

    #################
    # Class Methods #
    #################

    @staticmethod
    def from_dict(fields :dict[str,float]) -> 'EvoState':
        """Class method that initialises a state from a dictionary containing
        simulation parameters. Only a subset of parameters may be defined,
        the rest will hold `None` value.

        Parameters
        ----------
        fields  : dict[str,float]
            Parameters to initialise in the state.

        Returns
        -------
        state : EvoState
            Instance with specified parameters

        Raises
        ------
        AttributeError
            When attempting to defined a parameter
            that is not a member of the class.
        """
        return EvoState(**fields)

    
    ##################
    # Public Methods #
    ##################

    def copy(self) -> 'EvoState':
        """
        Returns a shallow copy of the state.
        Since all of the parameters are floats,
        a deep copy will yield the same result.
        """
        return copy.copy(self)

    def asdict(self) -> dict[str,float]:
        """Returns the instance parameters as a dictionary
        The instance parameters cannot be modified through
        the returned dictionary, as it is a copy of the state.
        """
        return dataclass.asdict(self.copy())

    def update(self, new_values :dict) -> 'EvoState':
        """Returns a new instance replacing fields with given values"""
        return dataclass.replace(self, **new_values)

    def validate(self) -> bool:
        """
        Returns True if all of the instance parameters are valid floats
        (neither None nor NaN) and False otherwise.
        """
        values = list(self.asdict().values())
        return not any(x is None or np.isnan(x) for x in values)


def _model_wrapper(func :callable, state :EvoState, model_kw :dict):
    """ Wraps a model function to take an EvoState """
    return func(**state.asdict(), **model_kw)

def wrap_callback(fn :typing.Callable[typing.Any,float]
    ) -> typing.Callable[['EvoState',dict],typing.Any]:
    """
    Wraps a function `fn` that takes arbitrary keyword parameters
    so that it takes two arguments instead: an EvoState and 
    extra arguments as a dictionary. 
    The wrapped function retains the internal dunder name (__name__) and
    docs from the original function.

    This is useful to make model functions compatible with the evolution code,
    which passes the current simulation state together with user-defined
    keyword arguments to the models.
    
    Parameters
    ----------
    fn  : Callable[Any, Any]
        Function to wrap. Must allow for an arbitrary number of
        keyword arguments that may map to state parameters
        and/or model keyword arguments.
        Example: 
        ```
        def my_model(mass :float, **kwargs) -> float:
            pass
        ```

    Returns
    -------
    wrapped_fn : Callable[[EvoState,dict], Any]
        Wrapped function.

    Examples
    --------
    ```
    def rocky_radius(mass :float, **kw) -> float:
        return mass**(1/3)
    
    rocky_fn = wrap_callback(rocky_radius)
    rcore = rocky_fn(state, model_args)
    ```
    """
    wrapper = functools.partial(_model_wrapper, fn)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__  = fn.__doc__
    return wrapper

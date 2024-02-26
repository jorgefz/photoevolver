import os
import warnings
import functools

# Root path of the project
_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
_MODEL_DATA_DIR = _ROOT_DIR + '/model_data/'

# Decorator emits a warning signalling the function is deprecated
def deprecated(func :callable):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		name = getattr(func, '__name__', repr(func))
		warnings.warn(f"{name} is deprecated", DeprecationWarning)
		return func(*args, **kwargs)
	return wrapper

verbose = False # Prints simulation state on each time step
warnings_as_errors = False #Any warning raised will stop the simulation.
no_nans = False # If true, when a NaN is produced during simulation, the last non-NaN value of that parameter is used onwards.
enforce_bounds = False #
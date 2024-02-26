import os

# Root path of the project
_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
_MODEL_DATA_DIR = _ROOT_DIR + '/model_data/'

# Global parameters
class settings:
	"""
	verbose: bool				Prints simulation state on each time step
	warnings_as_errors: bool	Any warning raised will stop the simulation.
	no_nans: bool				If true, when a NaN is produced during simulation,
								the last non-NaN value of that parameter is used onwards.
	"""
	verbose = False
	warnings_as_errors = False
	no_nans = False
	enforce_bounds = False
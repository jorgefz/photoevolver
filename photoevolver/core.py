


# Global parameters

class globals:
	"""
	verbose: bool				Prints simulation state on each time step
	warnings_as_errors: bool	Any warning raised will stop the simulation.
	no_nans: bool				If true, when a NaN is produced during simulation,
								the last non-NaN value of that parameter is used onwards.
	"""
	verbose = False
	warnings_as_errors = False
	no_nans = False
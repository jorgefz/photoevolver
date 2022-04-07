

import pickle
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from typing import Any, Union, List, Callable

from .planet import Planet, indexable

import functools

class Tracks:

	@classmethod
	def load(filename: str) -> 'Tracks':
		with open(filename, 'rb') as handle:
			tracks = pickle.load(handle)
		return tracks

	def __init__(
			self,
			data: dict[str,np.ndarray] = None,
			base_pl: Planet = None,
			keys: list[str] = None
		):
		"""
		Stores how the simulation state evolves through time.

		Parameters
		----------
		data:    dict[str, np.ndarray]
			Tracks with the values of physical parameters at each point in time.
		base_pl: Planet
			Planet being evolved
		keys: list[str]
			List of track names. If this argument is provided and the data is not given,
			the interpolation functions are not generated.
		"""
		self.pl = base_pl
		self._interp_funcs = dict()

		# initialise with data
		if data is not None:
			for k in data.keys():
				if not indexable(data[k]): continue
				data[k] = np.array(data[k])
			self.tracks = data
			self.interpolate()
		
		# initalise tracks as empty lists using provided keys
		# the purpose is to append data manually later on
		elif keys is not None:
			self.tracks = {k:[] for k in keys}
		
		else: raise ValueError("keys or data not provided")

	def __str__(self):
		msg = f"<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}>\n"
		for k,v in self.tracks.items():
			msg += f"{k} = ".ljust(10)
			msg += f"[ {v[0]:.3g} ... {v[-1]:.3g} ]" if len(v) > 0 else "[ ]"
			msg += '\n'
		return msg

	def __repr__(self):
		return self.__str__()

	def __getitem__(self, index: Union[int,str]) -> Union[list[float], dict[str,float]]:
		"""
		Retrieves an item in the Tracks.
		This could be a specific track, if provided with the string key,
		or a state at a specific point in time, if provided with an integer index
		"""
		if isinstance(index, str): # indexed by field key (like a dict)
			return self.tracks[index]
		elif isinstance(index, int): # indexed by integer index (like an array)
			view = { f: self.tracks[f][index] for f in self.tracks }
			return view
		else: raise ValueError(f"Unsupported type {type(index)} for __getitem__")
	
	def __setitem__(self, index: Union[int,str], value):
		pass

	def __len__(self):
		""" Number of ages at which physical parameters are sampled """
		return len(self.tracks)

	def __add__(self, t2: 'Tracks'):
		if not isinstance(t2, Tracks):
			raise TypeError(f"Tracks can only be concatenated \
					with another Tracks instance, not '{type(t2)}'")
		elif max(self['Age']) <= min(t2['Age']):
			# Track 2 is to the right (older ages)
			new = Tracks(self.tracks, t2.pl)
			for f in new.tracks.keys():
				new.tracks[f] = np.append( new.tracks[f], t2.tracks[f] )
			new.interpolate()
			return new
		elif min(self['Age']) >= max(t2['Age']):
			# Track 2 is to the left (younger ages)
			new = Tracks(t2.tracks, t2.pl)
			for f in new.tracks.keys():
				new.tracks[f] += self.tracks[f]
			new.interpolate()
			return new
		else:
			raise ValueError(f"The ages of Tracks to concatenate must not \
					overlap: ({min(self['Age'])},{max(self['Age'])}) \
					and ({min(t2['Age'])},{max(t2['Age'])}) Myr")

	def keys(self):
		return list(self.tracks.keys())
	
	def save(self, filename: str):
		""" stores tracks object as pickle file """
		with open(filename, 'wb') as handle:
			pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	def as_dict(self) -> dict:
		""" Returns tracks as a dictionary of arrays """
		return deepcopy(self.tracks)

	def append(self, t2: 'Tracks'):
		self = self + t2
		return self

	def Age(self, age: float) -> dict:
		"""
		Retrieves the value of all parameters at a given point in time.
		"""
		state = { k: float(self.__dict__[k](age))  for k in self.keys() if k != 'Age' }
		return state

	def _interp(self, key: str, age: float) -> float:
		"""
		Interpolates given track and returns its value at a given age
		"""
		if key in self._interp_funcs:
			return float(self._interp_funcs[key](age))
		self._interp_funcs[key] = interp1d(
			x=self.tracks['Age'],
			y=self.tracks[key],
			fill_value='extrapolate'
		)
		return float(self._interp_funcs[key](age))


	def interpolate(self):
		""" Interpolates given tracks and generates instance methods to retrieve parameters at any age """
		for key in self.keys():
			# Convert tracks to numpy arrays
			self.tracks[key] = np.array(self.tracks[key])

			# this allows easy access to tracks, e.g. tracks.AgeTracks instead of tracks['Age']
			setattr(self, key+'Track', self.tracks[key])

			# Tracks.Age returns simulation state instead (see above)
			if key == "Age": continue

			# this allows to call e.g. tracks.Lx(age) and get a value at that age
			wrapper = functools.partial(self._interp, key)
			setattr(self, key, wrapper)
	
	
	def planet(self, age: float, *_) -> Planet:
		""" Returns a planet at a specified age """
		p = self.pl
		pl = Planet(
			mp    = float(self.Mp(age)),   rp    = float(self.Rp(age)),
			mcore = float(p.mcore),        rcore = float(p.rcore),
			menv  = float(self.Menv(age)), renv  = float(self.Renv(age)),
			dist  = float(p.dist),         age   = float(age),
			fenv  = float(self.Fenv(age)),
		)
		return pl

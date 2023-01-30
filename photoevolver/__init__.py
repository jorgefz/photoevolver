
from .version import __version__,__author__
import os
import platform

from photoevolver import planet, tracks, evo, core, massloss, structure, owenwu17
from photoevolver.planet import Planet, Otegi20MR
from photoevolver.evo import evolve, evolve_back, evolve_forward, Tracks
from photoevolver.core import settings
from photoevolver import plotter
from photoevolver.startable import StarTable
from photoevolver.libcfn import libcfn
from photoevolver import utils

from photoevolver import dev

# TEMPORARY - DEBUG
from photoevolver.K18grid import interpolator as K18grid

# legacy
from photoevolver.libcfn import libcfn as libc

# import photoevolver.old

py_dir = os.path.dirname(os.path.realpath(__file__))


# Load C functions
__lib = libcfn._load(py_dir + "/libcfn/shared/libcfn.so")
libcfn._setup(__lib)
libcfn._lib = __lib

"""

===== Photoevolver =====

Package description here


"""


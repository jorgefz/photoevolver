
from .version import __version__,__author__
import os
import platform

from photoevolver import planet, tracks, evo, core, massloss, structure, owenwu17
from photoevolver.libcfn import libcfn

# legacy
from photoevolver.libcfn import libcfn as libc

from photoevolver.evo import Planet, evolve, evolve_back, evolve_forward, Tracks, OtegiMassRadiusRelation
from photoevolver.core import globals
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


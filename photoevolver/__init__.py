
"""

===== Photoevolver =====

Package description here


"""

from .version import __version__,__author__
import os
import platform

from photoevolver import planet, tracks, evo, core, massloss, structure, owenwu17
from photoevolver.planet import Planet, Otegi20MR
from photoevolver.evo import evolve, evolve_back, evolve_forward, Tracks
from photoevolver.core import settings
from photoevolver import plotter
from photoevolver.startable import StarTable
from photoevolver import utils
from photoevolver import dev
from photoevolver import cmodels

# TEMPORARY - DEBUG - integrate into ph.massloss
# from photoevolver.K18grid import interpolator as K18grid
from photoevolver.K18interp import K18Interpolator






"""

===== Photoevolver =====

Package description here


"""

from .version import __version__, __author__
import os
import platform

from photoevolver.settings import settings
# from photoevolver.startable import StarTable

from photoevolver import (
    models, cmodels,
    physics, utils, plotter,
    evostate, integrator, planet
)

from photoevolver.planet import Planet
from photoevolver.evostate import wrap_callback

from photoevolver import legacy 
from photoevolver import dev




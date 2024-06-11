
"""

===== Photoevolver =====

Package description here


"""

from .version import __version__, __author__
import os
import platform

from photoevolver import settings

from photoevolver import (
    models, cmodels,
    physics, utils,
    evostate, integrator, planet
)

from photoevolver.planet import Planet
from photoevolver.evostate import wrap_callback
from photoevolver import dev


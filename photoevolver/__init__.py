
from .version import __version__,__author__
import os
import platform

from photoevolver import planet, tracks, evo, core, massloss, structure, owenwu17
from photoevolver.libc import libc
from photoevolver.evo import Planet, evolve, evolve_back, evolve_forward, Tracks, OtegiMassRadiusRelation
from photoevolver.core import globals
# import photoevolver.old

py_dir = os.path.dirname(os.path.realpath(__file__))


if platform.system() == "Windows":
    try:
        __lib = libc._load_lib(py_dir + "libc/shared/libc.dll")
        print(__lib)
    except:
        raise OSError(" compilation on Windows is not yet supported. \
You can manually compile the code at photoevolver/libc/src \
as a shared library into photoevolver/libc/shared/libc.dll, \
and run the package setup again.")

else:
    try:
        __lib = libc._load_lib(py_dir + "/libc/shared/libc.so")
    except:
        libc._compile()
        __lib = libc._load_lib(py_dir + "/libc/shared/libc.so")

libc._setup_lib(__lib)
libc._lib = __lib

"""

===== Photoevolver =====

Package description here


"""


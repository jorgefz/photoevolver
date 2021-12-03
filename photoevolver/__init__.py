import photoevolver.core
import photoevolver.massloss
import photoevolver.structure
#import photoevolver.old
import photoevolver.owenwu17
import photoevolver.libc.libc as libc

from .version import __version__,__author__
import os

py_dir = os.path.dirname(os.path.realpath(__file__))

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




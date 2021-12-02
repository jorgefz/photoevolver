
import subprocess as sp
import ctypes
import os
import sys
import platform

_lib = None

def _compile():
    py_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(py_dir+"/shared"): os.mkdir(py_dir+"/shared")
    if platform.system() == "Windows":
        print(" Windows is currently not suported for compiling the C library. \
                You may manually compile the code at photoevolver/lib/src into a shared library, \
                name it 'libc.so', and place it in the folder photoevolver/libc/shared")
        return 0
    retcode = sp.run(["sh", py_dir+"/make.sh"]).returncode
    if retcode != 0: raise OSError(f" Failed to compile library {retcode}")
    return retcode

def _load_lib(path):
    return ctypes.CDLL(os.path.abspath(path))

def _setup_lib(lib):
    global _lib
    _lib = lib

    """
    double EnergyLimitedMassloss(double fxuv, double radius, double mass, double mstar, double a, double beta, double eff)
    """
    lib.EnergyLimitedMassloss.argtypes = [ctypes.c_double] * 7
    lib.EnergyLimitedMassloss.restype  = ctypes.c_double

    """
    double BetaSalz16(double fxuv, double mass, double radius)
    """
    lib.BetaSalz16.argtypes = [ctypes.c_double]*3
    lib.BetaSalz16.restype  = ctypes.c_double

    """
    double EffSalz16(double mass, double radius)
    """
    lib.EffSalz16.argtypes = [ctypes.c_double]*2
    lib.EffSalz16.restype  = ctypes.c_double

    """
    double Kubyshkina18Massloss(double fxuv, double fbol, double mass, double radius, double a)
    """
    lib.Kubyshkina18Massloss.argtypes = [ctypes.c_double]*5
    lib.Kubyshkina18Massloss.restype  = ctypes.c_double

    """
    double LopezFortney14Structure(double mass, double fenv, double fbol, double age, int enhanced_opacity){
    """
    lib.LopezFortney14Structure.argtypes = [ctypes.c_double]*4 + [ctypes.c_int]
    lib.LopezFortney14Structure.restype  = ctypes.c_double

    """
    double ChenRogers16Structure(double mass, double fenv, double fbol, double age){
    """
    lib.ChenRogers16Structure.argtypes = [ctypes.c_double]*4
    lib.ChenRogers16Structure.restype  = ctypes.c_double
    
    """
    double OwenWu17Structure(double mass, double fenv, double fbol, double age, double rcore){
    """
    lib.OwenWu17Structure.argtypes = [ctypes.c_double]*5
    lib.OwenWu17Structure.restype  = ctypes.c_double



import numpy as np
import astropy.units as U


def BetaSalz16(**kwargs):
    fxuv = kwargs['fxuv'] if 'fxuv' in kwargs.keys() else kwargs['Lxuv']/(4*np.pi*(kwargs['dist']*U.au.to('cm'))**2)
    mp = kwargs['mp']
    rp = kwargs['rp']
    return _lib.BetaSalz16(fxuv, mp, rp)

def EffSalz16(**kwargs):
    mp = kwargs['mp']
    rp = kwargs['rp']
    return _lib.EffSalz16(mp, rp)

def EnergyLimited(**kwargs):
    if 'fxuv' not in kwargs.keys():
        fxuv = kwargs['Lxuv'] / (4 * np.pi * (kwargs['dist']*U.au.to('cm'))**2 )
    else: fxuv = kwargs['fxuv']
    rp = kwargs['rp']
    mp = kwargs['mp']
    mstar = kwargs['mstar']
    dist = kwargs['dist']
    
    if 'beta' not in kwargs: beta = 1.0
    elif isinstance(kwargs['beta'],str): beta = BetaSalz16(**kwargs)
    elif callable(kwargs['beta']): beta = kwargs['beta'](**kwargs)
    else: beta = kwargs['beta']

    if 'eff' not in kwargs: eff = 0.15
    elif isinstance(kwargs['eff'],str): eff = EffSalz16(**kwargs)
    elif callable(kwargs['eff']): eff = kwargs['eff'](**kwargs)
    else: eff = kwargs['eff']
    
    MearthMyr2gs = 5.97e27 / 3.15e13
    return _lib.EnergyLimitedMassloss(fxuv, rp, mp, mstar, dist, beta, eff) / MearthMyr2gs


def K18(**kwargs):
    mp = kwargs['mp']
    rp = kwargs['rp']
    dist = kwargs['dist']
    fbol = kwargs['fbol']
    if 'fxuv' not in kwargs.keys():
        fxuv = kwargs['Lxuv'] / (4 * np.pi * (kwargs['dist']*U.au.to('cm'))**2 )
    else: fxuv = kwargs['fxuv']
    MearthMyr2gs = 5.97e27 / 3.15e13
    return _lib.Kubyshkina18Massloss(fxuv, fbol, mp, rp, dist) / MearthMyr2gs

def LF14(mass, fenv, fbol, age, enhanced_opacity = False, **kwargs):
    return _lib.LopezFortney14Structure(mass, fenv, fbol, age, enhanced_opacity)

def CR16(mass, fenv, fbol, age, **kwargs):
    return _lib.ChenRogers16Structure(mass, fenv, fbol, age)

def OW17(mass, fenv, fbol, age, rcore, **kwargs):
    return _lib.OwenWu17Structure(mass, fenv, fbol, age, rcore)




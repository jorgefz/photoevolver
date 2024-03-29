"""
Test functions for photoevolver.physics
meant to be used by pytest.
"""

import numpy as np
from astropy import units, constants
import pytest

from photoevolver import physics


def test_fbol_earth():
    """Tests photoevolver.physics.fbol_earth.
    Should return solar irradiance in erg/s/cm^2"""
    result = physics.fbol_earth()
    result_si = result * units.erg.to('J')/units.cm.to('m')**2
    nominal_si = 1400.0
    assert np.isclose(result_si, nominal_si, rtol=0.1)

def test_si_flux():
    """Tests photoevolver.physics.SI_flux.
    Should convert flux from erg/s/cm^2 to SI W/m^2"""
    flux = 100.0 # erg/s/cm^2
    flux_si = flux * units.erg.to('J')/units.cm.to('m')**2
    result_si = physics.SI_flux(flux)
    assert np.isclose(flux_si, result_si)

def test_get_flux():
    """Tests photoevolver.physics.get_flux.
    Should convert luminosity to flux """
    # Use distance in AU
    lum = constants.L_sun.to("erg/s").value
    dist_au = 1.0 # AU
    result_au = physics.get_flux(lum=lum, dist_au=dist_au)
    nominal_au = 1400.0 * units.J.to('erg') / units.m.to('cm')**2
    assert np.isclose(result_au, nominal_au, rtol=0.1)

    # Use distance in pc
    dist_pc = 1.0 # pc
    result_pc = physics.get_flux(lum=lum, dist_pc=dist_pc)
    nominal_pc = 3.2e-5 # erg/s/cm^2
    assert np.isclose(result_pc, nominal_pc, rtol=0.1)

    # Provide both distances - AU should take precedence
    result_au_2 = physics.get_flux(lum=lum, dist_au=dist_au, dist_pc=dist_pc)
    assert np.isclose(result_au_2, result_au)

    # Error when providing no distances
    with pytest.raises(ValueError):
        physics.get_flux(lum=lum)


def test_get_luminosity():
    """Tests photoevolver.physics.get_luminosity.
    Should convert flux to luminosity """
    # Use distance in AU
    dist_au = 1.0
    flux_au = physics.fbol_earth()
    nominal_au = constants.L_sun.to("erg/s").value
    result_au = physics.get_luminosity(flux=flux_au, dist_au=dist_au)
    assert np.isclose(result_au, nominal_au)

    # Use distance in pc
    dist_pc = 1.0 # pc
    flux_pc = 8.358e-11 # erg/s/cm^2
    nominal_pc = 1e28 # erg/s
    result_pc = physics.get_luminosity(flux=flux_pc, dist_pc=dist_pc)
    assert np.isclose(result_pc, nominal_pc, rtol=0.1)

    # Use both distances
    result_au_2 = physics.get_luminosity(flux=flux_au,
        dist_pc=dist_pc, dist_au=dist_au)
    assert np.isclose(result_au_2, result_au)

    with pytest.raises(ValueError):
        physics.get_luminosity(1.0)

def test_keplers_third_law():
    """Tests photoevolver.physics.keplers_third_law"""
    # Periods separations, and masses of Solar System planets
    # Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
    pl_per =  np.array(
        [88.0, 224.7, 365.2, 687.0, 4331, 10747, 30589, 59800])
    pl_sep = units.km.to('au') * 1e6 * np.array(
        [57.9, 108.2, 149.6, 228.0, 778.5, 1432.0, 2867.0, 4515.0])
    pl_mass = units.kg.to('M_earth') * 1e24 * np.array(
        [0.330, 4.87, 5.97, 0.642, 1898, 568, 86.8, 102])
    msun = 1.0

    kepler_fn = np.vectorize(physics.keplers_third_law)

    # Test separations
    result_sep = kepler_fn(big_mass=msun, small_mass=pl_mass, period=pl_per)
    assert np.allclose(result_sep, pl_sep, rtol=0.1)

    # Test periods
    result_per = kepler_fn(big_mass=msun, small_mass=pl_mass, sep=pl_sep)
    assert np.allclose(result_per, pl_per, rtol=0.1)

    # Test both - separation should have precedence
    result_sep_2 = kepler_fn(big_mass=msun, small_mass=pl_mass,
                             sep=pl_sep, period=pl_per)
    assert np.allclose(result_sep, result_sep_2)

    # Test neither - error
    with pytest.raises(ValueError):
        physics.keplers_third_law(big_mass=1.0)

def test_temp_eq():
    """Tests photoevolver.physics.temp_eq"""
    nominal_teq = 279 # K - blackbody equilibrium temperature of Earth
    result_teq  = physics.temp_eq(physics.fbol_earth(), albedo = 0)
    assert np.isclose(nominal_teq, result_teq, rtol=0.01)

def test_planet_density():
    """Tests photoevolver.physics.planet_density"""
    nominal_dens = 5.51 # g/cm^3 - Earth's density
    result_dens  = physics.planet_density(mass=1.0, radius=1.0)
    assert np.isclose(nominal_dens, result_dens, rtol=0.01)

def test_prot_from_vsini():
    """Tests photoevolver.physics.prot_from_vsini"""
    input_vsini  = 1.06 # km/s
    input_rstar  = 0.949  # solar radii
    nominal_prot = 45.0 # days
    result_prot = physics.prot_from_vsini(input_vsini, input_rstar)
    assert np.isclose(result_prot, nominal_prot, rtol=0.01)

def test_rossby_number():
    """Tests photoevolver.physics.rossby_number"""
    # Test default behaviour
    vkcolor = 1.2
    prot = 15.0 # days
    result_ross = physics.rossby_number(vkcolor, prot, safe=True)
    assert 0 <= result_ross <= 10.0

    # Test full output
    result_full = physics.rossby_number(vkcolor, prot, full_out=True)
    assert isinstance(result_full,tuple) and len(result_full) == 2
    assert result_full[0] == result_ross

    # Test unsafe mode with out-of-bounds parameter
    # vk_bounds = [1.1, 6.6]
    bad_vkcolor_low = 0.1
    bad_vkcolor_high = 10.0
    physics.rossby_number(bad_vkcolor_low, prot)
    physics.rossby_number(bad_vkcolor_high, prot)

    # Test safe mode with out-of-bounds parameter
    with pytest.raises(ValueError):
        physics.rossby_number(bad_vkcolor_low, prot, safe=True)
    with pytest.raises(ValueError):
        physics.rossby_number(bad_vkcolor_high, prot, safe=True)

def test_rossby_number_from_mass():
    """Tests photoevolver.physics.rossby_number_from_mass"""
    # Test default behaviour
    mstar = 1.0 # Msun
    prot = 15.0 # days
    result_ross = physics.rossby_number_from_mass(mstar, prot, safe=True)
    assert 0 <= result_ross <= 10.0

    # Test full output
    result_full = physics.rossby_number_from_mass(mstar, prot, full_out=True)
    assert isinstance(result_full,tuple) and len(result_full) == 2
    assert result_full[0] == result_ross

    # Test unsafe mode with out-of-bounds parameter
    # mass_bounds = [0.09, 1.36]
    bad_mstar_low = 0.05
    bad_mstar_high = 2.0
    physics.rossby_number_from_mass(bad_mstar_low, prot)
    physics.rossby_number_from_mass(bad_mstar_high, prot)

    # Test safe mode with out-of-bounds parameter
    with pytest.raises(ValueError):
        physics.rossby_number_from_mass(bad_mstar_low, prot, safe=True)
    with pytest.raises(ValueError):
        physics.rossby_number_from_mass(bad_mstar_high, prot, safe=True)


def test_xray_hardness():
    """Tests photoevolver.physics.xray_hardness"""
    # When hard > soft
    result = physics.xray_hardness(hard=2.0, soft=1.0)
    assert result > 0.0

    # When hard < soft
    result = physics.xray_hardness(hard=1.0, soft=2.0)
    assert result < 0.0

    # When hard = soft
    result = physics.xray_hardness(hard=1.0, soft=1.0)
    assert result == 0.0


def test_jeans_parameter():
    """Tests photoevolver.physics.jeans_parameter"""
    # A planet 10 times further away should be sqrt(10) times more tightly bound
    args = {'mass':1.0, 'radius':1.0, 'lbol':1e33}
    factor = 10.0
    sep_close = 0.1 # AU
    sep_far   = sep_close * factor # AU
    jeans_close = physics.jeans_parameter(**args, sep = sep_close)
    jeans_far   = physics.jeans_parameter(**args, sep = sep_far)
    assert np.isclose(jeans_far, jeans_close*np.sqrt(factor))

    # A planet twice as heavy should be twice as tightly bound
    args = {'radius':1.0, 'lbol':1e33, 'sep':0.5}
    factor = 2.0
    mass_l = 1.0
    mass_h = mass_l * factor
    jeans_light = physics.jeans_parameter(mass=mass_l, **args)
    jeans_heavy = physics.jeans_parameter(mass=mass_h, **args)
    assert np.isclose(jeans_heavy, jeans_light * factor)

    # A planet twice as large should be two times less tightly bound
    args = {'mass':1.0, 'lbol':1e33, 'sep':0.5}
    factor = 2.0
    radius_s = 1.0
    radius_l = radius_s * factor
    jeans_small = physics.jeans_parameter(radius=radius_s, **args)
    jeans_large = physics.jeans_parameter(radius=radius_l, **args)
    assert np.isclose(jeans_small, jeans_large * factor)


def test_get_fenv_planet_ratio():
    """Tests photoevolver.physics.get_fenv_planet_ratio"""
    # Small values sould match
    xenv = 1e-5
    fenv = physics.get_fenv_planet_ratio(xenv)
    assert np.isclose(xenv, fenv)
    # Large values diverge
    xenv = 1.0
    fenv = physics.get_fenv_planet_ratio(xenv)
    assert np.isclose(fenv, xenv/2)


def test_get_fenv_core_ratio():
    """Tests photoevolver.physics.get_fenv_core_ratio"""
    # Small values sould match
    fenv = 1e-5
    xenv = physics.get_fenv_core_ratio(fenv)
    assert np.isclose(xenv, fenv)
    # Large values diverge
    fenv = 0.50
    xenv = physics.get_fenv_core_ratio(fenv)
    assert np.isclose(fenv, xenv/2)

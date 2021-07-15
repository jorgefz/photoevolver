__version__ = "1.3"
__author__ = "Jorge Fernandez"

"""

########################
#     PHOTOEVOLVER     #
########################

--- To Do ---

- Turn Planet input variables to **kwargs
- Function to evolve BOTH forward and back
- Alternate star formulation: dict with Lbol and Lxuv, same length as ages
- Alternate time_step formulation: array of ages or time steps.
- Return more tracks, e.g. mass loss
- Make tracks into a class to retrieve values at any age.

--- Changelog ---

Version 1.3
    - Added Otegi (2020) and Owen (EvapMass) relations to calculate core mass from radius.
    - Added plot_tracks function to plot tracks against age.
    - Added new tracks: Lxuv, Renv, Menv, Fenv, Dens
    - Planet class now accepts either core mass or radius as input, and calculates the other from a M-R relation (default is Otegi 2020).
    - Fixed bug with Kubyshkina 2018 mass loss equation: required Lbol.
    - Removed legacy Evolve class

Version 1.2
    - Added option to calculate planet core radius from custom mass-radius relation.
    - Added Otegi et al (2020) mass-radius relations only to calculate radius from mass.
    - Changed class ph.core.Evolve.forward to function ph.core.evolve_forward
    - Added backwards evolution with evolve_back.
    - Updated README

Version 1.1
    - Fixed ph.core.Planet repr when some variables are None

Version 1.0
    - First working version

"""

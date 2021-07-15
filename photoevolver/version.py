__version__ = "1.3"
__author__ = "Jorge Fernandez"

"""

########################
#     PHOTOEVOLVER     #
########################

--- Changelog ---

Version 1.3
    - Added Otegi (2020) and Owen (EvapMass) relations to calculate core mass from radius.
    - Planet class now accepts either core mass or radius as input, and calculates the other from a M-R relation (default is Otegi 2020).

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

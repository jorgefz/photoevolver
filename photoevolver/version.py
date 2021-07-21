__version__ = "1.4.1"
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
- Return more tracks, e.g. mass loss, eff, beta, etc
- Fix bug with Kubyshkina 2018

--- Changelog ---

Version 1.4.1
    - Tracks __add__ now allows two tracks that overlap on one data point (the joint), such that two tracks, past and future, for the same planet can be sucessfully concatenated.
    - Better treatment for planets with zero envelope: evolve_back sets minimum renv and fenv, evolve_forward skips mass loss and structure calculation if renv or fenv fall down to zero.

Version 1.4
    - The envelope mass fraction can now be calculated by solving the structure equations. As a result, planets no longer require an input envelope mass fraction (fenv), but can accept the envelope radius 'renv' instead. Alternatively, one can also define the planet's total radius (rp), from which renv is calculated. 
    - Evolve functions now return a Tracks class. Works like a dictionary: you can do tracks['key'] and tracks.keys(). Furthermore, one can obtain the value of any parameter at any age by calling relevant methods: tracks.radius(55), tracks.fenv(100), and so on with all keys except 'Age'.
    - Added planet return method for Tracks class. Given an age (and a Planet class on which the tracks are based on), it returns a new instance of Planet with its parameters at the given age. E.g. planet2 = tracks.planet(100, planet1)
    - Added appending/adding tracks whose starting and ending ages don't overlap. Use append method or simply + operator.

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

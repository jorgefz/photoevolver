__version__ = "1.6.4"
__author__ = "Jorge Fernandez"

"""

########################
#     PHOTOEVOLVER     #
########################

--- To Do ---

- Turn Planet input variables to **kwargs
- Function to evolve BOTH forward and back
- Return more tracks including custom ones, e.g. eff, beta, etc
- Envelope loss timescale calculator
- Unphysical envelope enlargement calculator

--- Changelog ---

Version 1.6.4
    - Changed behaviour of `solve_structure` when solution could not be found.
        Now it returns the solution status as a boolean `success` as part of the return dict.
    - Updated function docs and readme.
    - Added Kubyshkina & Fossati (2021) grid and interpolator `ph.massloss.K18Interpolator`.
    - Added utils, plotter, and StarTable.
    - Changed Otegi 2020 mass-radius relations to work with the uncertainties package.
    - Added structure formulation for planets with water/icy cores `ChenRogers16Water`.

Version 1.6.3
    - Solved bug where ph.Tracks.load was missing the class argument required as a classmethod.
    - Added new quantities to tracks: Lx, Leuv, Teq.
    - Added EvoState method 'to_dict' which returns a copy of the state data.
    - Added `solve_structure` which calculates the envelope mass fraction of a planet
        given its observed mass and radius alone (along with its age and orbital distance,
        and the star's bolometric luminosity).
    - Added `solve_structure_uncert`, which behaves like `solve_structure` but takes
    and returns values with uncertainties (`uncertainties.ufloat`).

Version 1.6.2
	- Improved compiling pipeline for C functions by using a Makefile
	- Renamed photoevolver.libc to photoevolver.libcfn, although the first is still available for legacy reasons.

Version 1.6.1
    - Fixed bug where pickling a Track object failed due to using a curried function for interpolation. Using functools.partial instead fixes the issue.
    - Fixed bug where simulation fluxes were not updated until the end of the simulation step, now these are updated at the beginning.
    - Added ph.evo.King18Leuv function to calculate EUV luminosity from X-rays.

Version 1.6
    - Improved code standards, package structure, and simulation stability.
    - API changed. Import Planet, and evolution functions directly from top-level, e.g.:
        import photoevolver as ph
        ph.Planet
        ph.evolve_forward
    - Added core-powered mass loss to formulations. Access with:
        ph.massloss.CorePowered
    - Added new function 'evolve' that deprecates evolve_forward and evolve_back. Use argument 'mode' for future evolution ('future') or past evolution ('past'). 
    - Added more control over the simulation:
        * Set photoevolver.globals.verbose to True to print out the simulation state on every iteration.
        * Set photoevolver.globals.warnings_as_errors to True to stop the simulation if a NaN value is produced.
        * The argument 'nan_callback' on the function 'evolve' lets the user define a custom callback called when a NaN is produced.
    - Tracks method that intepolates data now extrapolates beyond predetermined ages.
    - Updated readme

Version 1.5.2
    - Changed structure formulation root-finding function from Scipy fsolve to Scipy least_squares. Now recalculating Renv with solved Fenv will not result in a wildly different Renv from the one originally input.

Version 1.5.1
    - Solved bug with Lopez & Fortney 2014 formulation. Age term power changed from -0.18 (enhanced opacity) to -0.11 (solar metallicity).
    - Confirmed correctness for Kubyshkina 2018 and Energy Limited mass loss formulations.
    - Confirmed correctness for Lopez & Fortney 2014 structure formulation.

Version 1.5
    - Evolve functions now accept an array of ages at which the planet's parameters are updated. If used, it overrides the age_end and time_step parameters.
    - Evolve functions now accept an alternative to a mors.Star object: a dictionary of two keys, 'Lxuv' and 'Lbol', each containing an array of the same length as 'ages' that tracks the XUV and bolometric fluxes of the star at each age.
    - Removed dependency on Mors module

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

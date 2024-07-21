
# photoevolver

# To do
- Add tests
- Improve tutorials
- Improve readme

# v2.5
- Added core-powered mass loss
- Removed plotter, dev, and legacy functions
- Updated jupyter notebooks and readme
- Improved model data for star_spada13
- Added power law x-ray evolution models, including Jackson+12.
- Added function to planet class to compute mass loss rate with uncertainties.

# v2.4
- Moved models to subfolder and separated them into files
- Added `deprecated` decorator
- Added Spada+13 and Pecaut+13 stellar models
- Copied Rossby number functions from physics file to models folder, and marked old ones as deprecated.
- Deprecated `StarTable` in dev folder, as it is a proxy of `star_pecaut13`.
- Moved Kubyshkina+18 model dataset to model data folder
- Fixed bug with `rotation_activity_johnstone21` using incorrect model Rossby number.
- Added `evolve_batch` to run multiple sims in parallel

# v2.3
- Removed usage of pow in compile-time constant in C extension, as it is a GCC extension but Clang does not allow it.
- Added Zeng+19 core models.
- Added albedo to equation of equilibrium temperature

# v2.2

- `Planet.set_models` returns its own instance, enabling chaining of `Planet` methods.
- Replaced `Planet._parse_star` with public static function `Planet.parse_star`.

# v2.1

- Added github banners
- Moved version history to CHANGELOG
- Added pytest settings ini

# v2.0

- Restructured folders and submodules
- Added tests with pytest
- Improved planet and evolution API
- Evolution code now outputs pandas dataframes
- Evolution now supports the RK45 integration method
- Improved pydocs
- Integrated C functions with CPython

# v1.6.4

- Changed behaviour of `solve_structure` when solution could not be found. Now it returns the solution status as a boolean `success` as part of the return dict.
- Updated function docs and readme.
- Added Kubyshkina & Fossati (2021) grid and interpolator `ph.massloss.K18Interpolator`.
- Added utils, plotter, and StarTable.
- Changed Otegi 2020 mass-radius relations to work with the uncertainties package.
- Added structure formulation for planets with water/icy cores `ChenRogers16Water`.

# v1.6.3

- Solved bug where ph.Tracks.load was missing the class argument required as a classmethod.
- Added new quantities to tracks: Lx, Leuv, Teq.
- Added EvoState method 'to_dict' which returns a copy of the state data.
- Added `solve_structure` which calculates the envelope mass fraction of a planet given its observed mass and radius alone (along with its age and orbital distance, and the star's bolometric luminosity).
- Added `solve_structure_uncert`, which behaves like `solve_structure` but takes
and returns values with uncertainties (`uncertainties.ufloat`).

# v1.6.2

- Improved compiling pipeline for C functions by using a Makefile
- Renamed photoevolver.libc to photoevolver.libcfn, although the first is still available for legacy reasons.

# v1.6.1

- Fixed bug where pickling a Track object failed due to using a curried function for interpolation. Using functools.partial instead fixes the issue.
- Fixed bug where simulation fluxes were not updated until the end of the simulation step, now these are updated at the beginning.
- Added ph.evo.King18Leuv function to calculate EUV luminosity from X-rays.

# v1.6

- Improved code standards, package structure, and simulation stability.
- API changed. Import Planet, and evolution functions directly from top-level, e.g.: `import photoevolver as ph`, `ph.Planet`, `ph.evolve_forward`.
- Added core-powered mass loss to formulations. Access with `ph.massloss.CorePowered`.
- Added new function 'evolve' that deprecates evolve_forward and evolve_back. Use argument 'mode' for future evolution ('future') or past evolution ('past'). 
- Added more control over the simulation:
    * Set `photoevolver.globals.verbose` to True to print out the simulation state on every iteration.
    * Set `photoevolver.globals.warnings_as_errors` to True to stop the simulation if a NaN value is produced.
    * The argument 'nan_callback' on the function 'evolve' lets the user define a custom callback called when a NaN is produced.
- Tracks method that intepolates data now extrapolates beyond predetermined ages.
- Updated readme

# v1.5.2

- Changed structure formulation root-finding function from Scipy fsolve to Scipy least_squares. Now recalculating Renv with solved Fenv will not result in a wildly different Renv from the one originally input.

# v1.5.1

- Solved bug with Lopez & Fortney 2014 formulation. Age term power changed from -0.18 (enhanced opacity) to -0.11 (solar metallicity).
- Confirmed correctness for Kubyshkina 2018 and Energy Limited mass loss formulations.
- Confirmed correctness for Lopez & Fortney 2014 structure formulation.

# v1.5

- Evolve functions now accept an array of ages at which the planet's parameters are updated. If used, it overrides the age_end and time_step parameters.
- Evolve functions now accept an alternative to a mors.Star object: a dictionary of two keys, 'Lxuv' and 'Lbol', each containing an array of the same length as 'ages' that tracks the XUV and bolometric fluxes of the star at each age.
- Removed dependency on Mors module

# v1.4.1

- Tracks __add__ now allows two tracks that overlap on one data point (the joint), such that two tracks, past and future, for the same planet can be sucessfully concatenated.
- Better treatment for planets with zero envelope: evolve_back sets minimum renv and fenv, evolve_forward skips mass loss and structure calculation if renv or fenv fall down to zero.

# v1.4

- The envelope mass fraction can now be calculated by solving the structure equations. As a result, planets no longer require an input envelope mass fraction (fenv), but can accept the envelope radius 'renv' instead. Alternatively, one can also define the planet's total radius (rp), from which renv is calculated. 
- Evolve functions now return a Tracks class. Works like a dictionary: you can do tracks['key'] and tracks.keys(). Furthermore, one can obtain the value of any parameter at any age by calling relevant methods: tracks.radius(55), tracks.fenv(100), and so on with all keys except 'Age'.
- Added planet return method for Tracks class. Given an age (and a Planet class on which the tracks are based on), it returns a new instance of Planet with its parameters at the given age. E.g. planet2 = tracks.planet(100, planet1)
- Added appending/adding tracks whose starting and ending ages don't overlap. Use append method or simply + operator.

# v1.3

- Added Otegi (2020) and Owen (EvapMass) relations to calculate core mass from radius.
- Added plot_tracks function to plot tracks against age.
- Added new tracks: Lxuv, Renv, Menv, Fenv, Dens
- Planet class now accepts either core mass or radius as input, and calculates the other from a M-R relation (default is Otegi 2020).
- Fixed bug with Kubyshkina 2018 mass loss equation: required Lbol.
- Removed legacy Evolve class

# v1.2

- Added option to calculate planet core radius from custom mass-radius relation.
- Added Otegi et al (2020) mass-radius relations only to calculate radius from mass.
- Changed class ph.core.Evolve.forward to function ph.core.evolve_forward
- Added backwards evolution with evolve_back.
- Updated README

# v1.1

- Fixed ph.core.Planet repr when some variables are None

# v1.0

- First working version

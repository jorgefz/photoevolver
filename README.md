# Photoevolver
[![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-notebooks-e67e22)](https://www.github.com/jorgefz/photoevolver/blob/main/examples)
[![Coverage](https://img.shields.io/badge/Coverage-66%25-brightgreen)](./README.md)
[![Python](https://img.shields.io/badge/Python->=3.10-cornflowerblue)](./README.md)
[![Read the paper](https://img.shields.io/badge/Read-the%20paper-blue)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.tmp.1197F/abstract)
[![License](https://img.shields.io/badge/License-GNU%20GPL%20v3.0-mediumorchid)](https://www.github.com/jorgefz/photoevolver/blob/main/LICENSE)


`photoevolver` is a Python module that simulates the atmospheric escape of extrasolar planets and their evolution.

It can evolve the gaseous atmosphere of a planet backwards and forwards in time, taking into account its internal structure and cooling rate, atmospheric mass loss processes, and the stellar emission history.

## Minimal example
```python
import photoevolver as ph
import Mors as mors # For the stellar model
import matplotlib.pyplot as plt

planet = ph.Planet(mass = 5.0, radius = 2.0, period = 5.0)
#                  M_earth,    R_earth,      days

planet.set_models(
    core  = ph.models.core_otegi20,
    env   = ph.models.envelope_chen16,
    mloss = ph.models.massloss_energy_limited,
    star  = mors.Star(Mstar = 1.0, percentile = 50.0)
)

# Returns pandas dataframe
evo = planet.evolve(
    start = 10.0, end = 1000.0, step = 1.0, # Myr
    progressbar = True
)

plt.plot(evo['age'], evo['radius'])
plt.xlabel("Age (Myr)")
plt.ylabel("Radius (Earth radii)")
plt.show()
```

Check out the folder [`examples`](./examples) for jupyter notebooks with examples.


## Installation

Download and install the repository with pip:
```bash
pip install git+https://github.com/jorgefz/photoevolver.git
```

Install the following dependencies:
```
numpy
uncertainties
scipy
astropy
pandas
tqdm
```
Alternatively, run `pip install -r requeriments.txt` to install them automatically.

Import the module and use it straight away:
```python
>>> import photoevolver as ph
```

## Scientific background

In the past few decades, thousands of extrasolar planets have been discovered.
Many of these planets are small, between Earth and Neptune in size, and orbit very close to their stars, well within the orbit of Mercury in the Solar System.
As a result, these planets receive large amounts of radiation from their stars, particularly X-rays and UV, which can "erode" the large atmospheres of these planets.
Over time, some of these planets can even be completely stripped of their atmospheres, down to barren rocky worlds.

`photoevolver` simulates the atmospheric escape from these planets across their lifetime and determines whether their atmospheres survive or are completely stripped by radiation from their host star.
See [Fernández Fernández et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.tmp.1197F/abstract) for more information.


## Credit

This repository uses datasets and models from the following sources:

* Mass loss model by [Kubyshkina & Fossati (2021)](https://ui.adsabs.harvard.edu/abs/2021RNAAS...5...74K/abstract), whose grid is available at this [Zenodo repository](https://zenodo.org/record/4643823), which is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International Public License](https://creativecommons.org/licenses/by-nc/4.0/).
* Envelope structure model by [Owen & Campos Estrada (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.5287O/abstract), available at this [GitHub repository](https://github.com/jo276/EvapMass).
* Stellar sequences by Eric Mamajek, available at this [link](http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt).
* Internal structure models by [Zeng et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019PNAS..116.9723Z/abstract) available at this [link](https://lweb.cfa.harvard.edu/~lzeng/planetmodels.html).
* Stellar evolution models by [Spada et al. (2013)]()

For the stellar emission history, it is recommended to use the models by [Johnstone et al. (2021)](https://ui.adsabs.harvard.edu/abs/2020arXiv200907695J/abstract), which you can install from this [GitHub repository](https://github.com/ColinPhilipJohnstone/Mors).


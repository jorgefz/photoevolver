# photoevolver v2.0

`photoevolver` is a Python module that evolves the gaseous envelope of planets backwards and forward in time, taking into account internal structure and cooling rate, atmospheric mass loss processes, and the stellar X-ray emission history.

## Minimal example
```python
import photoevolver as ph
import Mors as mors # For the stellar model
import matplotlib.pyplot as plt

planet = ph.Planet(mass = 5.0, radius = 2.0, period = 5.0)
#                  M_earth,    R_earth,      days

planet.set_models(
    core_model      = ph.models.core_otegi20,
    star_model      = mors.Star(Mstar = 1.0, percentile = 50.0),
    envelope_model  = ph.models.envelope_chen16,
    mass_loss_model = ph.models.massloss_energy_limited
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

See the jupyter notebooks on the folder `examples` for more information.


## Installation

Simply install the repository directly with pip:
```bash
pip install git+https://github.com/jorgefz/photoevolver.git
```

You can now import the module and use it straight away:
```python
>>> import photoevolver as ph
```


## Scientific Background

The Kepler telescope has found a surprising number of planets between the Earth and Neptune in size. These planets follow a bimodal (two-peaked) distribution in radii, with peaks at 1.5 and 2.4 Earth radii, and a *radius valley* at 1.8 Earth radii. The first peak is most likely populated by entirely rocky planets, whereas the second one consists of rocky cores with H/He-rich envelopes that can double the planet's radius and yet comprise only a few percent of its mass. This radius valley would be created by atmospheric mass loss processes that evaporate planetary atmospheres over time, completely stripping them in extreme cases (Fulton et al. 2017).

One such mechanism is *photoevaporation*, where stellar X-rays are readily absorbed by the upper layers of planetary atmospheres, which expand to the Roche lobe and thus escape. Stellar X-rays are not constant, but tied to the star's rotation period. Faster rotators produce a greater X-ray emission. Stars also spin down with time, reducing their X-ray output by an order of magnitude within the first 100 Myr of age (Wright et al. 2011).

A description of a planet's evaporation past would thus require the planet's internal structure (and current amount of gas), the X-ray emission history of the host star, and a mass loss model, which relates the amount of input energy from X-rays to the resulting mass loss rate from the planet.

## Dependencies, useful modules, and credits

### Mors (optional)
Stellar tracks code by Colin Johnstone (https://github.com/ColinPhilipJohnstone/Mors) described in the paper by Johnstone et al (2020).

### EvapMass (already included)
Thermal evolution code by James Owen (https://github.com/jo276/EvapMass), described in the paper Owen & Campos Estrada (2020). 
This is already included in this repository, so there's no need to install it.

### Kubyshkina & Fossati (2021) grid and interpolator (already included)
Updated grid of planet parameters and mass loss rates based on the
hydrodynamic simulations by Kubyshkina et al. (2018).
Reference: https://ui.adsabs.harvard.edu/abs/2021RNAAS...5...74K/abstract).
Zenodo repository: https://zenodo.org/record/4643823
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License (https://creativecommons.org/licenses/by-nc/4.0/).


## References

* [Johnstone et al. (2021)](https://ui.adsabs.harvard.edu/abs/2020arXiv200907695J/abstract)
* [Lopez & Fortney (2014)](https://ui.adsabs.harvard.edu/abs/2014ApJ...792....1L/abstract)
* [Chen & Rogers (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...831..180C/abstract)
* [Owen & Wu (2017)](https://ui.adsabs.harvard.edu/abs/2017ApJ...847...29O/abstract)
* [Lecavelier des Etangs (2007)](https://ui.adsabs.harvard.edu/abs/2007A&A...461.1185L)
* [Erkaev et al. (2007)](https://ui.adsabs.harvard.edu/abs/2007A%26A...472..329E/abstract)
* [Kubyshkina et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...866L..18K/abstract)
* [Wright et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011ApJ...743...48W/abstract)
* [Fulton et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017AJ....154..109F/abstract)
* [Owen & Campos Estrada (2021)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.5287O/abstract)
* [Otegi et al. (2021)](https://ui.adsabs.harvard.edu/abs/2020A%26A...634A..43O/abstract)
* [Salz et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016A%26A...585L...2S/abstract)
* [Kubyshkina & Fossati (2021)](https://ui.adsabs.harvard.edu/abs/2021RNAAS...5...74K/abstract)

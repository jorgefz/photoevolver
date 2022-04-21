# photoevolver

`photoevolver` is a Python module that evolves the gaseous envelope of planets backwards and forward in time, taking into account internal structure and cooling rate, atmospheric mass loss processes, and the stellar X-ray emission history.


## Installation

Download this repository somewhere in your machine and navigate to the modules folder
```bash
git clone https://github.com/jorgefz/photoevolver
cd photoevolver
```

Compile the C functions with make
```bash
make
```

Install the Python module with pip
```bash
pip install -e .
```

The module should now be accessible through Python.
```python
>>> import photoevolver as ph
```


## Introduction

The Kepler telescope has found a surprising number of planets between the Earth and Neptune in size. These planets follow a bimodal (two-peaked) distribution in radii, with peaks at 1.5 and 2.4 Earth radii, and a *radius valley* at 1.8 Earth radii. The first peak is most likely populated by entirely rocky planets, whereas the second one consists of rocky cores with H/He-rich envelopes that can double the planet's radius and yet comprise only a few percent of its mass. This radius valley would be created by atmospheric mass loss processes that evaporate planetary atmospheres over time, completely stripping them in extreme cases (Fulton et al. 2017).

The leading theory that explains the atmospheric mass loss is *photoevaporation*, which states that stellar X-rays produced at the corona are readily absorbed by the upper layers of planetary atmospheres, which expand to the Roche lobe and thus escape. Stellar X-rays are not constant, but tied to the star's rotation period. Faster rotators have a stronger internal dynamo mechanism that produces a greater X-ray emission. Stars also spin down with time, reducing their X-ray output by an order of magnitude within the first 100 Myr of age (Wright et al. 2011).

A description of a planet's evaporation past would thus require the planet's internal structure, the X-ray emission history of the host star, and a formulation for the interaction between the atmosphere and X-ray photons that outputs the mass loss rate.

## Dependencies

### Mors (optional)
Stellar tracks code by Colin Johnstone (https://github.com/ColinPhilipJohnstone/Mors) described in the paper by Johnstone et al (2020).

### EvapMass (already included)
Thermal evolution code by James Owen (https://github.com/jo276/EvapMass), described in the paper Owen & Campos Estrada (2020). 
This is already included in this repository, so there's no need to install it.


## Usage

First, import the package
```python
>>> import photoevolver as ph
```

### Planet

Then, create a planet by defining some of its physical quantities. 
You must define the distance to the host star in AU `dist`, its age in Myr `age`, and either the core mass `mcore` or core radius `rcore` in Earth radii, as well as a fourth parameter which can be used in combination with the previous ones to calculate all other parameters.

The input parameters are the following: 
planet mass `mp`, planet radius `rp`, core mass `mcore`, core radius `rcore`, envelope mass fraction `fenv` (defined as `mcore / mp`), envelope mass `menv`, and envelope radius `renv`.

You may specify only either core mass or radius, as the other will be automatically calculated based on a mass-radius relation, which can be specified with the `mr` parameter. The default one is `ph.core.OtegiMassRadiusRelation` (Otegi et al. 2021), but an alternative is `ph.core.OwenMassRadiusRelation` (Owen & Wu 2017) which allows to specify the core composition with the parameters `Xice` for the ice fraction and `Xiron` for the iron fraction (e.g. setting both to zero defines the core composition as 100% rocky, the default is 1/3 iron and 2/3 rocky). 

For instance, here we obtain a planet with a core of 5 Earth masses, an envelope 1% of the planet's mass, 
located 0.1 AU from its host star, and aged 10 Myr.
```python
>>> p = ph.Planet(mcore=5.0, fenv=0.01, dist=0.1, age=10)
Warning: core radius undefined. Will be estimated from a mass-radius relation  -> 1.521 Earth radii
>>> p
<photoevolver.core.Planet object at 0x7f7e7a35de80>

          Radius    Mass
Core      1.52      5.00      [ice:0% iron:33%]
Envelope  TBD       0.05      [1.00%]
Total     TBD       5.05

```
"TBD" on Envelope radius means it hasn't been calculated yet. The planet needs a formulation for its internal structure, 
a way to obtain the envelope radius from its mass. 

Now, we will define these formulations and evolve the planet. The main function to do so is the following.
```python
    ph.evolve(planet, star, struct, mloss, end)
```
where `planet` is the input planet generated above, `star` basically defines the bolometric and XUV luminosities at each point in time, `struct` is a formulation for the structure of the planet's envelope, `mloss` is the atmospheric mass loss formulation, and `end` is the age in Myr at which the simulation ends. Thus, the simulation will run from the input planet are (10 Myr above) to this value, with a default time step of 1 Myr (this can be modified with the input parameter `timestep`).

### Star

First we define the host star using the package Mors, by Colin P. Johnstone. 
Here, we define a star with 0.7 solar masses, and a rotation period of 15 days at an age of 700 Myr.
```python
>>> import Mors as mors
>>> star = mors.Star(Mstar = 0.7, Prot = 15.0, Age = 700.0)
```

Alternatively, one can also manually input the bolometric and XUV luminosity tracks as two arrays. These must be packaged in a python dictionary with keys `Lxuv` and `Lbol` and have a length equal to the number of steps in the simulation ( 1 + (age\_end - age\_start) / time\_step ).
```python
>>> import numpy as np
>>> star = dict(Lxuv = np.linspace(1e30, 1e27, 5001), Lbol = np.linspace(1e30, 1e30, 5001))
```

### Envelope formulation and evaporation

There are three options to choose from for a planetary interior formulation: 
Lopez & Fortney (2014) using `ph.structure.LopezFortney14`, Chen & Rogers (2016) using `ph.structure.ChenRogers16`, and Owen & Wu (2017) using `ph.structure.OwenWu17`. Note that this last one will only make use of the Owen & Wu (2017) mass-radius relations for the computation, so this relation must be chosen when defining the planet (see above).

Additionally, there are two options for a mass loss formulation: the energy-limited approach (Lecavelier des Etangs 2007, Erkaev 2007) with `ph.massloss.EnergyLimited` and the hydro-based formulation by Kubyshkina et al (2018) with `ph.massloss.Kubyshkina18`. The energy-limited one requires the mass of the star to be input with the parameter `mstar` (in solar masses), but this is only necessary if the Mors models are not being used. Additionally, this formulation can be provided with values for the efficiency (parameter `eff`) and effective XUV radius ratio (parameter `beta`). These two parameters can also be set to follow a given formulation by setting them to a function.

Finally, with your choice of formulations, run the evolver :
```python
>>> planet_tracks = ph.evolve(planet=p, star=star, struct=ph.structure.ChenRogers16, mloss=ph.massloss.EnergyLimited, end=5000)
```
This runs the simulation with a default time step of 1 Myr, up to an age of 5 Gyr. It returns a set of arrays, or tracks, describing the evolution of certain planetary parameters, such as mass, radius, envelope mass fraction, and so on.

The planet can also be evolved backwards in time by setting an additional argument `mode = "past"`, which requieres the exact same input parameters, with `end` being the earliest time to which to evolve.

To check the available tracks, print the dictionary keys:
```python
>>> planet_tracks.keys()
dict_keys(['Age', 'Lbol', 'Lxuv', 'Rp', 'Mp', 'Menv', 'Renv', 'Fenv', 'Dens', 'Mloss'])
```
Plot the parameters with age to see how they evolve:
```python
>>> import matplotlib.pyplot as plt
>>> plt.plot(planet_tracks['Age'], planet_tracks['Rp'], 'b-')
>>> plt.xscale('log')
>>> plt.show()
```

You can retrieve the value of each parameter at a given age by using predefined Track methods:
```python
>>> planet_tracks.Rp(100) # Planet radius at the age of 100 Myr
2.2630649874261013
```

Alternatively, you may obtain a snapshot of the simulation at a given point in time using the method `Tracks.Age` and a given age,
or by indexing the tracks with an integer (e.g. `planet_tracks[index]`):
```python
>>> planet_tracks.Age(100) # Simulation state at the age of 100 Myr
{'Lbol': 4.2794057821287445e+32, 'Lxuv': 2.867065743539218e+29,
'Rp': 2.2630649874261013, 'Mp': 5.044313581183174,
'Menv': 0.04431358118317383, 'Renv': 0.6204396180559271,
'Fenv': 0.008784858527145173, 'Dens': 2.3915553474325266,
'Mloss': 4.12629168057713e-05}
```

You can also generate a Planet object at any given age by using the `Track.planet(age)` method,
which you may then feed into another simulation.
```python
>>> p_100 = planet_tracks.planet(100) # return planet at 100 Myr
```

Furthermore, you can retrieve the tracks as a dictionary of arrays with the method `Tracks.as_dict()`.
Saving and loading from or to disk is done with the functions `Tracks.save(filename)` and `Tracks.load(filename)`, respectively.

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


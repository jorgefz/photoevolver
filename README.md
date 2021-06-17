# photoevolver

Evolves a given planet backwards and forward in time, taking into account its internal structure and cooling rate, the energy-limited atmospheric mass loss, and the XUV track of its host star.

## Dependencies

### Mors
Stellar tracks code by Colin Johnstone (https://github.com/ColinPhilipJohnstone/Mors) described in the paper by Johnstone et al (2020)

### EvapMass
Thermal evolution code by James Owen (https://github.com/jo276/EvapMass), described in the paper Owen & Campos Estrada (2020). 
This is already included in this repository, so there's no need to install it.

## Installation

Download this repository somewhere in your machine
```bash
git clone https://github.com/jorgefz/photoevolver
```

Install the Python module with pip
```bash
pip install -e photoevolver
```

The module should now be accessible through python.

## Usage

First, import package submodules
```python
>>> from photoevolver.core import Planet, Evolve
```

Then, create a planet by defining some of its physical quantities. 
You must define at least four of them: core mass, envelope mass fraction, distance to star, and age.
You can also define the core radius 'rcore', but it can also be calculated automatically 
from its mass and composition (default 1/3 iron, 2/3 rock).

Here, we obtain a planet with a core of 5 Earth masses, an envelope 1% of the planet's mass, 
located 0.1 AU from its host star, and aged 10 Myr.
```python
>>> p = Planet(mcore=5.0, fenv=0.01, dist=0.1, age=10)
 Warning: core radius undefined. Will be estimated from its composition (Iron:0.33, Ice:0.00)...
 -> 1.521 Earth radii
>>> p
<photoevolver.core.Planet object at 0x7f7e7a35de80>

          Radius    Mass
Core      1.52      5.00      [ice:0% iron:33%]
Envelope  TBD       0.05      [1.00%]
Total     TBD       5.05

```
"TBD" on Envelope radius means it hasn't been calculated yet. The planet needs a formulation for its internal structure, 
a way to obtain the envelope radius from its mass. 

Now, we will define these formulations and evolve the planet. 
First we define its host star using the package Mors, by Colin P. Johnstone. 
Here, we define a star with 0.7 solar masses and a rotation period of 13.6 days.
```python
>>> import Mors as mors
>>> star = mors.Star(Mstar = 0.7, Prot = 13.6)
```

There are three options to choose from for a planetary interior formulation: 
Lopez & Fortney (2014), Chen & Rogers (2016), and Owen & Wu (2017), all available in this submodule:
```python
>>> from photoevolver.structure import *
```
This command loads the functions LopezFortney14, ChenRogers16, and OwenWu14.

There are two options for a mass loss formulation: energy-limited (Lecavelier des Etangs 2007, Erkaev 2007) 
and Kubyshkina et al (2018), obtained with the following submodule:
```python
>>> from photoevolver.massloss import *
```
This loads the functions EnergyLimited and Kubyshkina18.

Finally, with your choice of formulations, run the evolver:
```python
>>> planet_tracks = Evolve.forward(planet=p, mloss=EnergyLimited, struct=ChenRogers16, star=star)
```
This runs the simulation with a default time step of 1 Myr, up to an age of 10 Gyr. 
These can be tweaked with the input keywords 'time\_step' and 'age\_end'. 
It returns a dictionary 'planet\_tracks' with a set of arrays, tracks, describing 
the evolution of some planetary parameters (mass, radius, etc), which can be plotted. 
To see the available tracks, print the dictionary keys:
```python
>>> planet_tracks.keys()
dict_keys(['Age', 'Lbol', 'Rp', 'Mp'])
```

Plot the parameters with age to see how they evolve:
```python
>>> import matplotlib.pyplot as plt
>>> plt.plot(planet_tracks['Age'], planet_tracks['Rp'], 'b-')
>>> plt.show()
```





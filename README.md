# photoevolver

Evolves a given planet backwards and forward in time, taking into account its internal structure and cooling rate, the energy-limited atmospheric mass loss, and the XUV track of its host star.

# Installation

Download this repository somewhere in your machine
```bash
git clone https://github.com/jorgefz/photoevolver
```

Install the Python module with pip
```bash
pip install -e photoevolver
```

The module should now be accessible through python.

# Making a planet

```python
>>> import photoevolver.photoevolver as ph
```


```python
>>> import Mors as mors
>>> star = mors.Star(Mstar = 0.7, Prot = 13.6)
```


```python
>>> planet = ph.Planet(dist = 0.50, age = 700, mstar = star.Mstar, Star = star, core_mass = 1.0, core_radius = 1.0, env_mass = 0.01)
>>> planet.generateTracks()
```

```python
>>> planet.AgeTrack
>>> planet.LxuvTrack
```

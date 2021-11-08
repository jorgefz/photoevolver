import matplotlib.pyplot as plt
import photoevolver as ph
import Mors as mors

star = mors.Star(Mstar=0.66, Age=700, Prot=15)
p = ph.core.Planet(age=700, dist=0.15, mcore=1.3, fenv=0.0)

future = ph.core.evolve_forward(planet=p, star=star, mloss=ph.massloss.EnergyLimited, struct=ph.structure.ChenRogers16, age_end=5000)
past = ph.core.evolve_back(planet=p, star=star, mloss=ph.massloss.EnergyLimited, struct=ph.structure.ChenRogers16, age_end=20)

print(future.planet(700,p))
print(f"Future starting params: Rp {future['Rp'][-1]}, Renv {future['Renv'][-1]}, Fenv {future['Fenv'][-1]}%")
print(f"Past starting params: Rp {past['Rp'][-1]}, Renv {past['Renv'][-1]}, Fenv {past['Fenv'][-1]}%")

# Plotting radii

plt.xlabel("Age")
plt.ylabel("R")

plt.plot(future['Age'], future['Rp'], 'b-')
plt.plot(future['Age'], future['Renv'], 'b--')

plt.plot(past['Age'], past['Rp'], 'r-')
plt.plot(past['Age'], past['Renv'], 'r--')
plt.plot(past['Age'], [p.rcore] * len(past['Age']), 'k:')

plt.show()


# Plotting masses

plt.xlabel("Age")
plt.ylabel("M")

plt.plot(future['Age'], future['Mp'], 'b-')
plt.plot(past['Age'], past['Mp'], 'r-')
plt.plot(past['Age'], [p.mcore] * len(past['Age']), 'k:')

plt.show()

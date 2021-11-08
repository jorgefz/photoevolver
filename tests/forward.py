import matplotlib.pyplot as plt
import photoevolver as ph
import Mors as mors

star = mors.Star(Mstar=0.66, Age=700, Prot=15)
p = ph.core.Planet(age=700, dist=0.15, rp=1.5, mcore=1.3)

future = ph.core.evolve_forward(planet=p, star=star, mloss=ph.massloss.EnergyLimited, struct=ph.structure.ChenRogers16, age_end=5000)
past = ph.core.evolve_back(planet=p, star=star, mloss=ph.massloss.EnergyLimited, struct=ph.structure.ChenRogers16, age_end=20)

print(future.planet(700,p))
print(f"Future starting params: Rp {future['Rp'][0]}, Renv {future['Renv'][0]}, Fenv {future['Fenv'][0]}%")
print(f"Past starting params: Rp {past['Rp'][-1]}, Renv {past['Renv'][-1]}, Fenv {past['Fenv'][-1]}%")

# Plotting radii

plt.xlabel("Age")
plt.ylabel("R")

plt.plot(past['Age'], past['Rp'], 'b-')
plt.plot(past['Age'], past['Renv'], 'b--')
plt.plot(past['Age'], [p.rcore] * len(past['Age']), 'k:')

plt.plot(future['Age'], future['Rp'], 'r-')
plt.plot(future['Age'], future['Renv'], 'r--')
plt.plot(future['Age'], [p.rcore] * len(future['Age']), 'k:')

plt.show()


# Plotting masses

plt.xlabel("Age")
plt.ylabel("M")

plt.plot(past['Age'], past['Mp'], 'b-')
#plt.plot(past['Age'], past['Menv'], 'b--')
plt.plot(past['Age'], [p.mcore] * len(past['Age']), 'k:')

plt.plot(future['Age'], future['Mp'], 'r-')
#plt.plot(future['Age'], future['Menv'], 'r--')
plt.plot(future['Age'], [p.mcore] * len(future['Age']), 'k:')

plt.show()

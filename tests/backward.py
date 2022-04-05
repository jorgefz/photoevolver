import matplotlib.pyplot as plt
import photoevolver as ph
import Mors as mors

#star = mors.Star(Mstar=0.66, Age=700, Prot=15)
#star.Save("star.pickle")

star = mors.Load("star.pickle")

p = ph.Planet(age=700, dist=0.15, mcore=1.3, rp=1.48)

future = ph.evolve_forward(planet=p, star=star, mloss=ph.massloss.EnergyLimited, struct=ph.structure.ChenRogers16, age_end=5000)
past = ph.evolve_back(planet=p, star=star, mloss=ph.massloss.EnergyLimited, struct=ph.structure.ChenRogers16, age_end=20)

print(future.planet(700,p))
print(f"Future starting params: Rp {future['Rp'][-1]}, Renv {future['Renv'][-1]}, Fenv {future['Fenv'][-1]}%")
print(f"Past starting params: Rp {past['Rp'][-1]}, Renv {past['Renv'][-1]}, Fenv {past['Fenv'][-1]}%")

# Plotting radii

plt.xlabel("Age")
plt.ylabel("R")

plt.plot(future['Age'], future['Rp'], 'b-')
plt.plot(past['Age'], past['Rp'], 'r-')
plt.axhline(p.rp, linestyle=':', color='k')
plt.axvline(p.age, linestyle=':', color='k')

plt.show()


# Plotting masses

plt.xlabel("Age")
plt.ylabel("M")

plt.plot(future['Age'], future['Mp'], 'b-')
plt.plot(past['Age'], past['Mp'], 'r-')
plt.axvline(p.age, linestyle=':', color='k')

plt.show()

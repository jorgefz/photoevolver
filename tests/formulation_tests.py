
import photoevolver as ph
import astropy.constants as Const
import numpy as np
import Mors as mors
import matplotlib.pyplot as plt
import scipy


def Fearth_to_Fxray(fe):
    """
    Converts incident flux in Earth fluxes received from the Sun,
    to X-ray style fluxes in erg/cm^2/s.
    """
    F_earth = Const.L_sun / (4 * np.pi * (Const.au)**2)
    return fe * F_earth.to("erg * cm^-2 * s^-1").value


def CheckValues(x, y):
    print(f"{x:.5g}".ljust(15) + f"== {y:.5g}")


def CheckFearth():
    print(" ")
    print(" Checking F_earth to F_xray...")
    CheckValues( 1.36e6,       Fearth_to_Fxray(1)   ) # value from Wolfram Alpha
    CheckValues( 1.36e6 * 0.1, Fearth_to_Fxray(0.1) ) # value from Wolfram Alpha
    CheckValues( 1.36e6 * 10,  Fearth_to_Fxray(10), ) # value from Wolfram Alpha
    CheckValues( 1.36e6 * 1000,  Fearth_to_Fxray(1000), ) # value from Wolfram Alpha

def CheckLopezFortney14():
    print(" ")
    print(" Checking Lopez & Fortney 2014 structure formulation...")
    """
    ph.structure.LopezFortney14
    Requires:
        - mass: mass of the planet (M_earth)
        - fenv: envelope mass fraction (Menv/Mplanet)
        - fbol: bolometric incident flux at planet distance (erg/cm^2/s)
        - age: age of the planet (Myr)
    Returns:
        - Envelope radius (R_earth)
    """
    func = ph.structure.LopezFortney14

    # ----- Age 100 Myr -----
    CheckValues( 1.21, 1 + func(age=100, fbol=Fearth_to_Fxray(0.1), mass=1.0, fenv=0.1/100) )
    CheckValues( 2.17, 1 + func(age=100, fbol=Fearth_to_Fxray(0.1), mass=1.0, fenv=1.0/100) )
    CheckValues( 1.69, 1 + func(age=100, fbol=Fearth_to_Fxray(0.1), mass=5.5, fenv=0.1/100) )
    CheckValues( 2.28, 1 + func(age=100, fbol=Fearth_to_Fxray(0.1), mass=5.5, fenv=1.0/100) )

    CheckValues( 1.31, 1 + func(age=100, fbol=Fearth_to_Fxray(10), mass=1.0, fenv=0.1/100) )
    CheckValues( 2.40, 1 + func(age=100, fbol=Fearth_to_Fxray(10), mass=1.0, fenv=1.0/100) )
    CheckValues( 1.73, 1 + func(age=100, fbol=Fearth_to_Fxray(10), mass=5.5, fenv=0.1/100) )
    CheckValues( 2.33, 1 + func(age=100, fbol=Fearth_to_Fxray(10), mass=5.5, fenv=1.0/100) )
   
    CheckValues( 1.96, 1 + func(age=100, fbol=Fearth_to_Fxray(1000), mass=1.0, fenv=0.1/100) )
    CheckValues( 2.18, 1 + func(age=100, fbol=Fearth_to_Fxray(1000), mass=1.0, fenv=1.0/100) )
    CheckValues( 2.04, 1 + func(age=100, fbol=Fearth_to_Fxray(1000), mass=5.5, fenv=0.1/100) )
    CheckValues( 2.38, 1 + func(age=100, fbol=Fearth_to_Fxray(1000), mass=5.5, fenv=1.0/100) )


def CheckPoppenhaeger2020():
    print(" ")
    print(" Checking against Poppenhaeger 2020 planets...")

    # Plotting stellar tracks
    star = mors.Star(Mstar=1.1, Prot=2.87, Age=23)

    # Generate similar star tracks to what Poppenhaeger 2020 used
    # only high activity one

    def high_activity_track():
        """
        Takes an age and returns the X-rays following the High Activity track
        """
        x = [23, 300, 1000, 5000]
        y = [1.5e30, 1.5e30, 2e28, 2e27]
        f = scipy.interpolate.interp1d(x, y)
        return f

    ages = np.arange(23, 5001)
    xrays = high_activity_track()
    star_data = dict()
    star_data['Lxuv'] = np.array([xrays(a) for a in ages])
    star_data['Lbol'] = np.array([star.Lbol(a) for a in ages])
    
    plt.title("Poppenhaeger 2020 tracks VS Johnstone 2021 track")
    plt.xlabel("Time [Myr]")
    plt.ylabel("Lx [erg/s]")
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(star.AgeTrack, star.LxTrack, 'm--')
    plt.plot( [23, 300, 1000], [1.5e30, 1.5e30, 2e28], 'b-') # high activity track
    plt.plot( [23,  45, 1000], [1.5e30, 0.8e29, 2e28], 'r-') # low activity track
    plt.plot( [23, 1000, 5000], [1.5e30, 2e28, 2e27], color='tab:gray', linestyle='-') # medium track
    plt.plot(23, 1.5e30, 'y*', markersize=12, mec='k') # starting point
    plt.plot(5000, 2e27, 'k*', markersize=12) # ending point
    plt.plot(ages, star_data['Lxuv'], 'm:') # test custom high track
    plt.show()


    # Modelling planetary tracks
    planets = ('c', 'd', 'b', 'e')
    radii = (5.6, 6.4, 10.3, 8.7)
    dists = (0.0825, 0.1083, 0.1688, 0.308)
    p_light = [ ph.Planet(age=23, dist=dists[i], mcore=5,  rp=radii[i]) for i in range(4) ]
    p_heavy = [ ph.Planet(age=23, dist=dists[i], mcore=10, rp=radii[i]) for i in range(4) ]
     
    LF14 = ph.structure.LopezFortney14
    EL = ph.massloss.EnergyLimited

    t_light = []
    for p in p_light:
        #t = ph.evolve_forward(planet=p, star=star, struct=LF14, mloss=EL, eff=0.1, age_end=5000)
        t = ph.evolve_forward(planet=p, star=star_data, struct=LF14, mloss=EL,
                eff=0.1, age_end=5000, mstar=star.Mstar, beta='salz16')
        t_light.append(t)

    t_heavy = []
    for p in p_heavy:
        #t = ph.evolve_forward(planet=p, star=star, struct=LF14, mloss=EL, eff=0.1, age_end=5000)
        t = ph.evolve_forward(planet=p, star=star_data, struct=LF14, mloss=EL,
                eff=0.1, age_end=5000, mstar=star.Mstar, beta='salz16')
        t_heavy.append(t)

    # Test plot
    plt.title(f" Planet {planets[0]}")
    plt.xlabel("Time [Myr]")
    plt.ylabel(r"R [R$_{\oplus}$]")
    plt.xscale('log')
    plt.plot(t_light[0]['Age'], t_light[0]['Rp'], 'b-', label='5 Me core')
    plt.plot(t_heavy[0]['Age'], t_heavy[0]['Rp'], 'r-', label='10 Me core')
    plt.legend()
    plt.show()

    print(" Radii at 5 Gyr ")
    print(" Planet".ljust(10)+" My R".ljust(10)+" high".ljust(10)+"medium".ljust(10)+"low".ljust(10))
 
    # final radii (fr) + light (l) or heavy (h)
    frl = ((1.5,1.5,1.5), (1.5,1.5,2.0), (1.5,3.3,4.8), (3.2,4.8,5.2))
    frh = ( (1.8,2.4,3.0), (1.9,3.3,3.8), (5.4,6.2,6.4), (5.3,5.6,5.6))

    for i in range(len(planets)):
        print(f"{planets[i]}-light".ljust(10), end='')
        print(f"{t_light[i]['Rp'][-1]:.3g}".ljust(10), end='')
        for j in range(3): print(f"{frl[i][j]:.3g}".ljust(10), end='')
        print("")

        print(f"{planets[i]}-heavy".ljust(10), end='')
        print(f"{t_heavy[i]['Rp'][-1]:.3g}".ljust(10), end='')
        for j in range(3): print(f"{frh[i][j]:.3g}".ljust(10), end='')
        print("")


def CheckMassLoss():
    print(" ")
    print("Checking Energy Limited formulation...")
    
    EL = ph.massloss.EnergyLimited
    K18 = ph.massloss.Kubyshkina18

    # Comparing to results from King 2019 (pi Men c)
    CheckValues(0.11, 1e-10 * 1.894e14 * EL(mass=4.52, radius=2.06, dist=0.067, mstar=1.02,
        Lxuv=2.27e28 , eff=0.15, beta=1.0)) 

    CheckValues(1.5, 1e-10 * 1.894e14 * EL(mass=4.52, radius=2.06, dist=0.067, mstar=1.02,
        Lxuv=2.27e28 , eff=0.15, beta=2.67))
 
    CheckValues(2.8, 1e-10 * 1.894e14 * K18(mass=4.52, radius=2.06, dist=0.067,
        Lbol=5.191e33, Lxuv=2.27e28))

    # Comparing to the plots on Kubyshina 2018
    flux2lum = lambda F,d: F * 4 * np.pi * (d * 1.496e13)**2
    Lxuv = flux2lum(F=10, d=0.03)
    Lbol = Const.L_sun.to("erg / s").value
    masses = np.arange(1,40)

    small_mloss = [ 1.894e14 * K18(mass=m, radius=3, dist=0.03, Lxuv=flux2lum(10,0.03),
            Lbol=Lbol) for m in masses]

    large_mloss = [ 1.894e14 * K18(mass=m, radius=3, dist=0.1, Lxuv=flux2lum(1e4,0.1),
            Lbol=Lbol) for m in masses]

    plt.title("Kubyshkina 2018")
    plt.xlabel("M")
    plt.ylabel("log10 Mdot")
    plt.plot(masses, np.log10(small_mloss), 'r-')
    plt.plot(masses, np.log10(large_mloss), 'b:')
    plt.show()


def CheckChenRogers16():
    
    masses = np.linspace(4, 20, 1000)
    fenv = np.array((25, 20, 15, 10, 5, 2, 1, 0.05, 0.01)) / 100
    dist = 0.1 # AU
    age = 300 # Gyr
    fbol = Const.L_sun.to("erg/s").value / (4 * np.pi * (dist * Const.au.to("cm").value)**2)
    rcore = 0.0
    
    Renv = ph.structure.ChenRogers16

    radii_tracks = []
    for f in fenv:
        track = np.array([ rcore + Renv(mass=m, fenv=f, age=age, fbol=fbol) for m in masses])
        radii_tracks.append(track)

    plt.title("Chen and Rogers 2016")
    plt.xlabel("Mass")
    plt.ylabel("Radius")
    for t in radii_tracks:
        plt.plot(masses, t, 'k-')
    plt.show()



def CheckOwenWu17():
    pass


def main():
    print(" Control == Output")
    #CheckFearth()
    #CheckMassLoss()
    #CheckLopezFortney14()
    #CheckPoppenhaeger2020()
    CheckChenRogers16()
    CheckOwenWu17()
    print(" ")
    print(" Done!")


if __name__ == '__main__':
    main()
    exit()

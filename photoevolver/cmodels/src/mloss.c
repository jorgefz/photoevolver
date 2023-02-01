#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "models.h"

static double pow10(double x){
    return pow(x, 10.0);
}

/*
 * Calculates beta parameter following hydrodynamic simulations
 * by Salz et al. (2016).
 * Beta is the ratio between the optical and XUV absoprtion radii of a planet
 * undergoing atmospheric mass loss.
 * Input:
 *      double fxuv:    XUV flux in erg/cm^2/s
 *      double mass:    Mass of the planet in Earth masses
 *      double radius:  Radius of the planet in Earth radii
 */
double BetaSalz16(double fxuv, double mass, double radius){
    double potential, lg_beta;
    potential = G_CGS_ERG * mass * MEARTH * 1000.0 /* to grams */ / (radius * REARTH);
    lg_beta = -0.185 * log10(potential) + 0.021 * log10(fxuv) + 2.42;
    if (lg_beta < 0.0) lg_beta = 0.0;
    //else if (lg_beta > log10(2.0)) lg_beta = log10(2.0);
    else if (lg_beta > log10(1.05) && potential < 1e11) lg_beta = log10(1.05);
    return pow10(lg_beta);
}


/*
 * Calculates evaporation efficiency following the hydrodynamic simulations
 * by Salz et al. (2016).
 * Input:
 *      double mass: planet mass in Earth masses
 *      double radius: planet radius in Earth radii
 */
double EffSalz16(double mass, double radius){
    double lg_gp, lg_eff;
    lg_gp = log10( G_CGS_ERG * mass * MEARTH / (radius * REARTH) );
    if      (lg_gp < 12.00)                     lg_eff = log10(0.23);
    else if (lg_gp > 12.00 && lg_gp <= 13.11)   lg_eff = -0.44 * (lg_gp-12.00) - 0.5;
    else if (lg_gp > 13.11 && lg_gp <= 13.60)   lg_eff = -7.29 * (lg_gp-13.11) -0.98;
    else                                        lg_eff = -7.29*(13.60-13.11) - 0.98; // stable against evaporation
    return pow10(lg_eff) * 5.0/4.0; // correction for converting evaporation efficiency to heating efficiency.
}


/*
 * Calculates energy-limited atmospheric mass loss rate.
 * Input:
 *      double fxuv: XUV flux in erg/cm^2/s
 *      double radius: planet radius in Earth radius
 *      double mass: planet mass in Earth mass
 *      double mstar: stellar mass in solar masses
 *      double a: planet to star distance in AU
 *      double beta:
 *      double eff:
 * Returns:
 *      double mloss: mass loss rate in grams/s
 */
double EnergyLimitedMassloss(
        double fxuv, double radius, double mass,
        double mstar, double a, double beta, double eff
    ){
    double xi, ktide, mloss;
    // unit conversions
    fxuv *= FX2SI;
    radius *= REARTH;
    mass *= MEARTH;
    mstar *= MSUN;
    a *= AU2M;
    // equation
    xi = (a / radius) * pow( mass/mstar/3.0, 1.0/3.0 );
    ktide = 1.0 - 3.0/(2.0*xi) + 1.0/(2.0*pow(xi,3.0));
    mloss = beta*beta * eff * M_PI * fxuv * pow(radius,3.0) / (G * ktide * mass);
    return mloss * 1e3; // convert kg/s to gram/s
}

static double K18Epsilon(double radius, double fxuv, double a){
    double numerator, denominator;
    numerator = 15.611 - 0.578*log(fxuv) + 1.537*log(a) + 1.018*log(radius);
    denominator = 5.564 + 0.894 * log(a);
    return numerator / denominator;
}

double Kubyshkina18Massloss(double fxuv, double fbol, double mass, double radius, double a){
    double alpha[3], beta, theta, zeta;
    double Teq, mH, Jeans, eps;
    double kappa, mloss;
    
    // Unit conversions
    //mass *= MEARTH;
    // NOTE: 'fxuv' stays in erg/cm^2s and 'a' in AU.

    Teq = pow( FX2SI * fbol / (4*SIGMASB), 1.0/4.0);
    mH = HMASS;
    Jeans = G * mass * MEARTH * mH / (KBOLTZ * Teq * radius * REARTH);
    eps = exp( K18Epsilon(radius, fxuv, a) );
    
    if (Jeans > eps){
        alpha[0] = 1.0; alpha[1] = -3.2861; alpha[2] = 2.75;
        beta = 16.4084;
        zeta = -1.2978;
        theta = 0.8846;
    } else {
        alpha[0] = 0.4222; alpha[1] = -1.7489; alpha[2] = 3.7679;
        beta = 32.0199;
        zeta = -6.8618;
        theta = 0.0095;
    }
    
    kappa = zeta + theta * log(a); 
    mloss = exp(beta) * pow(fxuv,alpha[0]) * pow(a,alpha[1]) * pow(radius,alpha[2]) * pow(Jeans,kappa);
    return mloss;
}







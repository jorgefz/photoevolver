
#ifndef CONSTANTS_H
#define CONSTANTS_H 1

#include <math.h>

#define G 6.674e-11
#define G_CGS 6.674e-8
#define G_CGS_ERG 6.674e-10

#define SI2FX 1e3 // Watt/m^2 to erg/cm^2/s
#define FX2SI 1.0/SI2FX
#define SI2LX 1e7 // Watt to erg/s
#define LX2SI 1.0/SI2LX

#define AU2M 1.496e11
#define MYR2SEC 3.154e13

#define MEARTH 5.97e24 // kg
#define REARTH 6.371e6 // m
#define MSUN 1.988e30 // kg
#define LSUN 3.848e26 // W
#define FBOLEARTH (LSUN/(4.0*M_PI*AU2M*AU2M)) // W / m^2

#define SIGMASB 5.670374e-8
#define HMASS 1.67356e-27
#define KBOLTZ 1.38064e-23

#define SIGMASB_CGS 5.670374e-5
#define HMASS_CGS (1.67356e-27*1e3)
#define KBOLTZ_CGS 1.38064e-16

#endif

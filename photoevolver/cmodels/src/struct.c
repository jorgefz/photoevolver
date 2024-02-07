#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <Python.h>
#include "constants.h"
#include "models.h"

const char docs_envelope_lopez14[] = 
    "Returns the envelope thickness in Earth radii using the model\n"
    "by Lopez & Fortney (2014).\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "mass   : float, planet mass in Earth masses\n"
    "fenv   : float, envelope mass fraction = (mass-mcore)/mcore\n"
    "fbol   : float, bolometric flux on the planet (erg/s/cm^2)\n"
    "age    : float, system age in Myr\n"
    "opaque : bool (optional), enables enhanced opacity\n"
    "\n"
    "Returns\n"
    "-------\n"
    "renv   : float, envelope thickness in Earth radii\n"
;
PyObject* envelope_lopez14(PyObject* self, PyObject* args, PyObject* kwargs){
    // Arguments
    double mass, fenv, fbol, age;
    int enhanced_opacity = 1; // optional
    // Keyword arguments
    static char* kwlist[] = {"mass", "fenv", "fbol", "age", "opaque", NULL};
    // Parse args
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "dddd|p", kwlist,
        &mass, &fenv, &fbol, &age, &enhanced_opacity)
    ){
        return NULL;
    }

    double age_pow, mass_term, flux_term, age_term, fenv_term;
    if (enhanced_opacity == 0) age_pow = -0.11; // solar metallicity
    else age_pow = -0.18; // enhanced opacity
    
    mass_term = 2.06 * pow(mass, -0.21);
    flux_term = pow( fbol * FX2SI / FBOLEARTH, 0.044 );
    age_term = pow( age / 5000.0, age_pow );
    fenv_term = pow(fenv / 0.05, 0.59);
    
    double result = mass_term * fenv_term * flux_term * age_term;
    return PyFloat_FromDouble(result);
}


const char docs_envelope_chen16[] = 
    "Returns the envelope thickness in Earth radii using the model\n"
    "by Chen & Rogers (2016).\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "mass   : float, planet mass in Earth masses\n"
    "fenv   : float, envelope mass fraction = (mass-mcore)/mcore\n"
    "fbol   : float, bolometric flux on the planet (erg/s/cm^2)\n"
    "age    : float, system age in Myr\n"
    "\n"
    "Returns\n"
    "-------\n"
    "renv   : float, envelope thickness in Earth radii\n"
;
PyObject* envelope_chen16(PyObject* self, PyObject* args, PyObject* kwargs){
    // Arguments
    double mass, fenv, fbol, age;
    // Keyword arguments
    static char* kwlist[] = {"mass", "fenv", "fbol", "age", NULL};
    // Parse args
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "dddd", kwlist,
        &mass, &fenv, &fbol, &age)
    ){
        return NULL;
    }

    double c0 = 0.131;
    double c1[4] = {-0.348, 0.631, 0.104, -0.179};
    double c2[4][4] = {
        {0.209, 0.028, -0.168, 0.008 },
        {NAN,   0.086, -0.045, -0.036},
        {NAN,   NAN,   0.052,  0.031 },
        {NAN,   NAN,   NAN,    -0.009}
    };

    double terms[4] = {
        log10( mass ),
        log10( fenv/0.05 ),
        log10( fbol * FX2SI / FBOLEARTH ),
        log10( age/5000.0 )
    };

    double lg_renv = c0;
    unsigned int i, j, length = sizeof(terms)/sizeof(double);
    
    for (i=0; i!=length; ++i) lg_renv += terms[i] * c1[i];
    
    for(i=0; i!=length; ++i){
        for(j=0; j!=length; ++j){
            if (j < i) continue;
            lg_renv += c2[i][j] * terms[i] * terms[j];
        }
    }
    double result = pow(10.0, lg_renv);

    return PyFloat_FromDouble(result);
}



/*
===== Functions for solving Owen & Wu (2017) envelope model =====
*/

/*
Finds the roots of a function numerically using Newton's method.
Input:
    double func: function to solve, fpointer of signature double(double x, void* args)
    double deriv: function derivative, fpointer of signature double(double x, void* args)
    double xguess: initial guess for the solution
    unsigned int n: number of iterations to perform
    void* args: arguments to be passed to the functions
Returns
    double xsol: solution for x such that f(x) = 0
*/
static double newton_solve(double (*func)(double,void*), double (*deriv)(double,void*), double xguess, unsigned int n, void* args){
    unsigned int i;
    double xn = xguess;
    for(i=0; i!=n; ++i){
        xn = xn - func(xn,args) / deriv(xn,args);
    }
    return xn;
}

/*
Finds the deriative of a function evaluated at x,
using Newton's symmetric difference quotient method.
Input:
    double func: function to differentiate of signature double(double x, void* args)
    double x: value to which evaluate the derivative
    void* args: arguments to pass onto function
Returns:
    double: value of the derivative at x
*/
static double sdq_derivative(double (*f)(double,void*), double x, void* args){
    double h, eps = 1e-15;
    h = sqrt(eps) * x;
    return (f(x+h,args) - f(x-h,args))/(2.0*h);
}


/*
Finds the definite integral of a function from points a to b,
using Simpson's composite 3/8 rule.
Input:
    double f: function to integrate of signature double(double x, void* args)
    double a: lower integration limit
    double b: upper integration limit
    unsigned int n: number of iterations
    void *args: arguments to pass onto function
*/
static double simpson_integral(double (*f)(double,void*), double a, double b, unsigned int n, void* args){
    double xn, df = 0.0, fn, h, c;
    unsigned int i;

    h = (b-a)/(double)n;
    for(i=0; i!=n+1; ++i){
        xn = a + h * (double)i; // calculate node
        fn = f(xn, args);
        if(i == 0 || i == n) c = 1.0;
        else if ((i % 3) == 0) c = 2.0;
        else c = 3.0;
        df += c * fn;
    }
    df *= 3.0 * h / 8.0;
    return df;
}



/* ========== Owen & Wu 2017 functions =========== */

#define INTEGRAL_ITERS 25
#define FSOLVE_ITERS 10

typedef struct params{
    double alpha, beta, gamma, kappa0, mu; 
} OW17Params;

static OW17Params ow17params = {
    .alpha  = 0.68,       // pressure dependence of opacity
    .beta   = 0.45,       // temperature dependence of opacity
    .gamma  = 5.0/3.0,    // ratio of specific heat capacities
    .kappa0 = 4.7863e-8,  // pow(10.0, -7.32), // opacity constant
    .mu     = 2.35 * HMASS_CGS // solar metallicity gas
};


static double integrand1(double x, void* args){
    (void)args;
    return x * pow( 1.0/x-1.0, 1.0/(ow17params.gamma-1.0) );
}


static double integrand2(double x, void* args){
    (void)args;
    return x * integrand1(x, args);
}


static double I1(double Rf){
    double df, a, b;
    unsigned int n = INTEGRAL_ITERS;
    a = 1.0 / (Rf + 1.0);
    b = 1.0;
    df = simpson_integral(integrand1, a, b, n, NULL);
    return df;
}


static double I2(double Rf){
    double df, a, b;
    unsigned int n = INTEGRAL_ITERS;
    a = 1.0 / (Rf + 1.0);
    b = 1.0;
    df = simpson_integral(integrand2, a, b, n, NULL);
    return df;
}


static double equilibrium_temperature(double fbol){
    return pow( fbol * FX2SI / (4.0 * SIGMASB) , 1.0/4.0);
}

static double calc_rho_rcb(double renv, double xenv, double mcore, double rcore, double Teq, double Tkh){
    double rho_rcb, Rrcb, Rf, ratio, numerator, denominator;
    Rrcb = renv + rcore;
    Rf = renv / rcore;
    ratio = I2(Rf) / I1(Rf);
    Tkh *= MYR2SEC; // Helmholtz timescale to seconds

    numerator = 64.0 * M_PI * ratio * SIGMASB_CGS * pow(Teq,3.0-ow17params.alpha-ow17params.beta) * Rrcb * Tkh;
    denominator = 3.0 * ow17params.kappa0 * (mcore * MEARTH * 1000.0 /* to g */) * xenv;
    rho_rcb = (ow17params.mu / KBOLTZ_CGS) * pow(numerator/denominator, 1.0/(1.0+ow17params.alpha));

    return rho_rcb;
}


static double xenv_wrapper(double renv, void* args){

    double rcore, mcore, xenv, Teq, Tkh;

    rcore = ((double*)args)[0];
    mcore = ((double*)args)[1];
    xenv  = ((double*)args)[2];
    Tkh = ((double*)args)[3];
    Teq = ((double*)args)[4];

    double Rrcb = renv + rcore;
    double Rf = renv / rcore;
    double rho_core = mcore * MEARTH * 1000.0 / (4.0/3.0 * M_PI * pow(rcore,3.0));
    double rho_rcb = calc_rho_rcb(renv, xenv, mcore, rcore, Teq, Tkh);
    double i2 = I2(Rf);
    double cs = KBOLTZ_CGS * Teq / ow17params.mu;
    double pow_term = (ow17params.gamma-1.0)/ow17params.gamma * G_CGS * mcore * MEARTH * 1000.0 / (cs*Rrcb);
    double xeval = 3.0 * i2 * pow(Rrcb/rcore,3.0) * (rho_rcb/rho_core) * pow(pow_term,1.0/(ow17params.gamma-1.0));

    return xeval - xenv;
}

static double xenv_derivative_wrapper(double renv, void* args){
    double deriv = sdq_derivative(xenv_wrapper, renv, args);
    return deriv;
}

static double rho_photosphere(double renv, double rcore, double mcore, double Teq){
    double numerator, denominator, power, pressure_phot, rho_phot;
    numerator = 2.0/3.0 * G_CGS * mcore * MEARTH * 1000.0;
    denominator = pow(renv + rcore, 2.0) * ow17params.kappa0 * pow(Teq, ow17params.beta);
    power = 1.0 / (1.0 + ow17params.alpha);
    pressure_phot = pow(numerator / denominator, power);
    rho_phot = (ow17params.mu / KBOLTZ_CGS) * pressure_phot / Teq;
    return rho_phot;
}

const char docs_envelope_owen17[] = 
    "Returns the envelope thickness in Earth radii using the model\n"
    "by Owen & Wu (2017).\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "mass   : float, planet mass in Earth masses\n"
    "fenv   : float, envelope mass fraction = (mass-mcore)/mcore\n"
    "fbol   : float, bolometric flux on the planet (erg/s/cm^2)\n"
    "age    : float, system age in Myr\n"
    "rcore  : float, core radius in Earth radii\n"
    "\n"
    "Returns\n"
    "-------\n"
    "renv   : float, envelope thickness in Earth radii\n"
;

PyObject* envelope_owen17(PyObject* self, PyObject* args, PyObject* kwargs){

    // Arguments
    double mass, fenv, fbol, age, rcore;
    // Keyword arguments
    static char* kwlist[] = {"mass", "fenv", "fbol", "age", "rcore", NULL};
    // Parse args
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "ddddd", kwlist,
        &mass, &fenv, &fbol, &age, &rcore)
    ){
        return NULL;
    }

    double xenv, mcore, Teq, Tkh;
    mcore = mass * ( 1.0 - fenv );
    xenv = fenv / (1.0 - fenv);
    Teq = equilibrium_temperature(fbol);
    Tkh = age > 100.0 ? age : 100.0;
    rcore *= REARTH * 100.0; // core radius in cm

    double renv_guess = 2.0 * rcore * pow(xenv/0.027, 1.0/1.31) * pow(mcore/5.0, -0.17);
    double solver_args[] = {rcore, mcore, xenv, Tkh, Teq};
    unsigned int n_iter = FSOLVE_ITERS; // iterations
    double renv_sol = newton_solve(xenv_wrapper, xenv_derivative_wrapper,
        renv_guess, n_iter, (void*)solver_args);
    
    //return renv_sol/100.0/REARTH;

    // Density and pressure at the photosphere
    double Rrcb, rho_rcb, rho_phot, Hscale, Ffactor, Rplanet;
    Rrcb = renv_sol + rcore;
    rho_rcb = calc_rho_rcb(renv_sol, xenv, mcore, rcore, Teq, Tkh);
    rho_phot = rho_photosphere(renv_sol, rcore, mcore, Teq);
    Hscale = KBOLTZ_CGS * Teq * pow(Rrcb, 2.0) / (ow17params.mu * G_CGS * mcore * MEARTH * 1000.0);
    Ffactor = 1.0 + (Hscale / Rrcb) * log(rho_rcb / rho_phot);
    Rplanet = Ffactor * Rrcb;
    double result = (Rplanet - rcore) / REARTH / 100.0;

    return PyFloat_FromDouble(result);
}



#include <Python.h>

#ifndef MODELS_H
#define MODELS_H

/*  Envelope Models */

extern const char docs_envelope_lopez14[];
extern const char docs_envelope_chen16[];
PyObject* envelope_lopez14(PyObject* self, PyObject* args, PyObject* kwargs);
PyObject* envelope_chen16(PyObject* self, PyObject* args, PyObject* kwargs);

double ChenRogers16Structure(double mass, double fenv, double fbol, double age);
double OwenWu17Structure(double mass, double fenv, double fbol, double age,
    double rcore);

/*  Mass Loss Models */
double BetaSalz16(double fxuv, double mass, double radius);
double EffSalz16(double mass, double radius);
double EnergyLimitedMassloss(double fxuv, double radius,
    double mass, double mstar, double a, double beta, double eff);
double Kubyshkina18Massloss(double fxuv, double fbol,
    double mass, double radius, double a);

#endif /* MODELS_H */
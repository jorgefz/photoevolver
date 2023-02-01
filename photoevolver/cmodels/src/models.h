#include <Python.h>

#ifndef MODELS_H
#define MODELS_H

/*  Envelope Models */
extern const char docs_envelope_lopez14[];
extern const char docs_envelope_chen16[];
extern const char docs_envelope_owen17[];
PyObject* envelope_lopez14(PyObject* self, PyObject* args, PyObject* kwargs);
PyObject* envelope_chen16(PyObject* self, PyObject* args, PyObject* kwargs);
PyObject* envelope_owen17(PyObject* self, PyObject* args, PyObject* kwargs);

/*  Mass Loss Models */
extern const char docs_massloss_energy_limited[];
extern  const char docs_massloss_kubyshkina18[];
PyObject* massloss_energy_limited(PyObject* self, PyObject* args, PyObject* kwargs);
PyObject* massloss_kubyshkina18(PyObject* self, PyObject* args, PyObject* kwargs);

double BetaSalz16(double fxuv, double mass, double radius);
double EffSalz16(double mass, double radius);

#endif /* MODELS_H */
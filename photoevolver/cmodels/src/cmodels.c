#include <Python.h>
#include "models.h"

static PyMethodDef cmodels_methods[] = {
    {"envelope_lopez14", (PyCFunction)envelope_lopez14, METH_VARARGS | METH_KEYWORDS, docs_envelope_lopez14},
    {"envelope_chen16",  (PyCFunction)envelope_chen16,  METH_VARARGS | METH_KEYWORDS, docs_envelope_chen16},
    {"envelope_owen17",  (PyCFunction)envelope_owen17,  METH_VARARGS | METH_KEYWORDS, docs_envelope_owen17},

    {"massloss_energy_limited",  (PyCFunction)massloss_energy_limited,  METH_VARARGS | METH_KEYWORDS, docs_massloss_energy_limited},
    {"massloss_kubyshkina18",  (PyCFunction)massloss_kubyshkina18,  METH_VARARGS | METH_KEYWORDS, docs_massloss_kubyshkina18},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cmodels_module = {
    PyModuleDef_HEAD_INIT,
    "cmodels",
    "Python interface for the C implementation of model functions",
    -1,
    cmodels_methods
};

PyMODINIT_FUNC PyInit_cmodels(void) {
    return PyModule_Create(&cmodels_module);
}
#include <Python.h>
#include "models.h"

static PyMethodDef cmodel_methods[] = {
    {"envelope_lopez14", (PyCFunction)envelope_lopez14, METH_VARARGS | METH_KEYWORDS, docs_envelope_lopez14},
    {"envelope_chen16",  (PyCFunction)envelope_chen16,  METH_VARARGS | METH_KEYWORDS, docs_envelope_chen16},
    
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cmodels_module = {
    PyModuleDef_HEAD_INIT,
    "cmodels",
    "Python interface for the C implementation of model functions",
    -1,
    cmodel_methods
};

PyMODINIT_FUNC PyInit_cmodels(void) {
    return PyModule_Create(&cmodels_module);
}
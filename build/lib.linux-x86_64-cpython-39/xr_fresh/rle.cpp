#include <Python.h>
#include <vector>

int longest_true_run(const std::vector<bool> &arr)
{
    int max_run = 0;
    int current_run = 0;
    for (bool value : arr)
    {
        if (value)
        {
            current_run++;
            max_run = std::max(max_run, current_run);
        }
        else
        {
            current_run = 0;
        }
    }
    return max_run;
}

static PyObject *longest_true_run_wrapper(PyObject *self, PyObject *args)
{
    PyObject *arr_obj;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "O", &arr_obj))
    {
        return NULL;
    }

    // Convert the Python list to a C++ vector of bools
    std::vector<bool> arr;
    PyObject *iter = PyObject_GetIter(arr_obj);
    if (iter == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Input must be an iterable");
        return NULL;
    }
    PyObject *item;
    while ((item = PyIter_Next(iter)))
    {
        if (PyObject_IsTrue(item))
        {
            arr.push_back(true);
        }
        else
        {
            arr.push_back(false);
        }
        Py_DECREF(item);
    }
    Py_DECREF(iter);
    if (PyErr_Occurred())
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to convert input to vector");
        return NULL;
    }

    // Call the actual C++ function
    int result = longest_true_run(arr);

    // Return the result as a Python object
    return PyLong_FromLong(result);
}

static PyMethodDef module_methods[] = {
    {"longest_true_run", longest_true_run_wrapper, METH_VARARGS, "Returns the longest run of True values in an array"},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "rle", // Module name
    NULL,  // Module documentation
    -1,    // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    module_methods};

PyMODINIT_FUNC PyInit_rle(void)
{
    return PyModule_Create(&module_def);
}

/* _perfmark: C extension so a Python marker costs a few hundred instructions
 * instead of the ~10K of ctypes plus a Python-level `with` protocol.
 *
 *   Region(name: bytes, states: list[(bytes, int)]|None, extra: list[(bytes, bytes)]|None)
 *     __enter__ -> perfmark_begin / perfmark_begin_v (+ perfmark_state for each extra pair)
 *     __exit__  -> perfmark_end
 *     state(name, value)
 *   begin(name, state_name, value) / end(name) / state(name, value)
 *
 * The calls go through the PLT into libperfmark.so, which is what the drperf
 * client wraps.  Built by build.sh into build/_perfmark.so; perfmark.py falls
 * back to ctypes when it is missing.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>
#include "../perfmark.h"

static const char *
as_cstr(PyObject *o)
{
    if (PyBytes_Check(o))
        return PyBytes_AS_STRING(o);
    if (PyUnicode_Check(o))
        return PyUnicode_AsUTF8(o);
    PyErr_SetString(PyExc_TypeError, "expected str or bytes");
    return NULL;
}

typedef struct {
    PyObject_HEAD
    PyObject *name;   /* bytes */
    PyObject *states; /* list of (bytes, int): declared states, all part of the key; or NULL */
    PyObject *extra;  /* list of (bytes, bytes) or NULL */
    int waited_null_on_exit;
} RegionObject;

static int
check_pairs(PyObject *lst, int int_values, const char *what)
{
    Py_ssize_t i;
    if (!PyList_Check(lst)) {
        PyErr_Format(PyExc_TypeError, "%s must be a list of (bytes, %s)", what, int_values ? "int" : "bytes");
        return -1;
    }
    for (i = 0; i < PyList_GET_SIZE(lst); i++) {
        PyObject *t = PyList_GET_ITEM(lst, i);
        if (!PyTuple_Check(t) || PyTuple_GET_SIZE(t) != 2 || !PyBytes_Check(PyTuple_GET_ITEM(t, 0)) ||
            (int_values ? !PyLong_Check(PyTuple_GET_ITEM(t, 1)) : !PyBytes_Check(PyTuple_GET_ITEM(t, 1)))) {
            PyErr_Format(PyExc_TypeError, "%s must be a list of (bytes, %s)", what, int_values ? "int" : "bytes");
            return -1;
        }
    }
    return 0;
}

static int
Region_init(RegionObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = { "name", "states", "extra", NULL };
    PyObject *name, *states = Py_None, *extra = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "S|OO", kwlist, &name, &states, &extra))
        return -1;
    if (states != Py_None && check_pairs(states, 1, "states") < 0)
        return -1;
    if (extra != Py_None && check_pairs(extra, 0, "extra") < 0)
        return -1;
    self->waited_null_on_exit = 0;
    Py_INCREF(name);
    Py_XSETREF(self->name, name);
    if (states == Py_None) {
        Py_CLEAR(self->states);
    } else {
        Py_INCREF(states);
        Py_XSETREF(self->states, states);
    }
    if (extra == Py_None) {
        Py_CLEAR(self->extra);
    } else {
        Py_INCREF(extra);
        Py_XSETREF(self->extra, extra);
    }
    return 0;
}

static void
Region_dealloc(RegionObject *self)
{
    Py_XDECREF(self->name);
    Py_XDECREF(self->states);
    Py_XDECREF(self->extra);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int capture_started;
static _Thread_local unsigned int region_depth;

#define STACK_KEY_STATES 8 /* allocation-free fast path, not a limit */

__attribute__((visibility("default"), noinline)) PyObject *
perfmark_py_region_enter(RegionObject *self, PyObject *Py_UNUSED(ignored))
{
    const char *name = PyBytes_AS_STRING(self->name);
    Py_ssize_t n = self->states != NULL ? PyList_GET_SIZE(self->states) : 0;
    if (n == 0) {
        perfmark_begin(name, "", 0);
    } else if (n == 1) {
        PyObject *t = PyList_GET_ITEM(self->states, 0);
        int64_t value = PyLong_AsLongLong(PyTuple_GET_ITEM(t, 1));
        if (PyErr_Occurred())
            return NULL;
        perfmark_begin(name, PyBytes_AS_STRING(PyTuple_GET_ITEM(t, 0)), value);
    } else {
        const char *stack_names[STACK_KEY_STATES];
        int64_t stack_vals[STACK_KEY_STATES];
        const char **names = stack_names;
        int64_t *vals = stack_vals;
        Py_ssize_t i;
        if (n > INT_MAX) {
            PyErr_SetString(PyExc_OverflowError, "number of states exceeds the C marker ABI");
            return NULL;
        }
        if (n > STACK_KEY_STATES) {
            names = PyMem_Calloc((size_t)n, sizeof(*names));
            vals = PyMem_Calloc((size_t)n, sizeof(*vals));
            if (names == NULL || vals == NULL) {
                PyMem_Free(names);
                PyMem_Free(vals);
                return PyErr_NoMemory();
            }
        }
        for (i = 0; i < n; i++) {
            PyObject *t = PyList_GET_ITEM(self->states, i);
            names[i] = PyBytes_AS_STRING(PyTuple_GET_ITEM(t, 0));
            vals[i] = PyLong_AsLongLong(PyTuple_GET_ITEM(t, 1));
            if (PyErr_Occurred())
                break;
        }
        if (!PyErr_Occurred())
            perfmark_begin_v(name, (int)n, names, vals);
        if (n > STACK_KEY_STATES) {
            PyMem_Free(names);
            PyMem_Free(vals);
        }
        if (PyErr_Occurred())
            return NULL;
    }
    if (self->extra != NULL) {
        Py_ssize_t i;
        for (i = 0; i < PyList_GET_SIZE(self->extra); i++) {
            PyObject *t = PyList_GET_ITEM(self->extra, i);
            perfmark_state(PyBytes_AS_STRING(PyTuple_GET_ITEM(t, 0)),
                           PyBytes_AS_STRING(PyTuple_GET_ITEM(t, 1)));
        }
    }
    capture_started = 1;
    region_depth++;
    Py_INCREF(self);
    return (PyObject *)self;
}

__attribute__((visibility("default"), noinline)) PyObject *
perfmark_py_region_exit(RegionObject *self, PyObject *Py_UNUSED(args))
{
    if (self->waited_null_on_exit)
        perfmark_waited_null();
    perfmark_end(PyBytes_AS_STRING(self->name));
    if (region_depth) region_depth--;
    Py_RETURN_FALSE;
}

static PyObject *
Region_waited_null_on_exit(RegionObject *self, PyObject *Py_UNUSED(ignored))
{
    self->waited_null_on_exit = 1;
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *
Region_state(RegionObject *self, PyObject *args)
{
    PyObject *ko, *vo;
    const char *k, *v;
    if (!PyArg_ParseTuple(args, "OO", &ko, &vo))
        return NULL;
    if ((k = as_cstr(ko)) == NULL || (v = as_cstr(vo)) == NULL)
        return NULL;
    perfmark_state(k, v);
    Py_RETURN_NONE;
}

static PyMethodDef Region_methods[] = {
    { "__enter__", (PyCFunction)perfmark_py_region_enter, METH_NOARGS, "open the region" },
    { "__exit__", (PyCFunction)perfmark_py_region_exit, METH_VARARGS, "close the region" },
    { "state", (PyCFunction)Region_state, METH_VARARGS, "attach an extra state to the open region" },
    { "waited_null_on_exit", (PyCFunction)Region_waited_null_on_exit, METH_NOARGS,
      "explicitly emit waited(null) immediately before exit, including exception exit; requires an interface reason and indicator" },
    { NULL, NULL, 0, NULL }
};

static PyTypeObject RegionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_perfmark.Region",
    .tp_basicsize = sizeof(RegionObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Region_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "perfmark region (C fast path)",
    .tp_methods = Region_methods,
    .tp_init = (initproc)Region_init,
    .tp_new = PyType_GenericNew,
};

__attribute__((visibility("default"), noinline)) PyObject *
perfmark_py_begin(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyObject *no, *so = NULL;
    const char *name, *sname = "";
    long long value = 0;
    if (!PyArg_ParseTuple(args, "O|OL", &no, &so, &value))
        return NULL;
    if ((name = as_cstr(no)) == NULL)
        return NULL;
    if (so != NULL && (sname = as_cstr(so)) == NULL)
        return NULL;
    perfmark_begin(name, sname, value);
    capture_started = 1;
    region_depth++;
    Py_RETURN_NONE;
}

__attribute__((visibility("default"), noinline)) PyObject *
perfmark_py_end(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyObject *no;
    const char *name;
    if (!PyArg_ParseTuple(args, "O", &no))
        return NULL;
    if ((name = as_cstr(no)) == NULL)
        return NULL;
    perfmark_end(name);
    if (region_depth) region_depth--;
    Py_RETURN_NONE;
}

static PyObject *
mod_state(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyObject *ko, *vo;
    const char *k, *v;
    if (!PyArg_ParseTuple(args, "OO", &ko, &vo))
        return NULL;
    if ((k = as_cstr(ko)) == NULL || (v = as_cstr(vo)) == NULL)
        return NULL;
    perfmark_state(k, v);
    Py_RETURN_NONE;
}

/* Exported entry boundaries let drperf exclude parsing, GIL handoff and all
 * callees exactly. The actual checkpoint still uses libperfmark's C ABI. */
static PyObject *event_checkpoint(PyObject *args, int publish)
{
    PyObject *eo, *go = NULL;
    unsigned long long event, generation;
    if (!PyArg_ParseTuple(args, "O|O", &eo, &go))
        return NULL;
    if (!publish && eo == Py_None && go == NULL) {
        Py_BEGIN_ALLOW_THREADS
        perfmark_waited_null();
        Py_END_ALLOW_THREADS
        Py_RETURN_NONE;
    }
    if (go == NULL || !PyLong_Check(eo) || !PyLong_Check(go) || PyBool_Check(eo) || PyBool_Check(go)) {
        PyErr_SetString(PyExc_ValueError, "event IDs and generations must be unsigned 64-bit integers");
        return NULL;
    }
    event = PyLong_AsUnsignedLongLong(eo);
    if (PyErr_Occurred()) return NULL;
    generation = PyLong_AsUnsignedLongLong(go);
    if (PyErr_Occurred()) return NULL;
    Py_BEGIN_ALLOW_THREADS
    if (publish == 2) perfmark_release(event, generation);
    else if (publish) perfmark_event_publish(event, generation);
    else perfmark_event_waited(event, generation);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

__attribute__((visibility("default"), noinline)) PyObject *
perfmark_py_event_publish(PyObject *m, PyObject *args)
{
    (void)m;
    return event_checkpoint(args, 1);
}

__attribute__((visibility("default"), noinline)) PyObject *
perfmark_py_event_waited(PyObject *m, PyObject *args)
{
    (void)m;
    return event_checkpoint(args, 0);
}

__attribute__((visibility("default"), noinline)) PyObject *
perfmark_py_release(PyObject *m, PyObject *args)
{
    (void)m;
    return event_checkpoint(args, 2);
}

__attribute__((visibility("default"), noinline)) PyObject *
perfmark_py_wait(PyObject *m, PyObject *args, PyObject *kwargs)
{
    PyObject *eo, *go = Py_None;
    const char *indicator = NULL, *producer = NULL, *reason = NULL;
    static char *keywords[] = {"event", "generation", "indicator", "producer", "reason", NULL};
    unsigned long long event, generation;
    (void)m;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Ozzz:wait", keywords,
                                    &eo, &go, &indicator, &producer, &reason))
        return NULL;
    if (!indicator || !indicator[0] || strlen(indicator) > 2048 ||
        (producer && (!producer[0] || strlen(producer) > 127)) ||
        (reason && (!reason[0] || strlen(reason) > 2048))) {
        PyErr_SetString(PyExc_ValueError, "wait requires a nonempty indicator expression (at most 2048 bytes); producer/reason must be nonempty when supplied");
        return NULL;
    }
    if (eo == Py_None) {
        if (go != Py_None || producer || !reason) {
            PyErr_SetString(PyExc_ValueError, "wait(None) requires a reason and no generation or producer");
            return NULL;
        }
        Py_BEGIN_ALLOW_THREADS
        perfmark_wait_null(indicator, reason);
        Py_END_ALLOW_THREADS
        Py_RETURN_NONE;
    }
    if (!PyLong_Check(eo) || !PyLong_Check(go) || PyBool_Check(eo) || PyBool_Check(go) || reason) {
        PyErr_SetString(PyExc_ValueError, "wait requires uint64 event/generation; reason is only for wait(None)");
        return NULL;
    }
    event = PyLong_AsUnsignedLongLong(eo);
    if (PyErr_Occurred()) return NULL;
    generation = PyLong_AsUnsignedLongLong(go);
    if (PyErr_Occurred()) return NULL;
    Py_BEGIN_ALLOW_THREADS
    perfmark_wait(event, generation, indicator, producer);
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

__attribute__((visibility("default"), noinline)) PyObject *
perfmark_py_runtime(PyObject *m, PyObject *args)
{
    int kind, status = 0;
    unsigned long long id, api;
    (void)m;
    if (!PyArg_ParseTuple(args, "iKK|i", &kind, &id, &api, &status))
        return NULL;
    if (kind == 0) perfmark_runtime_wait_begin(id, api);
    else if (kind == 1) perfmark_runtime_wait_end(id, api, status);
    else if (kind == 2) perfmark_async_scope(id, api);
    else { PyErr_SetString(PyExc_ValueError, "invalid runtime observation"); return NULL; }
    Py_RETURN_NONE;
}

/* Only observation bookkeeping belongs inside this dynamic exclusion. The
 * adapter invokes the original awaitable outside it. */
__attribute__((visibility("default"), noinline)) PyObject *
perfmark_py_runtime_call(PyObject *m, PyObject *args)
{
    (void)m;
    if (PyTuple_GET_SIZE(args) < 1) {
        PyErr_SetString(PyExc_TypeError, "runtime_call requires a callable");
        return NULL;
    }
    PyObject *tail = PyTuple_GetSlice(args, 1, PyTuple_GET_SIZE(args));
    if (tail == NULL) return NULL;
    PyObject *result = PyObject_CallObject(PyTuple_GET_ITEM(args, 0), tail);
    Py_DECREF(tail);
    return result;
}

static PyObject *
runtime_state(PyObject *m, PyObject *args)
{
    (void)m; (void)args;
    return Py_BuildValue("ii", capture_started, region_depth);
}

static PyMethodDef mod_methods[] = {
    { "runtime_call", perfmark_py_runtime_call, METH_VARARGS, "excluded observation bookkeeping" },
    { "runtime_state", runtime_state, METH_NOARGS, "capture started and active depth" },
    { "runtime", perfmark_py_runtime, METH_VARARGS, "internal runtime observation" },
    { "release", perfmark_py_release, METH_VARARGS, "record completion immediately before the real release" },
    { "wait", (PyCFunction)perfmark_py_wait, METH_VARARGS | METH_KEYWORDS, "record completed wait with an inline entry-PCV indicator" },
    { "event_publish", perfmark_py_event_publish, METH_VARARGS, "publish immediately before the original release" },
    { "event_waited", perfmark_py_event_waited, METH_VARARGS, "record observed readiness; never blocks" },
    { "begin", perfmark_py_begin, METH_VARARGS, "begin(name, state_name='', value=0)" },
    { "end", perfmark_py_end, METH_VARARGS, "end(name)" },
    { "state", mod_state, METH_VARARGS, "state(name, value)" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_perfmark", "perfmark markers, C fast path", -1, mod_methods,
};

PyMODINIT_FUNC
PyInit__perfmark(void)
{
    /* Resolve TLS before the first region can trigger late attachment. */
    region_depth = 0;
    PyObject *m;
    if (PyType_Ready(&RegionType) < 0)
        return NULL;
    m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;
    Py_INCREF(&RegionType);
    if (PyModule_AddObject(m, "Region", (PyObject *)&RegionType) < 0) {
        Py_DECREF(&RegionType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}

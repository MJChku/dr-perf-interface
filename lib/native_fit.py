"""Bounded batches for the C++ affine checker; Python only marshals observations."""
import ctypes as C
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def library():
    path = Path(__file__).resolve().parents[1] / 'build/libdrperf_fit.so'
    try:
        lib = C.CDLL(str(path))
    except OSError as error:
        raise ValueError('DrPerf C++ fitter unavailable; run ./build.sh') from error
    lib.drperf_fit_abi.restype = C.c_uint
    if lib.drperf_fit_abi() != 1:
        raise ValueError('DrPerf C++ fitter ABI mismatch; run ./build.sh')
    doubles = C.POINTER(C.c_double)
    lib.drperf_fit_create.argtypes = [C.c_size_t, C.c_size_t, doubles, doubles, doubles]
    lib.drperf_fit_create.restype = C.c_void_p
    lib.drperf_fit_destroy.argtypes = [C.c_void_p]
    lib.drperf_fit_destroy.restype = None
    lib.drperf_fit_dependent.argtypes = [C.c_void_p, C.POINTER(C.c_ubyte)]
    lib.drperf_fit_dependent.restype = None
    lib.drperf_fit_batch.argtypes = [C.c_void_p, C.c_size_t, doubles, C.c_double, C.c_double, doubles]
    lib.drperf_fit_batch.restype = C.c_int
    return lib


class Plan:
    def __init__(self, values):
        if not values:
            raise ValueError('An affine fit requires at least one observed state')
        self.n, self.k = len(values), len(values[0])
        if any(len(v) != self.k for v in values):
            raise ValueError('PCV state dimensions differ')
        self.lib = library()
        # Preserve exact integer summation before conversion, including PCVs >2**53.
        mean = [sum(v[j] for v in values) / self.n for j in range(self.k)]
        scale = [max(abs(v[j]-mean[j]) for v in values) or 1.0 for j in range(self.k)]
        points = (C.c_double * (self.n*self.k))(*(x for v in values for x in v))
        means = (C.c_double * self.k)(*mean)
        scales = (C.c_double * self.k)(*scale)
        self.handle = self.lib.drperf_fit_create(self.n, self.k, points, means, scales)
        if not self.handle:
            raise MemoryError('Cannot allocate DrPerf affine fit plan')
        dependent = (C.c_ubyte * self.k)()
        self.lib.drperf_fit_dependent(self.handle, dependent)
        self.dependent = {j for j, flag in enumerate(dependent) if flag}

    def close(self):
        if self.handle:
            self.lib.drperf_fit_destroy(self.handle)
            self.handle = None

    def fit(self, rows, absolute, relative):
        if not self.handle:
            raise ValueError('Affine fit plan is closed')
        if any(len(row) != self.n for row in rows):
            raise ValueError('Block counts must match the observed state count')
        count = len(rows)
        ys = (C.c_double * (count*self.n))(*(y for row in rows for y in row))
        out = (C.c_double * (count*(self.k+5)))()
        if self.lib.drperf_fit_batch(self.handle, count, ys, absolute, relative, out):
            raise MemoryError('Cannot allocate DrPerf affine fitting workspace')
        return [list(out[i*(self.k+5):(i+1)*(self.k+5)]) for i in range(count)]


def affine_fit(values, ys):
    plan = Plan(values)
    try:
        row = plan.fit([ys], 0.0, 0.0)[0]
        return row[:plan.k], row[plan.k], row[plan.k+1], plan.dependent
    finally:
        plan.close()


def block_fits(values, vecs, blocks, absolute, relative):
    plan = Plan(values)
    try:
        # Bound the dense transfer buffer even for large state sets.
        batch = max(1, min(128, 262144 // plan.n))
        for start in range(0, len(blocks), batch):
            selected = blocks[start:start+batch]
            rows = [[vecs[v].get(b, 0.0) for v in values] for b in selected]
            fits = plan.fit(rows, absolute, relative)
            for b, ys, fit in zip(selected, rows, fits):
                yield b, ys, fit[:plan.k], fit[plan.k:], plan.dependent
    finally:
        plan.close()

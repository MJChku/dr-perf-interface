"""Validation probe: pure-Python loop, numpy matmul, torch CPU matmul.

    drperf run -o out --state n=128,256 -- <venv>/bin/python probes/validate.py

Regions:  py_loop (state iters), numpy_matmul (state n), torch_matmul (state n).
Each is warmed up once outside any region before being measured.
"""
import perfmark

st = perfmark.states(n=128, warm=1)
n = int(st["n"])

import numpy as np  # noqa: E402

np.random.seed(0)


def py_loop(k):
    s = 0
    for i in range(k):
        s += i * i
    return s


iters = 50 * n
py_loop(100)
with perfmark.region("py_loop", iters=iters):
    py_loop(iters)

a = np.random.rand(n, n).astype(np.float32)
b = np.random.rand(n, n).astype(np.float32)
c = a @ b
with perfmark.region("numpy_matmul", n=n):
    c = a @ b

import torch  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(1)
ta = torch.from_numpy(a)
tb = torch.from_numpy(b)
tc = ta @ tb
with perfmark.region("torch_matmul", n=n):
    tc = ta @ tb

print("validate n=%d py_loop=%d numpy=%.3f torch=%.3f" % (n, py_loop(10), float(c[0, 0]), float(tc[0, 0])))

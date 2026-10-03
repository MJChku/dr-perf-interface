"""Compare two equiv_t5.py dumps: max absolute difference per array.

    .venv/bin/python cmp_t5.py BASE.npz FIX.npz
"""
import sys

import numpy as np

a = np.load(sys.argv[1])
b = np.load(sys.argv[2])
keys = sorted(set(a.files) | set(b.files))
worst = 0.0
nbytes = 0
for k in keys:
    if k not in a.files or k not in b.files:
        print("%-8s MISSING" % k)
        continue
    x, y = a[k], b[k]
    if x.shape != y.shape:
        print("%-8s SHAPE %s vs %s" % (k, x.shape, y.shape))
        worst = float("inf")
        continue
    d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
    same = np.array_equal(x, y)
    nbytes += x.nbytes
    worst = max(worst, d)
    print("%-8s shape=%-22s max|diff|=%.3e  bitwise_identical=%s  max|x|=%.4g"
          % (k, str(x.shape), d, same, float(np.max(np.abs(x))) if x.size else 0.0))
print("---- worst max|diff| over %d arrays (%.1f MB): %.3e" % (len(keys), nbytes / 1e6, worst))

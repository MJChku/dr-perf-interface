"""Regime example: a lookup that scans linearly for small directories and
uses a dict above a threshold.  One run triggers the region at several sizes
so `drperf model` sees both regimes:

    drperf run --model lookup -o out/regime -- python examples/py_regime/app.py
    drperf model out/regime
"""
import random

import perfmark

random.seed(0)
SIZES = [int(x) for x in str(perfmark.states(sizes="20,60,300,900")["sizes"]).split(",")]
KEYS = 64


def lookup(entries, index, keys):
    n = len(entries)
    with perfmark.region("lookup", n_entries=n):
        out = []
        if n < 100:
            for k in keys:                      # linear scan: 64 * n comparisons
                for key, sb in entries:
                    if key == k:
                        out.append(sb)
                        break
        else:
            for k in keys:                      # dict: 64 lookups
                out.append(index[k])
        return out


for n in SIZES:
    entries = [(k, (k * 7919) % 4096) for k in range(n)]
    index = dict(entries)
    keys = [random.randrange(n) for _ in range(KEYS)]
    lookup(entries, index, keys)               # warm-up
    for _ in range(3):
        lookup(entries, index, keys)
print("regime example sizes=%s" % SIZES)

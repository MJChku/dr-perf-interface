"""Delta example (Python), BASE: bind a batch of keys against a directory.

    drperf run -o out/base --state n=1000,10000 -- python examples/py_delta/base/app.py
"""
import random

import perfmark

st = perfmark.states(n=1000)
n = int(st["n"])
random.seed(0)

directory = {k: (k * 7919) % 4096 for k in range(n)}   # key -> superblock
keys = [random.randrange(n) for _ in range(256)]        # one batch


def bind_batch(directory, keys):
    out = []
    for k in keys:
        out.append(directory[k])
    return out


bind_batch(directory, keys)  # warm-up outside any region
with perfmark.region("bind_batch", n_entries=n):
    out = bind_batch(directory, keys)
print("base n=%d sum=%d" % (n, sum(out)))

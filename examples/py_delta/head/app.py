"""Delta example (Python), HEAD: bind_batch now validates the directory first.

The delta versus base/app.py is the `validate` function and its call, marked
as a nested region so the diff shows exactly what it costs.

    drperf run -o out/head --state n=1000,10000 -- python examples/py_delta/head/app.py
"""
import random

import perfmark

st = perfmark.states(n=1000)
n = int(st["n"])
random.seed(0)

directory = {k: (k * 7919) % 4096 for k in range(n)}   # key -> superblock
keys = [random.randrange(n) for _ in range(256)]        # one batch


def validate(directory):
    with perfmark.region("validate", n_entries=len(directory)):
        checksum = 0
        for k, v in directory.items():
            if v >= 4096:
                raise ValueError("superblock out of range")
            checksum = (checksum * 31 + (k ^ v)) & 0xFFFFFFFF
        return checksum


def bind_batch(directory, keys):
    validate(directory)
    out = []
    for k in keys:
        out.append(directory[k])
    return out


bind_batch(directory, keys)  # warm-up outside any region
with perfmark.region("bind_batch", n_entries=n):
    out = bind_batch(directory, keys)
print("head n=%d sum=%d" % (n, sum(out)))

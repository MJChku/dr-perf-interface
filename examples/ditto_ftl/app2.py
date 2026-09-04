"""Same FTL region, with the hidden variable declared.

app.py declared plan_load(keys, live) and drperf reported 67.9% of the cost as
irregular: the work is a superblock deep-clone per looked-up block, so it
scales with how many ENTRIES get copied, which is keys x blocks-per-superblock.
Here superblocks are built at three different fill levels in one run and the
region declares `entries` alongside `keys`.

    bin/drperf-dev run --blocks -o out/ftl2 -- python examples/ditto_ftl/app2.py
    bin/drperf-dev derive out/ftl2 --region plan_load
"""
import sys

sys.path.insert(0, "/home/ubuntu/compression/ditto_kv")

import perfmark  # noqa: E402

from src.ftl import (  # noqa: E402
    CallbackStorage,
    FreshAllocator,
    FTLController,
    HandleManager,
)

ROW = 4096
CAPACITY = 512            # blocks a superblock may hold
FILLS = (16, 64, 256)     # blocks actually bound per group, one group each
KEYS = (1, 2, 4, 8, 16)
REPS = 4


def main():
    total = sum(FILLS)
    store = HandleManager(total * 64 * ROW, chunk_bytes=ROW, pin=False)
    pool = FreshAllocator(store, ROW, total)
    ftl = FTLController(
        CAPACITY,
        storage=CallbackStorage(pool.allocate, pool.free, lambda: pool.allocatable_rows(0)),
    )

    lb = 0
    first_of_group = {}
    for group, fill in enumerate(FILLS):
        first_of_group[group] = lb
        for ordinal in range(fill):
            placement = ftl.bind(lb, f"owner{group}", group, ordinal, position=ordinal * 256)
            with store.write_lock(placement.handle) as locked:
                locked.ensure_row(placement.offset * ROW)
            lb += 1

    # plan_load against superblocks of three different fills: `entries` is the
    # number of map entries the lookups have to copy, keys x fill.
    for group, fill in enumerate(FILLS):
        base = first_of_group[group]
        for keys in KEYS:
            if keys > fill:
                continue
            request = {group: [(base + i, f"d{i}") for i in range(keys)]}
            for _ in range(REPS):
                with perfmark.region("plan_load", keys=keys, entries=keys * fill):
                    plan = ftl.plan_load(request)
                assert plan is not None
    print(f"ditto_ftl app2 fills={FILLS} keys={KEYS}")


if __name__ == "__main__":
    main()

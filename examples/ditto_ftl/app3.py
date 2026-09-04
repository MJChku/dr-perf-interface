"""One lookup per call, so the only thing varying is the map being cloned.

app2.py declared entries = keys x fill and the irregular share did not move.
The measurements suggest why: per-entry cost is not constant (127 instr/entry
between 256 and 1024 entries, 157 between 1024 and 4096), which is what a
B-tree clone does as depth grows. Here keys is pinned at 1 and `fill` sweeps,
so if `fill` is the state the blocks follow, the irregular share should fall.
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
CAPACITY = 1024
FILLS = (16, 32, 64, 128, 192, 256, 384, 512)
REPS = 6


def main():
    total = sum(FILLS)
    store = HandleManager(total * 64 * ROW, chunk_bytes=ROW, pin=False)
    pool = FreshAllocator(store, ROW, total)
    ftl = FTLController(
        CAPACITY,
        storage=CallbackStorage(pool.allocate, pool.free, lambda: pool.allocatable_rows(0)),
    )
    lb, first = 0, {}
    for group, fill in enumerate(FILLS):
        first[group] = lb
        for ordinal in range(fill):
            placement = ftl.bind(lb, f"owner{group}", group, ordinal, position=ordinal * 256)
            with store.write_lock(placement.handle) as locked:
                locked.ensure_row(placement.offset * ROW)
            lb += 1
    for group, fill in enumerate(FILLS):
        request = {group: [(first[group], "d0")]}
        for _ in range(REPS):
            with perfmark.region("plan_load", fill=fill):
                plan = ftl.plan_load(request)
            assert plan is not None
    print(f"ditto_ftl app3 fills={FILLS}")


if __name__ == "__main__":
    main()

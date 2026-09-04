"""drperf on the ditto FTL controller (CPU only, no CUDA).

Wraps the controller calls the serving path makes per step -- bind, plan_load,
compaction nomination -- in perfmark regions and declares the state each one
should scale with. Nothing in ditto-kv is modified: the regions are applied by
wrapping the bound methods here.

    bin/drperf python examples/ditto_ftl/app.py
"""
import os
import sys

sys.path.insert(0, "/home/ubuntu/compression/ditto_kv")

import perfmark  # noqa: E402

from src.ftl import (  # noqa: E402
    CallbackStorage,
    FreshAllocator,
    FTLController,
    HandleManager,
)

ROW = 4096          # io-block row bytes
NBLOCKS = int(os.environ.get("FTL_BLOCKS", "512"))
GROUPS = int(os.environ.get("FTL_GROUPS", "8"))


def build(nblocks):
    store = HandleManager(nblocks * 64 * ROW, chunk_bytes=ROW, pin=False)
    pool = FreshAllocator(store, ROW, nblocks)
    ftl = FTLController(
        nblocks,
        storage=CallbackStorage(pool.allocate, pool.free, lambda: pool.allocatable_rows(0)),
    )
    return ftl, pool, store


def main():
    ftl, pool, store = build(NBLOCKS)

    # --- bind: cost per io-block placed, state = blocks already live
    live = 0
    for lb in range(NBLOCKS):
        group = lb % GROUPS
        with perfmark.region("bind", live=live, group=group):
            placement = ftl.bind(lb, f"owner{group}", group, lb // GROUPS,
                                 position=(lb // GROUPS) * 256)
            with store.write_lock(placement.handle) as locked:
                locked.ensure_row(placement.offset * ROW)
        live += 1

    # --- plan_load: the per-step lookup, state = keys asked for
    for keys in (1, 2, 4, 8, 16, 32, 64, 128):
        for rep in range(4):
            request = {0: [(lb, f"d{lb}") for lb in range(keys)]}
            with perfmark.region("plan_load", keys=keys, live=live):
                plan = ftl.plan_load(request)
            assert plan is not None

    # --- invalidate: releases as the churn evicts, state = blocks dropped
    for drop in (1, 4, 16, 64):
        base = 0
        for rep in range(3):
            ids = list(range(base, base + drop))
            base += drop
            with perfmark.region("invalidate", drop=drop, live=live):
                for lb in ids:
                    ftl.invalidate(lb)
            live -= drop
    print(f"ditto_ftl blocks={NBLOCKS} groups={GROUPS} live={live}")


if __name__ == "__main__":
    main()

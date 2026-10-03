# vLLM case C: the persistent input batch under churn

Markers `mark_batch.py`, driver `driver.py`: 24 requests through 8 slots, so
requests queue, are admitted, finish at staggered steps and are replaced,
making `num_new` / `num_finished` / `condense` vary every few steps.

Study ("vLLM case C"):

    add_request        275.2*num_prompt + 46,457   ->   202.9*num_prompt + 42,629
    refresh_metadata   83,361 -> 44,595 per call at the steady replace (-46.5%)
    condense           47,366 -> 33,844 per moved row in bursts of >= 3 (-28.5%)

Ten candidates for the arrival copy were measured under drperf first: the
obvious rewrites (`np.asarray`, `torch.tensor`, `np.fromiter`) all cost more
than the status quo; a cached `memoryview` per row assigned from
`array("i", ids)` won on slope and constant.  `refresh_metadata` is memoised on
the tuple of scalars it depends on; `condense` batches its moves at three rows
(measured crossover 2.7).

Equivalence: `verify.py` dumps text and token ids of all 24 requests;
`check_equiv.py` replays a random add/remove/condense/refresh sequence on the
modified `InputBatch` and on the pristine module in `base_batch/`, comparing
every array and the derived sampling metadata.

## Rerun (2026-10-02, `run.sh`)

```
add_request       base  266*num_prompt + 47,917     3% irregular
                  fix   188*num_prompt + 43,760     3%
refresh_metadata  base  2*num_reqs + 2,953*num_added + 42,528      34%
                  fix   -235*num_reqs + 4,351*num_added + 36,043   27%
condense          base  -109*num_reqs + 3,198   (single departures; the batched path needs bursts of >= 3)
                  fix   -100*num_reqs + 3,134
```
266 -> 188 per prompt token (study 275 -> 203).  Text and token ids of all
24 requests identical; the differential test of the modified `InputBatch`
against the pristine module passes (1,800 replayed operations).

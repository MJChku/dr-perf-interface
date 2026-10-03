# vLLM case E: the output path -- the surprise was mostly the markers

Driver `driver.py`: the same 370-token workload through FINAL_ONLY (what
`LLM.generate` forces), CUMULATIVE and DELTA.  The first measurement, with
seven nested Python regions on the path, read 95.5k per request per step and
blamed two dataclass constructions; with two regions (`mark_out_clean.py`, used here;
`mark_out.py` is the seven-region set) it read 43.8k, and marker-free 8,248 in FINAL_ONLY.  A Python
region with a state costs ~11.4k to construct, and that is charged to the
parent region.

Study ("vLLM case E"), third beat:

    8,247.5 -> 6,989.4 per request per step (-15.3%), ~80k per step at 64 requests

by not entering `make_request_output` to return early in FINAL_ONLY, and by
testing `iteration_stats` before calling a no-op.  A lazy string join for
`output_text +=` was implemented, fuzzed, and rejected because the formula
predicted it nets to zero -- it is not in the patch.

The study's tree carries the fix in its unmarked snapshot, so `prepare.sh`
unmarks, reverses `output_path.patch` for the base, and re-marks both trees
with the same two regions.  `run.sh` reports `process_outputs` with its two
states; `num_outputs` and `num_active` are nearly collinear on this workload,
so only their sum is identified -- read the sum.

Equivalence: `verify.py` dumps every `RequestOutput` of the three phases
(752 records); `run.sh` compares base and fix.

## Rerun (2026-10-02, `run.sh`, `phases=A`)

```
process_outputs  base  -3,783*num_outputs + 11,820*num_active + 11,477    (num_outputs = num_active: 8,037 per request-step)
                 fix   -5,338*num_outputs + 12,096*num_active + 12,317    (6,758 per request-step)
per call         (10 outputs) 190,030 -> 179,917;  (12) 325,023 -> 311,763;  (8) 136,960 -> 128,378
```
With `process_outputs` as the only region on the path (the nested
`make_request_output` marker unwrapped by `prepare.sh`, since the fix skips
that call and would otherwise be credited with the marker's own ~11k) and
only the FINAL_ONLY phase, the per-request-step cost is 8,037 -> 6,758
(-15.9%; study 8,247.5 -> 6,989.4, -15.3%).  All 753 RequestOutput records of
the three phases identical.

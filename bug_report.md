# `bin/drperf` crashes on every run: `IndexError: tuple index out of range`

## Symptom

```
$ bin/drperf /home/ubuntu/compression/ditto_kv/.venv/bin/python examples/ditto_ftl/app.py
Traceback (most recent call last):
  File "bin/drperf", line 181, in main
    costs = cost_lines(rs, keys, slots, recs)
  File "bin/drperf", line 52, in cost_lines
    if len(set(v[j] for v in r.values)) <= 1:
IndexError: tuple index out of range
```

The program runs to completion and the client writes its JSON; only the
derivation at the end fails, so `bin/drperf` never prints anything. The same
data analysed through `bin/drperf-dev run --blocks -o OUT` and
`bin/drperf-dev derive OUT` works and prints the formulas.

## Cause

Late attach. DynamoRIO starts at the *first* marked region, and that first
trigger is recorded without its declared states. So for the region that
happened to start the process, `derive.per_state` returns one state tuple of
length 0 alongside the real ones, while `names` comes from a key that does have
states:

```
bind:       names=('live',)          value-tuple lengths = [0, 1]   <-- mixed
invalidate: names=('drop', 'live')   value-tuple lengths = [2]
plan_load:  names=('keys', 'live')   value-tuple lengths = [2]
```

`cost_lines` then indexes `v[j]` for every `j` in `range(len(names))` over
`r.values`, and the zero-length tuple raises. Reproduced with:

```python
import sys; sys.path.insert(0, "lib")
import runner, derive
rc, out, files = runner.run(argv, out_dir)
rs = runner.load_runs(out_dir); keys, slots = runner.blocks_of_set(rs)
recs = [r for r in runner.load_traces_all(rs) if not r["region"].startswith(runner.CALIB)]
for R in regions:
    vecs, trig, names, nested = derive.inclusive_vectors(keys, R, recs)
    print(R, names, sorted({len(v) for v in vecs}))
```

It is not specific to this workload: any program whose first marked region
declares at least one state should hit it. It did not show up in the examples
presumably because their first region is entered with the same states as every
later call, so the malformed key merges into an existing state point.

`bin/drperf-dev derive` hits the same malformed key by a different route, so
the fix belongs in `lib/derive.py` rather than in either CLI:

```
$ bin/drperf-dev derive OUT --region bind
  File "lib/derive.py", line 419, in describe
    top_a = sorted(r.by_sym_a[j].items(), key=lambda kv: -abs(kv[1]))[:top]
IndexError: list index out of range
```

Here the regime carries fewer per-state symbol maps than there are names, for
the same reason. `derive OUT --region plan_load` and `--region invalidate`,
whose regions did not trigger the attach, both work and print their formulas.

## Suggested fix

In `lib/derive.py:per_state`, ignore keys whose state arity does not match the
region's declared names (or pad them and let them fall into their own point):

```python
def per_state(keys, region):
    out, names = {}, None
    for k in keys.values():
        if k["region"] != region or k["count"] == 0:
            continue
        v = tuple(float(val) for _, val in k["states"])
        if names is None and v:
            names = tuple(n for n, _ in k["states"])
        if names is not None and len(v) != len(names):
            continue          # attach-time trigger, states not captured
        ...
```

Taking `names` from the first key *that has states* also matters: if the
attach trigger is the first key seen, `names` becomes `()` and every
coefficient silently disappears instead of crashing.

A guard in `bin/drperf:cost_lines` (`if j >= len(v)`) would stop the crash but
would still average a call whose states are unknown into a state point it does
not belong to, so the drop in `per_state` is the better place.

## Environment

DynamoRIO 11.3.0 as built by `./build.sh`, Python 3.12, client JSON from
`bin/drperf-dev run --blocks`. The failing analysis path is shared by
`bin/drperf` only; `drperf-dev show|derive` are fine.

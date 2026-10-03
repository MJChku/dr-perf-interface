# Conditional PCVs for a branch inside a loop

Run from the repository root, after `./build.sh`:

```sh
python3 examples/branch_pcvs/validate.py
```

The script builds `app.c`, checks its results natively, runs it under drperf's
DynamoRIO client, and validates the fits from the measured basic-block counts.
It preserves the measurements, a per-state CSV, and a text report in a new
`out/branch_pcvs-*` directory. To use the ordinary CLI after building:

```sh
bin/drperf examples/branch_pcvs/bin/branch_pcvs
```

The program runs this loop for 20 independently varied `(x, y)` pairs, with
three repetitions per pair and annotation:

```c
for (long i = 0; i < x; ++i) {
    if (i < y)
        result += expensive_step(i + 1);
    else
        result += cheap_step(i + 1);
}
```

Both regions wrap the same compiled function through the same call site:

- `raw` declares `x` and `y`.
- `conditional` declares `expensive_iters = x <= y ? x : y` and
  `cheap_iters = x <= y ? 0 : x - y`.

All inputs in the measured grid are positive. When `y >= x`, every iteration
takes the expensive branch. When `y < x`, the remaining `x - y` iterations
take the cheap branch. PCV expressions are evaluated before region entry.
The expensive helper performs 32 additions, with a compiler barrier to retain
the work; the cheap helper returns its argument. Assertions check each result
against an independently calculated arithmetic sum.

Observed with GCC `-O2` in this workspace:

```text
raw:         4*x + 19
             94.208292% unexplained

conditional: 109*expensive_iters + 11*cheap_iters + 19
             0% unexplained; 0 instructions/call maximum residual
```

The conditional representation has 14 distinct state points: several raw
inputs describe identical branch counts and are aggregated together. There
are 60 calls per region. The validator checks that no calls were dropped,
the run has no reported validity problems, the raw fit leaves more than half
the cost unexplained, and every block passes with the conditional PCVs. It
also prints the maximum residual instead of equating zero unexplained cost
with an exact fit automatically.

The measured coefficients depend on the compiler and binary. These results
validate the fits at the measured states; the branch-count expressions
themselves follow directly from the loop for all nonnegative `x` and `y`.
This example demonstrates why a piecewise annotation can be useful even
though the checker fits only affine formulas over the supplied PCVs.

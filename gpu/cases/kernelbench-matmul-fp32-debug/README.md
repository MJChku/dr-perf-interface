# Public compensated FP32 matmul: rejected tensor-work reduction

This experiment uses
[good-kernels/MatmulFP32](https://github.com/ScalingIntelligence/good-kernels/tree/5c94620c59ab9839a4ad4ccad309f3d91752d042/MatmulFP32)
at revision `5c94620c59ab9839a4ad4ccad309f3d91752d042`. The reference is FP32
square matrix multiplication, originally N=4096. The published CUDA source
splits each operand into high/low FP16 components and accumulates all four
products with WMMA. No inefficiency was inserted. This is a standalone
experiment, not an official KernelBench submission.

**Candidate:** omit the low×low product. **Decision:** retain the original
four-product kernel when preserving the tested numerical contract. A reduction
in tensor arithmetic is not sufficient evidence of an acceptable optimization.
No native GPU allocation is requested for the rejected candidate.

## DrPerf feedback and proposed change

The untouched extracted baseline is [before.cu](before.cu); only unused
Torch/ATen includes and the host wrapper were removed. No ABI compatibility
change was needed. DrPerf's CPU execution measures these aggregate rules:

```
P = (N/128)^2 * (N/16)       # 128×128×16 tile products; tested N divisible by128
I = N^2 * (N/128)           # input elements per macro-tile grid
O = N^2
baseline tensor FLOPs = 2097152 * P
candidate tensor FLOPs = 1572864 * P     # 25% less
both global read bytes = 8 * I
both global write bytes = 4 * O
```

At N128, the baseline executes 16,777,216 tensor FLOPs versus 4,194,304 for an
uncompensated mathematical GEMM. Source inspection explains the factor four;
DrPerf quantifies and verifies it, including instruction-site contributions.
The [candidate](after.cu) removes exactly one WMMA product. The numerical oracle,
not resource fitting, determines whether that change is valid.

The PCVs are meaningful work quantities but correlate strongly in this square
shape family. Only two training shapes are used, so some accepted affine
coefficients are nonunique. For example a global-write fit can select an
expression in `I` plus a constant rather than `4*O`. The runner checks the
simple aggregate laws against every observation instead of treating those
coefficients as unique inference. Held-out and training fits are separate
refits, not an extrapolation accuracy claim. Complete compact fits and their
unexplained fields are retained in `results/*-{train,heldout}.json`.

## Independent validation and rejection

Training uses N128 and N256 with seeded random finite values, constant values
and exactly represented binary fractions. Both variants pass those six cases.
Held-out data use N384 random values and N128 cancellation pairs. Every output
is compared with a double-precision dot-product oracle, using the same
`2e-6*(1+abs(reference))` tolerance for all final cases, plus input immutability
and 16 output guards.

For the cancellation state, let `e=2^-12`; each pair is:

```
A pair = [1+e, 1]
B pair = [1+e, -(1+2e)]
exact dot product = e^2
```

The high×high and both cross terms cancel; low×low supplies the remaining
representable value. Repeating this pair N/2 times gives the N128 expected
value `3.814697265625e-6` at every output. Dropping low×low loses that value,
exceeding the uniform tolerance. This intentionally challenging hold-out data
checks the published full-FP32-accuracy intent; ordinary random data would
hide this failure. The numerical contract is this experiment's explicit one,
not an assertion about unspecified official KernelBench tolerances.

The baseline passes eight of eight CPU cases. The candidate passes seven of
eight; every one of the 16,384 cancellation outputs fails with error
`3.814697265625e-6`. SM80 ptxas reports registers 130→126, shared storage
24,576 bytes unchanged and zero spills for both. Those resource improvements
do not rescue the failed numerical contract.

The final receipts record the actual pass/failure counts. For the expected
candidate hold-out rejection, the driver returns zero after printing explicit
failure counts so DrPerf can finish resource reporting. The runner independently
parses and asserts the expected one failed numerical case and records
`numerical_status: rejected`. The separate evidence verifier checks the exact
state and 16,384 failed outputs, all other cases passing, complete resource
extracts and matching source hashes. An initial nonzero-exit rejection receipt is also
retained. Successful resource collection does not mean numerical acceptance. The first small
cancellation probe used one pair and a specialized tight tolerance; that probe
is preserved as `initial-cancellation-probe.json` but is superseded by the
uniform-tolerance repeated-pair final test. A wider initial held-out sweep
hit DrPerf's default 120-second timeout, preserved in `initial-timeout.json`.
The final bounded sweep uses two held-out cases and an explicit 450-second
execution limit.

## Scope and reproduction

The original kernel stores full WMMA tiles without edge guards. Source reading
therefore restricts this experiment to N divisible by128; arbitrary square
sizes are not claimed safe. This source-side limitation was identified before
execution, not discovered by resource fitting. No unsafe shape is launched.
N4096 was not emulated and the baseline's complete FP32 domain, overflow,
underflow, NaNs and infinities are not exhaustively validated.

```
python3 gpu/cases/kernelbench-matmul-fp32-debug/run.py
python3 gpu/cases/kernelbench-matmul-fp32-debug/verify_evidence.py
```

No GPU is used. The default toolkit is
`/home/ubuntu/GX/toolchains/cuda13`, overridable with `CUDA_HOME`.
[driver.cc](driver.cc) provides the independent oracle and input generation;
[upstream_src.py](upstream_src.py) and [upstream_ref.py](upstream_ref.py) are
untouched source downloads and are never imported. Generated PTX, cubins,
translated modules and raw JSONL traces are temporary and deleted. Retained
reports, compact metric/site extracts, diffs, ptxas diagnostics, hashes and
command receipts are under [results](results).

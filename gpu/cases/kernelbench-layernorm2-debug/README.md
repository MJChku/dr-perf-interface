# Public LayerNorm2: numerical gate rejects optimization

Source: [ScalingIntelligence/good-kernels, LayerNorm2](https://github.com/ScalingIntelligence/good-kernels/tree/5c94620c59ab9839a4ad4ccad309f3d91752d042/LayerNorm2),
revision `5c94620c59ab9839a4ad4ccad309f3d91752d042`.
The published reference uses PyTorch LayerNorm on 16×4,194,304 FP32 values,
with input mean 1,000 and standard deviation 0.1. The implementation uses four
cooperative CTAs per sample and per-thread/CTA Welford statistics.
No defect or inefficiency was injected. **This is a rejected candidate, not a
performance success. No physical GPU was allocated or timing claimed for it.**

## Compatibility and CPU execution

[before.cu](before.cu) preserves the published device source, removing only
framework includes and the host wrapper. The runner checks this extraction
against [upstream_src.py](upstream_src.py); the
[reference](upstream_ref.py) is saved and never imported. Initial translation
fails because module-scope `__device__` arrays are referenced as undeclared
C++ variables in generated code. The
[failed translation receipt](results/initial-translation-failure.json) is retained.

[compatible.cu](compatible.cu) replaces the four module-scope arrays with
explicit scratch-pointer arguments. The scratch storage has one entry per
sample; synchronization, arithmetic, indexing and launch settings remain as
published. [The diff](results/before-to-compatible.patch) makes this ABI
adaptation explicit. No DrPerf/GX implementation was changed. Cooperative grid
barriers execute with all CTAs resident in the CPU runtime.

The CPU harness uses one/two samples, widths 128/256 for training and
4,096/8,192 for held-out checks, 256 threads and four CTAs per sample.
Data patterns are seeded finite random values, a constant, and shifted
1,024-element segments. It uses nontrivial affine weights/biases, an independent
double-precision two-pass variance oracle, input immutability and output guards.
The 12 small tests pass with maximum error below 3e-7, but **five of 12
held-out tests fail**, with maximum absolute error **47.2374**. A random case
also fails, so the failure is not confined to the shifted pattern.

Source inspection explains the failure: the kernel atomically sums each
CTA's local Welford M2, but omits the correction for different CTA means.
The required global M2 includes `count_cta * (mean_cta - mean_global)^2`.
When only CTA zero has data at small widths, that omitted term is zero; at
larger widths, CTA means differ. A shifted pattern makes the missing variance
term conspicuous. This was discovered by numerical execution and the oracle,
**not DrPerf's resource fitting**, and was not reproduced on physical hardware.

## What DrPerf did and did not establish

DrPerf captures the small, numerically valid training states and exposes the
extra reduction and statistics traffic. Those small states are insufficient
to predict general multi-CTA numerical validity. Optimizing requests before
resolving this error would compare an invalid normalization algorithm, so no
optimized variant or timing is presented.

The failing held-out command returns exit one as a numerical gate should.
A second, explicitly named `heldout-diagnostic` capture reports the same
failed checks but returns zero solely to retain resource fits. Its
`validity: complete` refers to resource capture, not numerical correctness.
Rejected resource fits remain in the compact extract. No claim is made that
DrPerf diagnosed the missing statistical correction.

## Evidence and reproduction

```sh
python3 gpu/cases/kernelbench-layernorm2-debug/run.py
```

The runner expects the held-out numerical failure and checks that it occurs.
It uses no GPU, compiles SM80 PTX, removes generated modules/cubins/raw captures,
and saves compact receipts. `CUDA_HOME` defaults to
`/home/ubuntu/GX/toolchains/cuda13`.

- [Commands, hashes and numerical failures](results/verification.json)
- [Training resource report](results/compatible-report.txt)
- [Training compact evidence](results/compatible-train.json)
- [Failing held-out diagnostic resources](results/compatible-heldout-diagnostic.json)
- [Driver and independent oracle](driver.cc)

Limits: the original large shape and high-offset distribution were not run;
the demonstrated failure is at bounded shapes with the same multi-CTA
algorithm. Only widths divisible by four and non-null affine parameters were
tested. Passing the small set does not establish full LayerNorm correctness.

# Concrete GLM-5.3 kernel scaling

This standalone CPU/PTX experiment demonstrates why short tests can hide
history-dependent GPU work. It does not modify the serving stack or allocate
GPUs. Original sources are pinned in sources.json and source/; kernels.cu
retains the original bodies with only namespace/C-linkage adaptation. The
compact-output variant is an explicitly separate demonstration.

Reproduce from the drperf repository:

```sh
bin/drperf --gpu build
python3 gpu/examples/glm53_scaling/run.py --output gpu/build/glm53-scaling-reproduction
```

The checked-in PTX makes reproduction independent of CUDA/GPU access. To
regenerate it using the pinned local CUDA13 compiler:

```sh
/home/ubuntu/GX/toolchains/cuda13/bin/nvcc -std=c++17 -O2 \
  -arch=compute_103a -ptx gpu/examples/glm53_scaling/kernels.cu \
  -o gpu/examples/glm53_scaling/kernels.ptx
```

Output directories must be new. Each command records UTC timestamps, monotonic
duration, exact argv, exit status and logs. manifest.json pins source, driver,
PTX, translator and runtime hashes. Complete raw records and compressed reports
are retained. No host elapsed time is presented as GPU latency.

The initial interface for history materialization declares query count only.
The sparse-softmax interface declares selected count K only. DrPerf rejects
the affected write-byte models. Add history T and fit on32–512 histories:

| Kernel | Learned requested write bytes |
|---|---|
| Original materialization, reduced H=N=R=V=1 |12T|
| Original sparse softmax |2T+2K|
| Compact-output softmax demonstration |2K|

At T2048/K2048, original softmax writes8192bytes; at T100000/K2048,
it writes204096bytes, despite identical31230FP32 FLOPs and2048 special-function
operations. Compact output writes4096bytes in both cases. The one-query,
one-head materialization writes24576 versus1200000bytes. The two long states
are actual held-out PTX executions, not synthetic timing rows or merely
formula extrapolations. Independent numeric oracles and poisoned guards pass
for all45 launches. Learn on small states, then use analyze.py to verify14
held-out byte/SFU predictions exactly.

The selected probabilities are unchanged by compact representation on the
covered fixtures. Its consumer must use K-slot probabilities and the selected
IDs. Dropping the dense clear while continuing to call dense PV would be wrong.
No production integration, native performance claim or full-input proof follows
from these resource results. Requested bytes are not HBM traffic. Uniform finite
scores and valid unique IDs isolate the scaling effect; widths are deliberately
small. The original historical paths may no longer be used by the latest
FlashMLA serving candidate.

See [the measured report](results/REPORT.md), [structured results](results/summary.json),
[plot](results/scaling.svg), and [self-contained HTML/data](results/scaling.html).
The plot shows resource counts, not time. Initial exploratory developer runs
were under gpu/build; the final run.py invocation records the reproducible run.
The BF16/NVFP4 tcgen05 fixtures in gpu/tests are complementary: these two
scaling kernels do not themselves execute tcgen05.

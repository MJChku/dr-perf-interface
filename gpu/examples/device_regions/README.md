# Explain work inside a CUDA kernel

This example uses actual NVCC-generated PTX from
[`device_regions.cu`](../../tests/device_regions.cu) and an independent
[numerical harness](../../tests/device_regions_driver.cc). No GPU is used.
The checked-in PTX means NVCC is optional when reproducing the example.

```sh
bin/drperf --gpu build
python3 gpu/examples/device_regions/run.py --output gpu/build/inner-region-example
```

The script builds the harness, translates PTX, and collects two executions.
It retains command timestamps, durations, hashes, logs, captures, reports and
a small `summary.json`. Each output directory must be new.

The `sparse_regions` kernel contains two block regions:

- `dense_clear` zeros a history-sized FP32 buffer.
- `selected_math` performs one add per selected element.

With only `selected` as a PCV, DrPerf accepts the math's `selected` FLOP
formula but rejects the clear's write-byte explanation. Adding `history` to
that inner region gives exactly `global_write_bytes = 4 * history`. At fixed
selected count, changing history from 2,048 to 100,000 increases its requested
writes from 8,192 to 400,000 bytes (48.828125 times), while selected math stays
constant. This is a reduced FP32 fixture of the dense-output scaling pattern,
not a measurement of the serving stack or GPU latency.

For each selected count the harness includes histories 2,048, 100,000 and
`selected*selected`. Thus the mean history changes nonlinearly with the only
declared PCV in the incomplete annotation. With only a fully crossed grid of
the first two histories, their mean would be constant: the new mean-based
checker would correctly accept a constant and report the same-state spread.

An agent can refine the PCV expression, move the annotation, or optimize the
code and repeat. Identifying a history-dependent clear does not by itself
justify deleting it: changing to compact output requires adapting consumers.

The same fixture also demonstrates nested lane regions with data-dependent
PCVs, warp regions under block parents, partial warps, repeated scopes, and
explicit paired markers. Integration tests check their independent numerical
oracles, exclusive counts and parent relationships, and reject invalid marker
use. A separately compiled annotated tcgen05 fixture checks that BF16/NVFP4
semantic tile work reaches the inner region exactly once.

To regenerate the main fixture:

```sh
nvcc -std=c++17 -O2 -Igpu/include -arch=compute_103a -ptx \
  gpu/tests/device_regions.cu -o gpu/tests/device_regions.ptx
```

To regenerate the annotated tensor fixture:

```sh
nvcc -std=c++17 -O2 -Igpu/include -arch=compute_103a \
  -DDRPERF_GPU_TEST_DEVICE_REGIONS -ptx gpu/tests/tcgen05.cu \
  -o gpu/tests/device_tensor_regions.ptx
```

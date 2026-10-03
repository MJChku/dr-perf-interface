# Ditto: a workflow performance interface from the current Qwen capture

This is a composed **CPU-work interface**, organized around storing KV, restoring RAW or encoded KV, and advancing background work. It is a concrete first interface, not a complete latency predictor. The codebase was read to construct it; practitioners can read the interface below without tracing the implementation.

## Capture and scope

- Profile: [qwen.drperf.json](/home/ubuntu/compression/ditto_kv/example/qwen2.5B/qwen.drperf.json).
- Profile SHA-256: `84dd61a3e8c638048ce4af995a871412f98bc5fc0488ab29b9ea543ca450bf69`.
- Raw capture recorded in profile: `/home/ubuntu/compression/ditto_kv/exp/gx_kimi/runs/drperf-offline-qwen-cache-opt-20260924-r1/raw`.
- 105 regions; 39,623 traced application calls; complete trace; no reported measurement errors.
- 10/12 annotated source-file hashes match the current checkout; all 7 annotated cache-core files match. Changed adapter files: `src/integration/vllm/connector.py`, `src/integration/vllm/worker.py`.
- Capture manifest revision: `c82c4eb8789f73b0bba9ddddd315d95e3d5b1b88`; 5 of 64 manifest file hashes differ now: `src/integration/kv_layout.py`, `src/integration/vllm/compat.py`, `src/integration/vllm/connector.py`, `src/integration/vllm/spec.py`, `src/integration/vllm/worker.py`. Recorded numbers describe the captured binaries; source matches do not independently verify native binaries.
- GX functional emulation, with CUDA emulator calls excluded. No GPU execution time, DMA duration, or end-to-end latency is inferred. Unmarked worker-thread work is not followed in this capture.
- Marker-library blocks and explicit PCV regions are excluded; Python boundary overhead remains. No wrapper calibration is subtracted.
- `framework.fake_output` is emulator-only synthetic metadata construction. It is shown separately, not charged to the proposed production interface. Other host paths may still depend on synthetic codec output sizes.

## What the practitioner sees

```text
New KV from model execution
  -> STORE: place RAW rows, copy GPU -> host, register sources
  -> POLL: finish transfers; optionally encode and publish complete extents

Later cache hit
  -> RELOAD / submit_load
       RAW data     -> group copy runs -> copy host -> GPU
       encoded data -> bind decode units -> upload/decode into destination pages
       mixed data   -> both branches
  -> POLL: collect completion, release locks, advance compaction
```

There is no separate public `reload` method: restoration uses `submit_load`. Initial model loading and cache-miss prefill are outside this cache interface. A store submits RAW storage first; compression is deferred. Charging every encode to the immediate store call would miss that lifecycle.

| Semantic quantity | Meaning |
| --- | --- |
| B | Logical blocks / placements in this request |
| P | Physical pages transferred |
| R_raw, E | RAW read records and encoded read records/extents in the load plan |
| H | Distinct storage handles to lock |
| K | Contiguous RAW copy parts after grouping |
| D | RAW destinations processed while grouping |
| I_enc | `int(E > 0)`: whether decode setup is needed |
| L, U | Participating layer buffers; codec units bound across encoded extents |
| N_done, N_query | Completed jobs and completion queries during this poll |
| Pool state | Stream/descriptor/workspace reuse, growth, and first launch |

`O[region]` denotes its own CPU work, including its unexplained part; `F[region]` includes its child interfaces. A multiplier means that many applications at the actual child states, not that many copies of a global average. All displayed fitted coefficients are rounded to integers. Exact observed states, rather than a continuous box of inputs, define the checked domain.

## Store

For the successful codec-enabled store path exercised by this capture:

```text
F_store = O[rt.submit_store]
        + F[enc.record_tokens] + F[gc.invalidate_blocks]
        + F[ftl.bind_store_batch] + F[gc.trigger]
        + F[store.wait_conflicts] + F[store.execute]

F[store.execute] = O[store.execute]
                 + H * F[ftl.write_lock]
                 + B * F[ftl.ensure_row]
                 + F[carrier.transfer]
```

`H` is the number of distinct placement handles, not generally B or a fixed fraction of B. The existing parent PCVs do not name H, which explains its unresolved write-lock multiplier. This equation also needs a separate refusal/cleanup branch for capacity failures; those were not exercised here. The conflict-wait path had no nested drain calls in these stores, so concurrent conflicting stores remain outside its measured coverage.

The three own-work fits above combine, on this capture's observed states, to:

```text
10408*B + 117704 + unexplained_own
```

This is only the store wrapper, conflict check, and execution bookkeeping. Binding, token recording, locking, row materialization, transfer setup, and deferred compression remain explicit child costs. It is not the complete store cost.

## RAW, encoded, and mixed reload

```text
F_reload = O[rt.submit_load]
         + F[ftl.plan_load] + F[gc.trigger] + F[rt.arm_load]
         + N_stats * F[ftl.count] + F[load.execute]

F[load.execute] = O[load.execute] + H * F[ftl.lock] + F[load.plan]

F[load.plan] = O[load.plan]
            + F[load.raw_parts] + F[carrier.raw_load]
            + E * F[dec.prepare]
            + I_enc * (F[rt.acquire_stream] + F[dec.submit]
                       + F[rt.register_ready_pages])

F[carrier.raw_load] = O[carrier.raw_load] + K * F[carrier.h2d_part]
F[carrier.h2d_part] = O[carrier.h2d_part] + F[carrier.transfer]
F[dec.submit]      = O[dec.submit] + F[codec.decode_units]
```

The decode equation is for the exercised GPU decode path with nonempty units and source-registration callbacks enabled. Empty inputs, CPU decode, failures, and disabled callbacks need their own paths. `N_stats` counts runtime counter updates; it is not a compression parameter. In this capture the lock count equals the read count, but the source deduplicates handles, so H is the stronger semantic input.

Observed reloads: **38 RAW-only, 10 encoded-only, 70 mixed**.

The same four logical blocks exercised both of these paths:

| Path for B = 4 | RAW reads | Encoded reads | RAW copy parts | Calls observed |
| --- | ---: | ---: | ---: | ---: |
| RAW-only | 1 | 0 | 1 | 7 |
| encoded-only | 0 | 1 | 0 | 4 |

The encoded branch introduces decode preparation and submission even at the same request size. The RAW branch instead transfers the restored pages directly. This is a real reason to expose representation and extent counts, rather than promise a single formula in B.

### Branches checked from source and trace

These are source-derived workflow rules, checked against every recorded parent invocation. They are not new fitted drperf coefficients and do not modify the profile. In particular, `I_enc` resolves a Boolean branch that cannot be affine in E alone when E ranges over 0, 1, 2, 3. The RAW part-count rule is already affine, but the generic export declined it with too few distinct parent states.

| Parent | Child | Calls per parent | Invocations checked |
| --- | --- | --- | ---: |
| `load.plan` | `carrier.raw_load` | `1` | 118 |
| `load.plan` | `load.raw_parts` | `1` | 118 |
| `load.plan` | `dec.prepare` | `encoded` | 118 |
| `load.plan` | `dec.submit` | `has_encoded = int(encoded > 0)` | 118 |
| `load.plan` | `rt.acquire_stream` | `has_encoded = int(encoded > 0)` | 118 |
| `load.plan` | `rt.register_ready_pages` | `has_encoded = int(encoded > 0)` | 118 |
| `carrier.raw_load` | `carrier.h2d_part` | `parts` | 118 |

All 7,035 call-count checks across 28 rules passed, including the store and maintenance skeletons. Full trace nesting and per-state call counts were validated before composition.

## Completion and deferred compression

```text
F_poll = O[rt.poll]
       + N_query * F[rt.query] + N_done * F[rt.finish]
       + N_depart * F[rt.depart_restore] + N_stats * F[ftl.count]
       + I_settle * F[rt.queue_settle] + F_background

F_background = O[gc.run_trigger]
             + F[gc.publish] + F[ftl.reserve_landing] + F[gc.launch]

encode launch -> prepare candidates -> encode submission
publication  -> output copy -> witness check -> commit -> release
```

The poll coefficients above are actual event counts, not all current entry PCVs; the existing interface does not yet predict all of them. In this functional capture every pending job was completed during its poll, so the exporter learned `pending * F[rt.finish]`. Real asynchronous execution can leave jobs pending. Expose `N_done`, or a completion-state model, rather than generalize that correlation. Similarly, call frequency is controlled by the client; there is no fixed number of polls per request.

This run contains 957 polls, 184 job completions, 28 encode-batch submissions, and 28 commits. The run mixes initialization, pool reuse, and later restoration states.

The measured `gc.run_trigger` interface also includes `framework.fake_output`. That node is omitted only from the proposed production equation above; its recorded cost remains visible in the evidence below. Removing that node does not turn a functional-emulation trace into a timing-accurate production capture.

## What the measured formulas support

Each row is **own work only**, excluding child regions. Add its reported unexplained work and compose the children above. For unfitted regions, the observed mean is evidence at the sampled states, not a constant interface or a prediction. Unexplained percentages below are the export's per-state fit summary, rounded to whole percentages.

| Region | Own explained formula / fit status | Own observed mean | Unexplained share | Calls / states |
| --- | --- | ---: | --- | ---: |
| `rt.submit_store` | `671*entries + 69799` | 73,305 | 3% | 66 / 11 |
| `ftl.bind_store_batch` | `9898*entries + 7702*new_superblocks + 35*sb_blocks - 104*completes + 4*directory + 7932` | 51,763 | 18% | 66 / 50 |
| `store.wait_conflicts` | `1028*placements + 3844` | 6,668 | 0% | 66 / 11 |
| `store.execute` | `8709*placements + 44060` | 66,901 | 0% | 66 / 11 |
| `ftl.ensure_row` | `no affine fit` | 10,733 | not fitted | 190 / 1 |
| `ftl.write_lock` | `no affine fit` | 4,904 | not fitted | 90 / 1 |
| `carrier.transfer` | `803*pages + 32226*d2h + 303198` | 334,460 | 4% | 174 / 16 |
| `carrier.swap_launch` | `0*ops - 103440*d2h + 8142914*first + 129068` | 140,851 | 1% | 174 / 17 |
| `rt.submit_load` | `116*blocks + 8400*cold + 73634` | 75,913 | 3% | 118 / 15 |
| `ftl.plan_load` | `2580*blocks + 1191*superblocks - 88*sealed + 149*cold + 8230` | 31,114 | 9% | 118 / 84 |
| `load.execute` | `6650*reads + 62*blocks + 33934*encoded + 4487*cold + 18493` | 91,277 | 20% | 118 / 18 |
| `load.plan` | `384*raw + 10702*encoded + 1199*blocks + 41477*new_copy_stream + 55575` | 102,526 | 30% | 118 / 18 |
| `ftl.lock` | `no affine fit` | 18,151 | not fitted | 252 / 3 |
| `load.raw_parts` | `7404*reads + 10191*dests + 14269*cold + 25914` | 57,136 | 3% | 118 / 7 |
| `carrier.raw_load` | `no affine fit` | 13,734 | not fitted | 118 / 3 |
| `carrier.h2d_part` | `536*cold + 13581` | 13,579 | 0% | 108 / 6 |
| `dec.prepare` | `no affine fit` | 2,222,797 | not fitted | 140 / 2 |
| `rt.acquire_stream` | `no affine fit` | 17,042 | not fitted | 80 / 2 |
| `dec.submit` | `no affine fit` | 27,793 | not fitted | 80 / 4 |
| `codec.decode_units` | `151467*units - 58700*retained - 64468*free_slabs + 944171*soa_misses + 21480*cold + 1469167` | 7,693,973 | 5% | 80 / 13 |
| `rt.poll` | `no affine fit` | 27,668 | not fitted | 957 / 5 |
| `gc.run_trigger` | `31167` | 31,160 | 0% | 957 / 27 |
| `gc.launch` | `3*available + 8430` | 13,949 | 38% | 957 / 27 |
| `gc.publish` | `186505*batches - 179177*sizes + 9712` | 19,617 | 32% | 957 / 103 |
| `gc.batches` | `no affine fit` | 23,604 | not fitted | 163 / 4 |
| `enc.submit_batch` | `no affine fit` | 25,920 | not fitted | 28 / 1 |
| `codec.encode_units` | `no affine fit` | 6,723,303 | not fitted | 28 / 1 |
| `framework.fake_output` | `no affine fit` | 4,391,169 | not fitted | 28 / 1 |

Zero displayed percent can mean a small nonzero residual. Likewise a displayed `0*PCV` can be a tiny nonzero fitted coefficient rounded for readability. Neither display rounding nor a missing term proves zero physical cost.

## Transfer work is a separate interface

The transport source provides a semantic byte/descriptor model independently of CPU fits:

```text
RAW copy operations = sum over groups (pages_in_group * layer_buffers_in_group)
RAW bytes           = sum over groups (pages_in_group * bytes_per_page_across_buffers)
encoded input bytes = sum of prepared_decodes.wire_bytes
```

These are source-derived work counts, not measured bandwidth formulas. The RAW path computes `transfer_bytes` while constructing descriptors; the decode bridge totals stored stream bytes. Encoded input bytes are distinct from all end-to-end traffic: metadata, output placement, and other codec transfers can add traffic. This GX capture synthesizes codec output sizes, so it cannot establish a real compression ratio or GPU decode cost.

## What a practitioner can and cannot decide yet

- **Useful now:** distinguish RAW/encoded/mixed reloads; see that store and compression are separate lifecycle stages; identify per-request setup, per-block work, distinct-handle locking, and per-extent decode preparation; understand which costs move when the workload changes these quantities.
- **Not identified here:** effects of changing layer count or page geometry. The capture fixes 24 layer references and 8 pages per stored block; pages, rows, and codec units are correlated. Those fitted coefficients cannot independently price a different architecture or geometry.
- **Incomplete numerical interfaces:** decode preparation and encode submission lack sufficient state variation; load planning and compaction still have appreciable unexplained own work. Stream/workspace state also changes CPU cost. Neither an unresolved child multiplier nor a single-state mean should be presented as a complete formula.
- **Next measurements:** vary encoded extent count and partial load sizes independently; exercise multiple RAW parts, store handle layouts, layer counts, page sizes, and pool cold/reuse states; separate idle polls, incomplete jobs, and completed jobs under realistic event readiness; cover contention and refusal branches.
- **Next semantic PCVs:** `has_encoded`, distinct read/write handles, completed/query/departure counts for polling, and descriptor/workspace growth. Some are internal state summaries rather than knobs the practitioner sets; the interface should explain how configuration and workload produce them.

This already gives a useful structural performance interface. It does **not yet** justify a complete numeric model of compression-method trade-offs or a latency prediction. No scalar total across concurrent CPU/GPU work is constructed here.

## Source and observed-domain audit

The table below retains the region arguments and observed ranges so readers do not mistake fixed quantities for independently varied inputs. Ranges summarize discrete states; combinations inside a range may never have occurred.

| Region | Observed PCV ranges | Source |
| --- | --- | --- |
| `rt.submit_store` | `entries=1..13, groups=1, cold=0` | [src/cache/runtime.py:524](/home/ubuntu/compression/ditto_kv/src/cache/runtime.py:524) |
| `ftl.bind_store_batch` | `entries=1..13, rebinds=0, new_superblocks=0..4, sb_blocks=0..3, completes=0..3, directory=12..68, cold=0` | [src/cache/runtime.py:541](/home/ubuntu/compression/ditto_kv/src/cache/runtime.py:541) |
| `store.wait_conflicts` | `placements=1..13, cold=0` | [src/cache/store_engine.py:181](/home/ubuntu/compression/ditto_kv/src/cache/store_engine.py:181) |
| `store.execute` | `placements=1..13, pages=8..104, cold=0` | [src/cache/store_engine.py:197](/home/ubuntu/compression/ditto_kv/src/cache/store_engine.py:197) |
| `ftl.ensure_row` | `new=1, cold=0` | [src/cache/store_engine.py:240](/home/ubuntu/compression/ditto_kv/src/cache/store_engine.py:240) |
| `ftl.write_lock` | `cold=0` | [src/cache/store_engine.py:223](/home/ubuntu/compression/ditto_kv/src/cache/store_engine.py:223) |
| `carrier.transfer` | `pages=8..104, rows=1..13, d2h=0..1, refs=24, cold=0` | [src/cache/transport.py:95](/home/ubuntu/compression/ditto_kv/src/cache/transport.py:95) |
| `carrier.swap_launch` | `ops=192..2496, d2h=0..1, dma=0, first=0..1, cold=0` | [src/cache/transport.py:223](/home/ubuntu/compression/ditto_kv/src/cache/transport.py:223) |
| `rt.submit_load` | `blocks=1..14, groups=1, cold=0..1` | [src/cache/runtime.py:576](/home/ubuntu/compression/ditto_kv/src/cache/runtime.py:576) |
| `ftl.plan_load` | `blocks=1..14, superblocks=1..4, sealed=0..3, sb_blocks=1..14, directory=58..68, cold=0..1` | [src/cache/runtime.py:592](/home/ubuntu/compression/ditto_kv/src/cache/runtime.py:592) |
| `load.execute` | `reads=1..4, blocks=1..14, encoded=0..3, cold=0..1` | [src/cache/load_engine.py:368](/home/ubuntu/compression/ditto_kv/src/cache/load_engine.py:368) |
| `load.plan` | `raw=0..2, encoded=0..3, blocks=1..14, new_copy_stream=0..1, cold=0..1` | [src/cache/load_engine.py:411](/home/ubuntu/compression/ditto_kv/src/cache/load_engine.py:411) |
| `ftl.lock` | `encoded=0..1, cold=0..1` | [src/cache/load_engine.py:390](/home/ubuntu/compression/ditto_kv/src/cache/load_engine.py:390) |
| `load.raw_parts` | `reads=0..2, dests=0..5, cold=0..1` | [src/cache/load_engine.py:300](/home/ubuntu/compression/ditto_kv/src/cache/load_engine.py:300) |
| `carrier.raw_load` | `parts=0..1, cold=0..1` | [src/cache/transport.py:322](/home/ubuntu/compression/ditto_kv/src/cache/transport.py:322) |
| `carrier.h2d_part` | `pages=8..40, rows=1..5, cold=0..1` | [src/cache/transport.py:308](/home/ubuntu/compression/ditto_kv/src/cache/transport.py:308) |
| `dec.prepare` | `blocks=4, layers=24, cold=0..1` | [src/cache/extent_decode.py:49](/home/ubuntu/compression/ditto_kv/src/cache/extent_decode.py:49) |
| `rt.acquire_stream` | `new_stream=0..1, ring=0..1, cold=0..1` | [src/cache/load_engine.py:185](/home/ubuntu/compression/ditto_kv/src/cache/load_engine.py:185) |
| `dec.submit` | `batches=1..3, units=24..72, blocks=4..12, cold=0..1` | [src/cache/extent_decode.py:122](/home/ubuntu/compression/ditto_kv/src/cache/extent_decode.py:122) |
| `codec.decode_units` | `units=24..72, blocks=4..12, streams=4, offset=0..3, ranges=24..72, stored_kib=1536..4608, retained=0..1, free_slabs=4..8, soa_misses=0..1, cold=0..1` | [src/cache/extent_decode.py:158](/home/ubuntu/compression/ditto_kv/src/cache/extent_decode.py:158) |
| `rt.poll` | `pending=0..1, loads=0..1, settle=0..1, cold=0` | [src/cache/runtime.py:651](/home/ubuntu/compression/ditto_kv/src/cache/runtime.py:651) |
| `gc.run_trigger` | `pending=0..1, queued=1..24, cold=0` | [src/cache/compaction.py:1138](/home/ubuntu/compression/ditto_kv/src/cache/compaction.py:1138) |
| `gc.launch` | `queued=1..24, available=0..1, cold=0` | [src/cache/compaction.py:1092](/home/ubuntu/compression/ditto_kv/src/cache/compaction.py:1092) |
| `gc.publish` | `batches=0..1, members=0..1, sizes=0..1, landed=0..1, ready=0..1, ready_blocks=0..4, aborted=0, untimed=0..1, bindings=0..14, tracked=0..14, queued=1..24, cold=0` | [src/cache/compaction.py:957](/home/ubuntu/compression/ditto_kv/src/cache/compaction.py:957) |
| `gc.batches` | `keys=1..3, limit=1, complete=0..3, ready=0..1, evaluated=1, cold=0` | [src/cache/compaction.py:1038](/home/ubuntu/compression/ditto_kv/src/cache/compaction.py:1038) |
| `enc.submit_batch` | `members=1, units=24, cold=0` | [src/cache/extent_encode.py:218](/home/ubuntu/compression/ditto_kv/src/cache/extent_encode.py:218) |
| `codec.encode_units` | `units=24, cold=0` | [src/cache/extent_encode.py:252](/home/ubuntu/compression/ditto_kv/src/cache/extent_encode.py:252) |
| `framework.fake_output` | `units=24, cold=0` | [src/compressor/framework_test.py:732](/home/ubuntu/compression/ditto_kv/src/compressor/framework_test.py:732) |

Reproduce this document (reads the existing export; no GPU or new capture needed):

```sh
python3 /home/ubuntu/drperf/examples/explorer/ditto_workflows.py \
  --profile /home/ubuntu/compression/ditto_kv/example/qwen2.5B/qwen.drperf.json \
  --source-root /home/ubuntu/compression/ditto_kv \
  --output /home/ubuntu/drperf/examples/explorer/DITTO_WORKFLOWS.md
```

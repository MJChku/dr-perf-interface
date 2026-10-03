# vLLM case H2: one request's logprobs bills the whole batch

Markers: `mark_spec.py <tree> stop` (tree `vllm-cpu-spec`); driver
`driver.py mode=stop max_tokens=128 lp=1`: stop strings on all eight
requests, `logprobs=5` on half, `min_tokens=8` on two.

Study ("vLLM case H2"):

    gather_logprobs   1,009,495.2*num_reqs + 115,043.4    (913,749 per row in at::native::AVX2::topk_impl_loop)
    learn:            gather_logprobs.num_reqs = last(engine_step.unfinished)   exact at 118/118 triggers

`InputBatch.max_num_logprobs` is a batch-wide maximum, so `Sampler.forward`
runs `log_softmax` and `torch.topk` over the whole `[num_reqs, vocab]` matrix
as soon as one request asks.  `apply_logprobs_rows.py` makes `InputBatch`
publish the asking rows and the sampler run the top-k on those rows only,
scattering back into a full-height tensor; the rows nobody asked about are
never read.  Measured: the other-thread part of `gather_logprobs` 16.2M ->
11.5M per call (-29.1%), `sample` +250k for the gather and scatters, and no
wall-clock change at this batch size with eight OpenMP threads -- the honest
shape of this fix, which the formula states (at 64 requests with 8 asking it
is 66M instructions per step).

Equivalence: `SPEC_DUMP=` makes the driver dump token ids, text, finish and
stop reasons, cumulative logprob and the full top-5 logprob dicts; `run.sh`
compares base and fix.  Also check `check_stop`: 4,196.8 flat at 0.0%
irregular -- the suspected stop-string quadratic is not there in 0.28.

## Rerun (2026-10-02, `run.sh`, batches of 3, 4, 5, 6, 8 requests)

```
gather_logprobs, 8 requests, 4 asking     base   total 62.84M   own 48.22M   other 14.62M     (per call, 59 calls)
                                          fix    total 59.75M   own 48.23M   other 11.53M
sample (contains it)                      base   total 62.11M   ->   fix 66.18M   (+4.1M: the row gather and three scatters land on the workers)
```
The other-thread part falls 14.62M -> 11.53M (-21%; study 16.2M -> 11.5M,
-29%), the own-thread part is flat (the OpenMP pool spin), and the top-k is
done on 4 rows instead of 8.  After the fix `num_reqs` in `gather_logprobs`
is the number of asking rows, so `run.sh` pairs the runs by grid point rather
than by state.  Token ids, text, stop reasons and the top-5 logprob dicts of
every request identical.  In this environment all requests of a batch finish
at the same step, so the grid supplies the batch sizes the study got from
staggered completions.

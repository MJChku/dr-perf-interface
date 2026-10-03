# Inferix Self-Forcing control metadata audit

Source: Inferix commit `77170deca738914c1999bc9e48eea84acb534db3`.
The source hash for each adapted file is locked in
`examples/causal_video/inferix_metadata.py`; the adapter writes before/after
SHA-256 values to `inferix-metadata-adaptation.json` in the target checkout.

## Safe host-derived metadata

* `inferix/models/self_forcing/wrapper.py`, `WanTextEncoder.forward`: the
  tokenizer creates `ids, mask` on CPU, then moves both to GPU and derives
  `seq_lens = mask.gt(0).sum(dim=1).long()`. The lengths are used as Python
  slice bounds while zeroing T5 context padding. The adapter computes lengths
  from the original CPU mask before transferring the unchanged mask to T5.
  This preserves the exact positive-mask counting rule and context operation.
* `inferix/pipeline/self_forcing/CausalInferencePipeline.py` and
  `CausalDiffusionInferencePipeline.py`: `global_end_index` and
  `local_end_index` are initialized/reset to host zero. The attention block
  reads them through `.item()` for cache branch and slice bounds, then calls
  `.fill_(current_end)`/`.fill_(local_end_index)` with host-derived positions.
  The adapter creates only these two scalar tensors on CPU. GPU K/V tensors,
  attention, cache content, and computed latents remain on their original
  devices and execute unchanged. The adapter preserves the existing eviction,
  sink, and append branch formulas.

Run `python3 examples/causal_video/inferix_metadata.py --validate` to check
CPU mask slicing and cache counter transitions. Run `--check --root PATH` to
verify the exact original sources and `--apply --root PATH` to adapt a separate
Inferix checkout. `--apply` deliberately refuses drift or a second application.
Use the same adapted checkout for native and GX comparisons. Keep a separate
unmodified native measurement and label the comparison against this adaptation
as an **adapted native baseline**, since removing GPU `.item()` synchronizations
changes native timing even when tensor outputs match.

## Control reads still requiring investigation

* `inferix/models/schedulers/flow_match.py::step` computes `timestep_id`
  through a GPU `argmin` and branches on `.any()`. Its sigma selection then
  indexes model tensors. This is meaningful scheduler logic; the adapter does
  not substitute a fabricated index or remove the branch.
* `inferix/models/wan_base/utils/fm_solvers.py` and `fm_solvers_unipc.py`
  use `(schedule_timesteps == timestep).nonzero()` followed by `.item()` in
  `index_for_timestep`; the Self-Forcing diffusion pipeline calls one of these
  solvers. A GX run that skips their GPU kernels can obtain an invalid step
  index. Preserve native execution or derive the index from the original
  host schedule with an independently validated, solver-specific change.
* `inferix/models/self_forcing/causal_model.py` has `seq_lens[0].item()` in
  the training/no-cache branch and `grid_sizes[0][1:]` reduction followed by
  `.item()` in cached attention. `seq_lens` and `grid_sizes` are constructed
  from host tensor shapes in the model. These are not included in the narrow
  adapter because they currently stay on CPU in the inspected path; verify
  their devices in the actual configuration.
* `.cpu()` of generated video and VAE outputs in the pipeline is model data
  readback, not control metadata. It cannot be made meaningful when computation
  is skipped. A GX timing run may stop before export, while native quality
  validation must retain full output computation.

The GX `partial_sync` report is an optimistic approximate dependency timeline,
not a guaranteed hardware lower bound. Prediction misses contribute zero
modeled GPU time and must accompany every reported virtual interval.

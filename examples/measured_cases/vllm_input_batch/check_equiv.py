"""Randomized equivalence check: modified InputBatch vs the pristine one.

Loads /home/ubuntu/drperf-cases/patches/base_batch/gpu_input_batch.py (the
marked-but-unmodified file) as a second module and replays the same
add_request / remove_request / condense / refresh_metadata sequence on both,
comparing every array, list, dict and the derived SamplingMetadata.
"""
import importlib.util
import os
import random
import sys

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors
import vllm.v1.worker.gpu_input_batch as NEW

_spec = importlib.util.spec_from_file_location(
    "base_gpu_input_batch", os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_batch", "gpu_input_batch.py")
)
OLD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OLD)

MAX_REQS, MAX_LEN, VOCAB = 16, 512, 50272
ARRAYS = ["token_ids_cpu", "is_token_ids", "num_tokens_no_spec", "num_prompt_tokens",
          "num_computed_tokens_cpu", "request_lora_mapping", "temperature_cpu",
          "top_p_cpu", "top_k_cpu", "frequency_penalties_cpu", "presence_penalties_cpu",
          "repetition_penalties_cpu", "num_accepted_tokens_cpu"]


def make_batch(mod, batch_move_min=None):
    b = mod.InputBatch(
        max_num_reqs=MAX_REQS, max_model_len=MAX_LEN, max_num_batched_tokens=256,
        device=torch.device("cpu"), vocab_size=VOCAB, block_sizes=[16],
        kernel_block_sizes=[16], max_num_blocks_per_req=[MAX_LEN // 16],
        logitsprocs=LogitsProcessors(),
    )
    if batch_move_min is not None:
        b._BATCH_MOVE_MIN = batch_move_min
    for name in ARRAYS:
        getattr(b, name)[...] = 0
    # Device-side tensors also come from torch.empty(); when the corresponding
    # no_* flag is set copy_slice() never touches them, so zero them as well.
    for name in ("temperature", "top_p", "top_k", "frequency_penalties",
                 "presence_penalties", "repetition_penalties"):
        getattr(b, name).zero_()
    for t in b.block_table.block_tables:
        t.block_table.np[...] = 0
        t.num_blocks_per_row[...] = 0
    return b


def make_req(mod, rng, i, resumed):
    n = rng.randrange(4, 60)
    sp = SamplingParams(
        temperature=rng.choice([0.0, 0.7, 1.3]),
        top_p=rng.choice([1.0, 0.9]),
        top_k=rng.choice([-1, 20]),
        frequency_penalty=rng.choice([0.0, 0.5]),
        presence_penalty=rng.choice([0.0, 0.3]),
        repetition_penalty=rng.choice([1.0, 1.1]),
        max_tokens=8,
    )
    return mod.CachedRequestState(
        req_id="r%d" % i,
        prompt_token_ids=[rng.randrange(VOCAB) for _ in range(n)],
        mm_features=[], sampling_params=sp, generator=None,
        block_ids=([rng.randrange(1, 100) for _ in range((n + 15) // 16)],),
        num_computed_tokens=rng.randrange(n + 1),
        output_token_ids=[rng.randrange(VOCAB) for _ in range(rng.randrange(6) if resumed else 0)],
    )


def compare(a, b, tag):
    for name in ARRAYS:
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            bad = np.argwhere(getattr(a, name) != getattr(b, name))
            sys.exit("MISMATCH %s: %s at %s" % (tag, name, bad[:5].tolist()))
    for name in ["_req_ids", "req_output_token_ids", "spec_token_ids"]:
        if getattr(a, name) != getattr(b, name):
            sys.exit("MISMATCH %s: %s" % (tag, name))
    if a.req_id_to_index != b.req_id_to_index:
        sys.exit("MISMATCH %s: req_id_to_index" % tag)
    for g in range(len(a.block_table.block_tables)):
        ta, tb = a.block_table.block_tables[g], b.block_table.block_tables[g]
        if not np.array_equal(ta.block_table.np, tb.block_table.np):
            sys.exit("MISMATCH %s: block_table[%d]" % (tag, g))
        if not np.array_equal(ta.num_blocks_per_row, tb.num_blocks_per_row):
            sys.exit("MISMATCH %s: num_blocks_per_row[%d]" % (tag, g))
    if a.batch_update_builder.moved != b.batch_update_builder.moved:
        sys.exit("MISMATCH %s: moved" % tag)
    sa, sb = a.sampling_metadata, b.sampling_metadata
    for f in ("all_greedy", "all_random", "no_penalties", "max_num_logprobs",
              "logprob_token_ids", "output_token_ids", "spec_token_ids",
              "bad_words_token_ids"):
        if getattr(sa, f) != getattr(sb, f):
            sys.exit("MISMATCH %s: sampling_metadata.%s" % (tag, f))
    for f in ("temperature", "top_p", "top_k", "prompt_token_ids",
              "frequency_penalties", "presence_penalties", "repetition_penalties",
              "allowed_token_ids_mask"):
        x, y = getattr(sa, f), getattr(sb, f)
        if (x is None) != (y is None):
            sys.exit("MISMATCH %s: sampling_metadata.%s presence" % (tag, f))
        if x is not None and not torch.equal(x, y):
            sys.exit("MISMATCH %s: sampling_metadata.%s" % (tag, f))


moves = {}
for trial in range(150):
    rng = random.Random(trial)
    new_b, old_b = make_batch(NEW, 1), make_batch(OLD)
    live = []
    for step in range(12):
        n_add = min(MAX_REQS - len(live), rng.randrange(0, MAX_REQS + 1))
        for k in range(n_add):
            i = trial * 1000 + step * 100 + k
            resumed = rng.random() < 0.3
            new_b.add_request(make_req(NEW, random.Random(i), i, resumed))
            old_b.add_request(make_req(OLD, random.Random(i), i, resumed))
            live.append("r%d" % i)
        for v in rng.sample(live, rng.randrange(0, len(live) + 1)):
            new_b.remove_request(v)
            old_b.remove_request(v)
            live.remove(v)
        new_b.condense()
        old_b.condense()
        m = len(new_b.batch_update_builder.moved)
        moves[m] = moves.get(m, 0) + 1
        new_b.refresh_metadata()
        old_b.refresh_metadata()
        compare(new_b, old_b, "trial %d step %d" % (trial, step))

print("InputBatch equivalence OK (new vs pristine); moved-row counts:",
      dict(sorted(moves.items())))

# vLLM case B: admitting a request

Eleven regions on the admission and teardown path (`mark_req.py`), driver
`driver.py`: 16 prompts from 12 to 662 tokens, four output tokens each so
admission dominates, and a second `generate()` over the same prompts so
anything cached across calls shows as a difference between the passes.

Study (`CASES.md`, "vLLM case B"):

    process_inputs       315.6*num_tokens + 372,902.8   ->   109.2*num_tokens + 386,776.3
    engine_add_request   544.8*num_tokens + 603,204.4   ->   227.5*num_tokens + 618,328.6

* the per-token slope was `max(prompt_ids)`/`min(prompt_ids)`, two Python
  scans of a list the tokenizer just produced; the fix is one C pass,
  `array("L", prompt_ids)`, falling back to the original scans and messages
  when it fails.  Three other candidates were measured under drperf and
  rejected (`np.asarray` 332 per token, worse than the scans it replaces);
* the 373k constant is `SamplingParams.clone()` = `copy.deepcopy`, plus
  validation; it is reported, not removed by this patch;
* `FastIncrementalDetokenizer` primed its stream with the whole prompt; the
  fix primes with the last seven tokens, the constant the slow path already
  uses -- provably equivalent for opt-125m's byte-level decoder (checked
  against all 50,265 vocabulary ids and 48k random sequences), decoder-specific.

`request_init.patch` is the second beat of the same case (lazy block hashes,
`request_init` 254,024 -> 29,128 at 662 tokens): it moves 692k per 2k prompt
off admission onto `schedule`, it does not remove work, and is kept here for
the record but not applied by `run.sh`.

Equivalence: `DUMP=` makes the driver write every generated text and token-id
list of both passes; `run.sh` compares base and fix byte for byte.

## Rerun (2026-10-02, `run.sh`)

```
process_inputs      base  318*num_tokens + 377,447   [num_tokens >= 62]   3% irregular
                    fix   115*num_tokens + 390,124   [num_tokens >= 62]   8%
engine_add_request  base  124*num_tokens + 120,930   (own cost, nested regions excluded)
                    fix    -1*num_tokens + 141,530   [num_tokens >= 62]
```
The study's 315.6 -> 109.2 per token reproduces; `engine_add_request` is
reported as its own cost now (the study's 544.8 -> 227.5 was inclusive of
`process_inputs` and the others), and its own per-token term is the
detokenizer priming, which the fix removes entirely.  Both passes' texts and
token ids identical.

"""Insert perfmark regions for speculative decoding (set `spec`) and for the
stop-string / logprobs / logits-processor paths (set `stop`) into a vLLM tree.

    python mark_spec.py /home/ubuntu/drperf-cases/vllm-cpu-spec spec [--undo]
    python mark_spec.py /home/ubuntu/drperf-cases/vllm-cpu-spec stop [--undo]

Annotation only: each marked body is kept verbatim, one indent level deeper,
inside `with perfmark.region(...)`.  Backups are kept as <file>.<set>.orig.
Reuses wrap_method() from /home/ubuntu/drperf/examples/vllm_cpu/mark.py.
"""
import importlib.util
import os
import re
import shutil
import sys

_spec = importlib.util.spec_from_file_location(
    "vllm_mark_ref", "/home/ubuntu/drperf/examples/vllm_cpu/mark.py")
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
wrap_method = _ref.wrap_method

NDRAFT = "sum(map(len, scheduler_output.scheduled_spec_decode_tokens.values()))"

# (file, class or None, function, region, states)
SETS = {
    "spec": [
        ("vllm/v1/spec_decode/ngram_proposer.py", "NgramProposer", "propose", "ngram_propose",
         "num_reqs=len(sampled_token_ids), k=num_speculative_tokens, "
         "ctx=int(num_tokens_no_spec[:len(sampled_token_ids)].sum()), tag=9002"),
        ("vllm/v1/sample/rejection_sampler.py", "RejectionSampler", "forward", "reject_sample",
         "num_reqs=len(metadata.num_draft_tokens), num_draft=sum(metadata.num_draft_tokens), "
         "max_spec=metadata.max_spec_len, tag=9005"),
        ("vllm/v1/worker/gpu_model_runner.py", "GPUModelRunner", "_update_states", "update_states",
         "num_reqs=len(scheduler_output.num_scheduled_tokens), "
         "num_new=len(scheduler_output.scheduled_new_reqs), num_draft=" + NDRAFT + ", tag=9006"),
        ("vllm/v1/worker/gpu_model_runner.py", "GPUModelRunner", "_prepare_inputs", "prepare_inputs",
         "num_reqs=len(scheduler_output.num_scheduled_tokens), "
         "num_tokens=scheduler_output.total_num_scheduled_tokens, num_draft=" + NDRAFT + ", tag=9007"),
    ],
    "stop": [
        ("vllm/v1/engine/detokenizer.py", "BaseIncrementalDetokenizer", "update", "detok_update",
         "num_new=len(new_token_ids), have=len(self.output_text), toks=self.num_output_tokens(), tag=9010"),
        ("vllm/v1/engine/detokenizer.py", None, "check_stop_strings", "check_stop",
         "have=len(output_text), new_chars=new_char_count, nstop=len(stop), tag=9011"),
        ("vllm/v1/engine/logprobs.py", "LogprobsProcessor", "_update_sample_logprobs", "logprobs_update",
         "num_logprobs=(self.num_logprobs or 0), num_new=len(logprobs_lists[0]), "
         "have=len(self.logprobs or ()), tag=9012"),
        ("vllm/v1/sample/sampler.py", "Sampler", "gather_logprobs", "gather_logprobs",
         "num_reqs=logprobs.shape[0], num_logprobs=num_logprobs, tag=9013"),
        ("vllm/v1/sample/logits_processor/builtin.py", "MinTokensLogitsProcessor", "apply",
         "min_tokens_apply", "num_reqs=logits.shape[0], num_affected=len(self.min_toks), tag=9014"),
        ("vllm/v1/sample/logits_processor/builtin.py", "MinTokensLogitsProcessor", "update_state",
         "min_tokens_update",
         "num_affected=len(self.min_toks), has_update=(0 if batch_update is None else 1), tag=9015"),
        ("vllm/v1/sample/logits_processor/builtin.py", "LogitBiasLogitsProcessor", "apply",
         "logit_bias_apply", "num_reqs=logits.shape[0], num_affected=len(self.biases), tag=9016"),
        ("vllm/v1/sample/logits_processor/builtin.py", "LogitBiasLogitsProcessor", "update_state",
         "logit_bias_update",
         "num_affected=len(self.biases), has_update=(0 if batch_update is None else 1), tag=9017"),
    ],
}

# Textual insertions: (set, file, marker-region, old, new).  Used where a
# whole method is the wrong granularity.
NGRAM = "vllm/v1/spec_decode/ngram_proposer.py"
TEXT = [
    ("spec", NGRAM, "ngram_scan", """            batch_propose_numba(
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                self.min_n,
                self.max_n,
                self.max_model_len,
                k,
                self.valid_ngram_draft,
                self.valid_ngram_num_drafts,
            )
""", """            with perfmark.region("ngram_scan", num_valid=num_ngram_requests,
                                 ctx=int(num_tokens_no_spec[valid_ngram_requests].sum()), k=k,
                                 tag=9003):
                batch_propose_numba(
                    valid_ngram_requests,
                    num_tokens_no_spec,
                    token_ids_cpu,
                    self.min_n,
                    self.max_n,
                    self.max_model_len,
                    k,
                    self.valid_ngram_draft,
                    self.valid_ngram_num_drafts,
                )
"""),
    ("spec", NGRAM, "ngram_gather", """        for i in range(num_requests):
            if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(
                    self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
                )
            else:
                draft_token_ids.append([])
""", """        with perfmark.region("ngram_gather", num_reqs=num_requests,
                             num_valid=len(valid_ngram_requests), tag=9004):
            for i in range(num_requests):
                if i in valid_ngram_requests and self.valid_ngram_num_drafts[i] > 0:
                    draft_token_ids.append(
                        self.valid_ngram_draft[i, : self.valid_ngram_num_drafts[i]].tolist()
                    )
                else:
                    draft_token_ids.append([])
"""),
]

# Existing regions whose declared states change for a set.
RESTATE = [
    ("spec", "vllm/v1/core/sched/scheduler.py", "update_from_output",
     "running=len(self.running), num_draft=" + NDRAFT +
     ", num_accepted=(sum(map(len, model_runner_output.sampled_token_ids)) "
     "- sum(1 for _t in model_runner_output.sampled_token_ids if _t)), tag=9008"),
    ("spec", "vllm/v1/sample/sampler.py", "sample",
     "num_reqs=logits.shape[0], tag=9001"),
    ("stop", "vllm/v1/core/sched/scheduler.py", "update_from_output",
     "running=len(self.running), "
     "num_reqs=len(scheduler_output.num_scheduled_tokens), tag=9018"),
    ("stop", "vllm/v1/sample/sampler.py", "sample",
     "num_reqs=logits.shape[0], tag=9001"),
]


def wrap_function(src, func, region, state):
    """Wrap a module-level `def func(...)` body (indent 4) in a region."""
    lines = src.splitlines(True)
    i = 0
    while i < len(lines):
        if re.match(r"def %s\(" % re.escape(func), lines[i]):
            j = i
            depth = 0
            while True:
                depth += lines[j].count("(") - lines[j].count(")")
                if depth <= 0 and lines[j].rstrip().endswith(":"):
                    break
                j += 1
            k = j + 1
            if lines[k].lstrip().startswith(('"""', "'''")):
                q = lines[k].lstrip()[:3]
                if lines[k].count(q) >= 2 and len(lines[k].strip()) > 3:
                    k += 1
                else:
                    k += 1
                    while q not in lines[k]:
                        k += 1
                    k += 1
            end = k
            while end < len(lines):
                l = lines[end]
                if l.strip() and (len(l) - len(l.lstrip())) == 0:
                    break
                end += 1
            if any('perfmark.region("%s"' % region in l for l in lines[k:end]):
                return src, False
            head = '    with perfmark.region("%s", %s):\n' % (region, state)
            body = ["    " + l if l.strip() else l for l in lines[k:end]]
            lines[k:end] = [head] + body
            return "".join(lines), True
        i += 1
    raise SystemExit("mark_spec.py: function %s not found" % func)


def restate(src, region, state):
    pat = re.compile(r'^([ \t]*with perfmark\.region\("%s", )(.*)(\):)$'
                     % re.escape(region), re.M)
    m = pat.search(src)
    if not m:
        raise SystemExit("mark_spec.py: marker for region %s not found" % region)
    if m.group(2) == state:
        return src, False
    return src[:m.start(2)] + state + src[m.end(2):], True


def fix_future(src):
    """`import perfmark` must not precede a `from __future__` import."""
    lines = src.splitlines(True)
    if "import perfmark\n" not in lines:
        return src
    fut = [i for i, l in enumerate(lines) if l.startswith("from __future__ ")]
    if not fut:
        return src
    i = lines.index("import perfmark\n")
    if i > max(fut):
        return src
    del lines[i]
    fut = [j for j, l in enumerate(lines) if l.startswith("from __future__ ")]
    lines.insert(max(fut) + 1, "import perfmark\n")
    return "".join(lines)


def ensure_import(src):
    if "import perfmark\n" in src:
        return src
    m = re.search(r"^(?:from|import) .*\n", src, re.M)
    pos = m.start() if m else 0
    return src[:pos] + "import perfmark\n" + src[pos:]


def main():
    root = sys.argv[1]
    name = sys.argv[2]
    undo = "--undo" in sys.argv
    marks = SETS[name]
    text = [t for t in TEXT if t[0] == name]
    rest = [r for r in RESTATE if r[0] == name]
    files = []
    for e in marks:
        if e[0] not in files:
            files.append(e[0])
    for _, f, _, _, _ in text:
        if f not in files:
            files.append(f)
    for _, f, _, _ in rest:
        if f not in files:
            files.append(f)
    if undo:
        for path in files:
            full = os.path.join(root, path)
            keep = full + ".%s.orig" % name
            if os.path.exists(keep):
                shutil.move(keep, full)
                print("restored", path)
        return
    for path in files:
        full = os.path.join(root, path)
        keep = full + ".%s.orig" % name
        if not os.path.exists(keep):
            shutil.copy(full, keep)
    for path, cls, method, region, state in marks:
        full = os.path.join(root, path)
        src = open(full).read()
        if cls is None:
            out, changed = wrap_function(src, method, region, state)
            if changed:
                out = ensure_import(out)
        else:
            out, changed = wrap_method(src, cls, method, region, state)
        if changed:
            open(full, "w").write(fix_future(out))
            print("marked %s %s -> %s" % (path, method, region))
        else:
            print("already marked %s %s" % (path, method))
    for _, path, region, old, new in text:
        full = os.path.join(root, path)
        src = open(full).read()
        if new in src:
            print("already inserted %s" % region)
            continue
        if old not in src:
            raise SystemExit("mark_spec.py: anchor for %s not found in %s" % (region, path))
        src = fix_future(ensure_import(src.replace(old, new, 1)))
        open(full, "w").write(src)
        print("inserted region %s in %s" % (region, path))
    for _, path, region, state in rest:
        full = os.path.join(root, path)
        src = open(full).read()
        out, changed = restate(src, region, state)
        if changed:
            open(full, "w").write(out)
            print("re-declared %s" % region)
        else:
            print("already declared %s" % region)


if __name__ == "__main__":
    main()

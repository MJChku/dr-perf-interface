"""Insert perfmark regions into a vLLM source tree (editable install).

    python examples/vllm_cpu/mark.py third_party/vllm-cpu/vllm [--undo]

Each marked method gets its whole body wrapped in
`with perfmark.region("<name>", <symbol>=<expr>):` so that `drperf run --model`
can analyse its structure and capture its free variables.  Idempotent; --undo
restores the files from the .orig copies it keeps.
"""
import os
import re
import shutil
import sys

MARKS = [
    # file, class, method, region name, declared states (integer keywords; all
    # of them form the key, the cost formula is derived in all of them)
    ("vllm/v1/core/sched/scheduler.py", "Scheduler", "schedule", "schedule",
     "running=len(self.running), waiting=len(self.waiting)"),
    ("vllm/v1/core/sched/scheduler.py", "Scheduler", "update_from_output", "update_from_output",
     "running=len(self.running), num_tokens=scheduler_output.total_num_scheduled_tokens"),
    ("vllm/v1/worker/gpu_model_runner.py", "GPUModelRunner", "execute_model", "execute_model",
     "num_tokens=scheduler_output.total_num_scheduled_tokens, num_reqs=len(scheduler_output.num_scheduled_tokens)"),
    ("vllm/v1/sample/sampler.py", "Sampler", "forward", "sample", "num_reqs=logits.shape[0]"),
    ("vllm/v1/attention/backends/cpu_attn.py", "CPUAttentionBackendImpl", "forward", "attention",
     "num_tokens=query.shape[0], kv_tokens=(int(attn_metadata.seq_lens.sum()) if attn_metadata is not None else 0)"),
    # the engine loop: one region around the whole generate loop, one per step
    ("vllm/entrypoints/offline_utils.py", "OfflineInferenceMixin", "_run_engine", "generate_loop",
     "unfinished=self.llm_engine.get_num_unfinished_requests()"),
    ("vllm/v1/engine/llm_engine.py", "LLMEngine", "step", "engine_step", "unfinished=self.get_num_unfinished_requests()"),
]


def wrap_method(src, cls, method, region, state):
    lines = src.splitlines(True)
    in_class = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"class %s\b" % re.escape(cls), line):
            in_class = True
        elif re.match(r"class \w", line):
            in_class = False
        if in_class and re.match(r"    def %s\(" % re.escape(method), line):
            # find end of signature (line ending with ':' at nesting depth 0)
            j = i
            depth = 0
            while True:
                depth += lines[j].count("(") - lines[j].count(")")
                if depth <= 0 and lines[j].rstrip().endswith(":"):
                    break
                j += 1
            body_start = j + 1
            # skip a docstring
            k = body_start
            if lines[k].lstrip().startswith(('"""', "'''")):
                q = lines[k].lstrip()[:3]
                if lines[k].count(q) >= 2 and len(lines[k].strip()) > 3:
                    k += 1
                else:
                    k += 1
                    while q not in lines[k]:
                        k += 1
                    k += 1
            # body ends at the next line with indentation <= 4 that is not blank
            end = k
            while end < len(lines):
                l = lines[end]
                if l.strip() and (len(l) - len(l.lstrip())) <= 4:
                    break
                end += 1
            if any("perfmark.region(\"%s\"" % region in l for l in lines[k:end]):
                return src, False
            head = "        with perfmark.region(\"%s\", %s):\n" % (region, state)
            body = ["    " + l if l.strip() else l for l in lines[k:end]]
            lines[k:end] = [head] + body
            out = "".join(lines)
            if "import perfmark\n" not in out:
                # after the first block of imports
                m = re.search(r"^(?:from|import) .*\n", out, re.M)
                pos = m.start() if m else 0
                out = out[:pos] + "import perfmark\n" + out[pos:]
            return out, True
        i += 1
    raise SystemExit("mark.py: %s.%s not found in %s" % (cls, method, region))


def main():
    root = sys.argv[1]
    undo = "--undo" in sys.argv
    for path, cls, method, region, state in MARKS:
        full = os.path.join(root, path)
        orig = full + ".orig"
        if undo:
            if os.path.exists(orig):
                shutil.move(orig, full)
                print("restored", path)
            continue
        if not os.path.exists(orig):
            shutil.copy(full, orig)
        src = open(full).read()
        out, changed = wrap_method(src, cls, method, region, state)
        if changed:
            open(full, "w").write(out)
            print("marked %s.%s -> region %s (%s)" % (cls, method, region, state))
        else:
            print("already marked %s.%s" % (cls, method))


if __name__ == "__main__":
    main()

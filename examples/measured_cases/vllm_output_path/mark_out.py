"""Insert perfmark regions for the *output path* into a vLLM source tree.

    python mark_out.py /home/ubuntu/drperf-cases/vllm-cpu-out/vllm [--undo]

Reuses wrap_method() from /home/ubuntu/drperf/examples/vllm_cpu/mark.py.
Annotation only: every marked method keeps its body verbatim, one indent level
deeper, inside `with perfmark.region(...)`.
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

_TOT = "(self.detokenizer.num_output_tokens() if self.detokenizer is not None else 0)"

MARKS = [
    # --- engine side: per-request helper of the EngineCoreOutput loop ---------
    ("vllm/v1/core/sched/scheduler.py", "Scheduler", "_update_request_with_output",
     "req_update_output",
     "num_new=len(new_token_ids), total=request.num_output_tokens, "
     "prompt=request.num_prompt_tokens"),

    # --- output processor -----------------------------------------------------
    ("vllm/v1/engine/output_processor.py", "OutputProcessor", "process_outputs",
     "process_outputs",
     "num_outputs=len(engine_core_outputs), num_active=len(self.request_states)"),
    ("vllm/v1/engine/output_processor.py", "RequestState", "make_request_output",
     "make_request_output",
     "num_tokens=len(new_token_ids), total=%s, kind=self.output_kind.value, "
     "prompt=self.prompt_len" % _TOT),
    ("vllm/v1/engine/output_processor.py", "RequestState", "_new_completion_output",
     "completion_output",
     "num_tokens=len(token_ids), total=%s, kind=self.output_kind.value" % _TOT),
    ("vllm/v1/engine/output_processor.py", "RequestState", "_new_request_output",
     "request_output",
     "num_outputs=len(outputs), total=%s, prompt=self.prompt_len, "
     "kind=self.output_kind.value" % _TOT),

    # --- incremental detokenizer ---------------------------------------------
    ("vllm/v1/engine/detokenizer.py", "BaseIncrementalDetokenizer", "update",
     "detok_update",
     "num_new=len(new_token_ids), have=len(self.output_text), "
     "toks=self.num_output_tokens()"),
    ("vllm/v1/engine/detokenizer.py", "BaseIncrementalDetokenizer",
     "get_next_output_text", "detok_text",
     "have=len(self.output_text), delta=(1 if delta else 0), "
     "finished=(1 if finished else 0)"),
]

# The per-step output collection inside OfflineInferenceMixin._run_engine: a
# `with` around the `for output in step_outputs:` loop only (never per token).
COLLECT_FILE = "vllm/entrypoints/offline_utils.py"
COLLECT_OLD = """                step_outputs = self.llm_engine.step()
                for output in step_outputs:
"""
COLLECT_NEW = """                step_outputs = self.llm_engine.step()
                with perfmark.region("collect_step", num_outputs=len(step_outputs),
                                     done=len(outputs)):
                  for output in step_outputs:
"""


def patch_collect(root, undo=False):
    full = os.path.join(root, COLLECT_FILE)
    src = open(full).read()
    if undo or COLLECT_OLD not in src:
        print("collect_step: nothing to do" if COLLECT_OLD not in src
              else "collect_step: skipped")
        return
    i = src.index(COLLECT_OLD)
    # indent the whole `for` body by 2 spaces (the extra `with` level)
    tail = src[i + len(COLLECT_OLD):]
    lines = tail.splitlines(True)
    end = 0
    while end < len(lines):
        l = lines[end]
        if l.strip() and (len(l) - len(l.lstrip())) <= 16:
            break
        end += 1
    body = ["  " + l if l.strip() else l for l in lines[:end]]
    src = src[:i] + COLLECT_NEW + "".join(body) + "".join(lines[end:])
    open(full, "w").write(src)
    print("marked _run_engine per-step collection -> region collect_step")


def main():
    root = sys.argv[1]
    undo = "--undo" in sys.argv
    for path, cls, method, region, state in MARKS:
        full = os.path.join(root, path)
        orig = full + ".out.orig"
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
            print("marked %s.%s -> region %s" % (cls, method, region))
        else:
            print("already marked %s.%s" % (cls, method))
    if undo:
        orig = os.path.join(root, COLLECT_FILE) + ".out.orig"
        if os.path.exists(orig):
            shutil.move(orig, os.path.join(root, COLLECT_FILE))
            print("restored", COLLECT_FILE)
        return
    orig = os.path.join(root, COLLECT_FILE) + ".out.orig"
    if not os.path.exists(orig):
        shutil.copy(os.path.join(root, COLLECT_FILE), orig)
    patch_collect(root)


if __name__ == "__main__":
    main()

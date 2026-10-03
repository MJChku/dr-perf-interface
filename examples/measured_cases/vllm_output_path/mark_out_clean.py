"""Clean (low-overhead) output-path marking for the vLLM copy.

`mark_out.py` puts SEVEN regions on the output path, five of them nested inside
`process_outputs`.  A Python perfmark region costs instructions to construct and
that construction is charged to the PARENT region, so `process_outputs` in the
first run was measuring vLLM plus the markers of its own children.

This script leaves exactly two regions on the path:

  * process_outputs      (same two states as before, so the cost line is
                          directly comparable with out/vllm_out)
  * make_request_output  (ONE integer state, `kind`, so the FINAL_ONLY early
                          return can be told apart from the CUMULATIVE/DELTA
                          object build)

    python mark_out_clean.py /home/ubuntu/drperf-cases/vllm-cpu-out/vllm [--undo]
"""
import importlib.util
import os
import shutil
import sys

_spec = importlib.util.spec_from_file_location(
    "vllm_mark_ref", "/home/ubuntu/drperf/examples/vllm_cpu/mark.py")
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
wrap_method = _ref.wrap_method

# Every file mark_out.py touched: restore it from the .out.orig snapshot first.
TOUCHED = [
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/engine/output_processor.py",
    "vllm/v1/engine/detokenizer.py",
    "vllm/entrypoints/offline_utils.py",
]

MARKS = [
    ("vllm/v1/engine/output_processor.py", "OutputProcessor", "process_outputs",
     "process_outputs",
     "num_outputs=len(engine_core_outputs), num_active=len(self.request_states)"),
    ("vllm/v1/engine/output_processor.py", "RequestState", "make_request_output",
     "make_request_output", "kind=self.output_kind.value"),
]


def restore(root):
    for path in TOUCHED:
        full = os.path.join(root, path)
        orig = full + ".out.orig"
        if os.path.exists(orig):
            shutil.copy(orig, full)
            print("restored", path)


def main():
    root = sys.argv[1]
    restore(root)
    if "--undo" in sys.argv:
        return
    marks = MARKS
    if "--po-only" in sys.argv:
        # control configuration: process_outputs with NO nested marker at all,
        # so its per-output cost is free of perfmark region construction.
        marks = [m for m in MARKS if m[3] == "process_outputs"]
    for path, cls, method, region, state in marks:
        full = os.path.join(root, path)
        src = open(full).read()
        out, changed = wrap_method(src, cls, method, region, state)
        if not changed:
            sys.exit("failed to mark %s.%s" % (cls, method))
        open(full, "w").write(out)
        print("marked %s.%s -> region %s (%s)" % (cls, method, region, state))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU-only reproduction; temporary modules and raw captures are removed on exit."""
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[2]
CUDA = Path(os.environ.get("CUDA_HOME", "/home/ubuntu/GX/toolchains/cuda13"))
RESULTS = CASE / "results"
METRICS = ["global_read_bytes", "global_write_bytes", "global_unique_read_bytes",
           "global_unique_write_bytes", "local_read_bytes", "local_write_bytes",
           "shared_read_bytes", "shared_write_bytes", "fp32_flops", "special_function_ops"]


def main():
    RESULTS.mkdir(exist_ok=True)
    record = {"physical_gpu_used": False, "commands": [], "modules": {}, "runs": {}}
    started = time.monotonic()

    def run(argv, expected=0):
        argv = list(map(str, argv))
        begin = time.monotonic()
        p = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=600)
        record["commands"].append({"argv": argv, "seconds": time.monotonic()-begin,
                                   "exit_code": p.returncode, "output": p.stdout})
        if p.returncode != expected:
            raise RuntimeError(f"Unexpected exit {p.returncode}: {argv}\n{p.stdout}")
        return p.stdout

    # Verify the baseline device body is byte-for-byte the published device code.
    tree = ast.parse((CASE / "upstream_src.py").read_text())
    src = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "cuda_source" for t in n.targets))
    src = src[:src.index("torch::Tensor softmax_forward_cuda")].replace("#include <torch/extension.h>\n", "")
    assert src == (CASE / "before.cu").read_text()
    record["source_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in CASE.iterdir() if p.suffix in {".py", ".cc", ".cu"}}
    run([CUDA / "bin/nvcc", "--version"])
    with tempfile.TemporaryDirectory(prefix="drperf-softmax-") as tmp:
        work = Path(tmp)
        driver = work / "driver"
        run(["g++", "-std=c++17", "-O2", "-I"+str(ROOT/"gpu/include"), CASE/"driver.cc",
             "-L"+str(ROOT/"gpu/build"), "-ldrperf_gpu",
             "-Wl,-rpath,"+str(ROOT/"gpu/build"), "-o", driver])
        for variant in ["before", "synchronized", "after"]:
            print(f"Compile/translate {variant}", flush=True)
            ptx, module = work/f"{variant}.ptx", work/f"{variant}.so"
            run([CUDA/"bin/nvcc", "-ptx", "-arch=sm_80", "-lineinfo", CASE/f"{variant}.cu", "-o", ptx])
            assembly = run([CUDA/"bin/ptxas", "-arch=sm_80", "-v", ptx, "-o", work/f"{variant}.cubin"])
            run([ROOT/"bin/drperf", "--gpu", "translate", ptx, "-o", module])
            meta = json.loads((work/f"{variant}.so.gpu.json").read_text())
            assert not meta["instructionCoverage"]["unsupportedSites"]
            record["modules"][variant] = {"ptxas": assembly, "metadata": meta}
            if variant == "before":
                for suite in ["train", "original-width"]:
                    output = run([driver, module, variant, suite], expected=1)
                    assert "softmax numerical oracle failed" in output
                    record["runs"][f"{variant}-{suite}"] = {"expected_failure": True, "output": output}
                continue
            for suite in ["train", "heldout"]:
                print(f"Emulate/check {variant} {suite}", flush=True)
                report = work/f"{variant}-{suite}"
                output = run([ROOT/"bin/drperf", "--gpu", "--output", report,
                              "--", driver, module, variant, suite])
                model = json.loads((report/"profile.gpu.drperf.json").read_text())
                r = model["regions"][0]
                assert model["validity"] == "complete"
                # Resource assertions are independent of fitted coefficients.
                # Cached elements = rows * min(columns, 32768).
                for obs in r["observations"]:
                    n, rows = obs["pcvs"]["elements"], obs["pcvs"]["rows"]
                    cached = rows * min(n//rows, 32768) if variant == "after" else 0
                    m = obs["metrics"]
                    assert m["global_read_bytes"] == 12*n - 8*cached, (variant, suite, obs["pcvs"], m)
                    assert m["global_write_bytes"] == 8*n - 4*cached
                    assert m.get("local_read_bytes", 0) == m.get("local_write_bytes", 0) == 0
                    assert m["fp32_flops"] == 5*n + 2047*rows
                    assert m["special_function_ops"] == n
                checks = json.loads(output.splitlines()[0])
                compact = {"numerical_checks": checks, "state_count": r["stateCount"],
                    "pcvs": r["pcvs"], "fits": {m: r["fits"].get(m) for m in METRICS},
                    "interfaces": {m: r["interfaces"].get(m) for m in METRICS},
                    "states": [{"pcvs": s["pcvs"], "invocations": s["invocations"],
                                "metrics": {m: s["metrics"].get(m) for m in METRICS}}
                               for s in r["stateStatistics"]],
                    "global_memory_sites": {s: {m: f[m] for m in ["global_read_bytes", "global_write_bytes"] if m in f}
                        for s, f in r["sites"].items() if any(m in f for m in ["global_read_bytes", "global_write_bytes"])}}
                (RESULTS/f"{variant}-{suite}.json").write_text(json.dumps(compact, indent=2)+"\n")
                record["runs"][f"{variant}-{suite}"] = {"numerical_checks": checks,
                    "resource_assertions": "passed", "report": f"{variant}-{suite}.json"}
                # Preserve DrPerf's unmodified training report; large heldout
                # per-site fits are regenerable with the command above.
                if suite == "train":
                    (RESULTS/f"{variant}-report.txt").write_text((report/"report.txt").read_text())
        record["elapsed_seconds"] = time.monotonic()-started
    (RESULTS/"verification.json").write_text(json.dumps(record, indent=2)+"\n")
    for a,b in [("before", "synchronized"), ("synchronized", "after")]:
        patch = "".join(difflib.unified_diff((CASE/f"{a}.cu").read_text().splitlines(True),
                        (CASE/f"{b}.cu").read_text().splitlines(True), fromfile=f"{a}.cu", tofile=f"{b}.cu"))
        (RESULTS/f"{a}-to-{b}.patch").write_text(patch)
    print(f"PASS: CPU-only evidence in {RESULTS}")


if __name__ == "__main__":
    main()

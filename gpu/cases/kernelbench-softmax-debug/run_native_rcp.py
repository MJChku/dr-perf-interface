#!/usr/bin/env python3
"""One short RCP GPU allocation; collect receipts and delete the owned job."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

CASE = Path(__file__).resolve().parent
NS = "runai-vca-jma"
IMAGE = "vllm/vllm-openai@sha256:0674a3ea3c971706cea310bca71daadff68ea1717c758d2f54affa5def001b86"


def main():
    job = "drperf-softmax-" + datetime.datetime.now(datetime.timezone.utc).strftime("%m%d-%H%M%S")
    output = CASE / "results" / job
    output.mkdir()
    record = {"job": job, "namespace": NS, "image": IMAGE, "pool": "b300", "gpu_count": 1,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "commands": []}

    def run(cmd, timeout=60, check=True):
        start = time.monotonic()
        p = subprocess.run(list(map(str, cmd)), text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        saved_output = p.stdout
        if list(map(str, cmd))[:3] == ["kubectl", "get", "pods"] and p.returncode == 0:
            # Keep only this allocation's status, not unrelated jobs' pod specs.
            saved_output = json.dumps([{"name": x["metadata"]["name"],
                "phase": x["status"]["phase"], "node": x["spec"].get("nodeName")}
                for x in json.loads(p.stdout)["items"] if x["metadata"]["name"].startswith(job+"-")])
        record["commands"].append({"argv": list(map(str, cmd)), "seconds": time.monotonic()-start,
                                   "exit_code": p.returncode, "output": saved_output})
        if check and p.returncode:
            raise RuntimeError(p.stdout)
        return p.stdout

    def pods():
        return [p for p in json.loads(run(["kubectl", "get", "pods", "-n", NS, "-o", "json"]))["items"]
                if p["metadata"]["name"].startswith(job+"-")]

    try:
        with tempfile.TemporaryDirectory(prefix="drperf-native-bundle-") as tmp:
            bundle=Path(tmp)/"bundle"; bundle.mkdir()
            record["sources"]={}
            for name in ["synchronized.cu", "after.cu", "native_benchmark.cu"]:
                data=(CASE/name).read_bytes(); (bundle/name).write_bytes(data)
                record["sources"][name]=hashlib.sha256(data).hexdigest()
            (bundle/"run.sh").write_text('''#!/bin/bash
set -euo pipefail
cd /tmp/drperf-softmax-native
mkdir results
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.sm,clocks.mem,power.limit --format=csv > results/gpu-before.txt
nvcc --version > results/compiler.txt
nvcc -std=c++17 -O3 -lineinfo -arch=native -Xcompiler=-fopenmp -Xptxas=-v native_benchmark.cu -o native_benchmark > results/compile.log 2>&1
OMP_NUM_THREADS=8 timeout 300 ./native_benchmark > results/native.json 2> results/progress.log
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > results/gpu-after.txt
''')
            print(f"Submitting one B300: {job}", flush=True)
            run(["runai", "submit", job, "-p", "vca-jma", "--image", IMAGE,
                 "--gpu", "1", "--cpu", "8", "--cpu-limit", "8", "--memory", "16Gi", "--memory-limit", "16Gi",
                 "--node-pools", "b300", "--interactive", "--preemptible", "--run-as-uid", "228407",
                 "--run-as-gid", "83691", "--backoff-limit", "0", "--auto-deletion-time-after-completion", "1m",
                 "--command", "--", "/usr/bin/timeout", "--signal=TERM", "--kill-after=10", "600", "sleep", "590"])
            start=time.monotonic(); pod=None
            while time.monotonic()-start<300:
                selected=pods()
                if selected:
                    p=selected[0]
                    if p["status"]["phase"] in ["Failed", "Succeeded"]:
                        raise RuntimeError(f"Pod ended early: {p['status']}")
                    if p["status"]["phase"]=="Running" and all(c.get("ready") for c in p["status"].get("containerStatuses", [])):
                        pod=p["metadata"]["name"];record["node"]=p["spec"].get("nodeName");break
                time.sleep(5)
            if pod is None: raise RuntimeError("No ready GPU within five minutes")
            record["pod"]=pod
            print(f"Running on {record['node']}: {pod}", flush=True)
            run(["kubectl", "cp", "-n", NS, bundle, pod+":/tmp/drperf-softmax-native"])
            try:
                run(["kubectl", "exec", "-n", NS, pod, "--", "bash", "/tmp/drperf-softmax-native/run.sh"], timeout=420)
            finally:
                run(["kubectl", "cp", "-n", NS, pod+":/tmp/drperf-softmax-native/results", output/"native"], check=False)
            results=json.loads((output/"native/native.json").read_text())
            assert len(results["cases"])==4 and all(c["passed"] for c in results["cases"])
            record["result"]="passed"
            print(json.dumps(results), flush=True)
    finally:
        print(f"Deleting owned GPU job {job}", flush=True)
        run(["runai", "delete", "job", job, "-p", "vca-jma"], check=False)
        for _ in range(12):
            if not pods(): break
            time.sleep(5)
        record["job_removed"]=not bool(pods())
        record["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (output/"allocation.json").write_text(json.dumps(record,indent=2)+"\n")
        if not record["job_removed"]: raise RuntimeError("Owned pod remains; inspect cleanup")
    print(f"Evidence: {output}", flush=True)


if __name__=="__main__": main()

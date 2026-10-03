"""Build an isolated drperf bundle against GX's patched DynamoRIO runtime.

The repository's normal DynamoRIO installation is left untouched. This is for
workloads whose pre-existing workers block DynamoRIO's suspend signal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dynamorio", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    dr = args.dynamorio.resolve()
    runtime = dr / "lib64/release/libdynamorio.so"
    if b"attach_unmask_suspend_signal" not in runtime.read_bytes():
        parser.error("DynamoRIO lacks GX's blocked-suspend-signal support")
    if output.exists():
        parser.error("output directory must be new")

    subprocess.run([
        sys.executable, str(root / "examples/wan_gx/prepare_drperf_bundle.py"),
        str(output),
    ], check=True)
    for directory in ("lib64/release", "ext/lib64/release"):
        for source in (dr / directory).glob("*.so"):
            shutil.copyfile(source, output / "third_party/dynamorio" / directory / source.name)

    client = output / "client"
    client.mkdir()
    shutil.copyfile(root / "client/CMakeLists.txt", client / "CMakeLists.txt")
    source = (root / "client/drperf.c").read_text()
    # New drmgr versions own these event registrations. Callback bodies and
    # instruction-counting logic are unchanged.
    for old, new in (
        ("dr_register_exit_event(event_exit)", "drmgr_register_exit_event(event_exit)"),
        ("dr_register_filter_syscall_event(event_filter_syscall)",
         "drmgr_register_filter_syscall_event(event_filter_syscall)"),
    ):
        if source.count(old) != 1:
            raise RuntimeError(f"unexpected client source: {old}")
        source = source.replace(old, new)
    (client / "drperf.c").write_text(source)
    with tempfile.TemporaryDirectory(prefix="drperf-gx-build-") as build:
        subprocess.run([
            "cmake", "-S", str(client), "-B", build,
            "-DCMAKE_BUILD_TYPE=Release", f"-DDynamoRIO_DIR={dr / 'cmake'}",
            "-DCMAKE_C_FLAGS=-Werror=implicit-function-declaration",
        ], check=True)
        subprocess.run(["cmake", "--build", build, "-j2"], check=True)
    subprocess.run([
        "gcc", "-O2", "-shared", "-fPIC", "-o",
        str(output / "build/libdrperf_attach.so"), str(root / "perfmark/attach.c"),
        f"-L{dr / 'lib64/release'}", "-ldynamorio",
        "-Wl,-rpath,$ORIGIN/../third_party/dynamorio/lib64/release",
    ], check=True)

    runner = output / "lib/runner.py"
    source = runner.read_text()
    old = '"-code_api -client_lib'
    if source.count(old) != 1:
        raise RuntimeError("unexpected runner attachment options")
    runner.write_text(source.replace(
        old, '"-code_api -no_synchronous_attach -attach_unmask_suspend_signal -client_lib'
    ))
    manifest = {
        str(f.relative_to(output)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(output.rglob("*"))
        if f.is_file() and f.name != "manifest.json"
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()

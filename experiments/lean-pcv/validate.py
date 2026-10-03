"""Build all proofs, audit their axioms, and check expected rejection cases."""
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent


def run(*args):
    return subprocess.run(args, cwd=HERE, text=True, capture_output=True, timeout=120)


def main():
    build = run("lake", "build")
    print(build.stdout, end="")
    print(build.stderr, end="")
    assert build.returncode == 0, "Lean build failed"
    audit = run("lake", "env", "lean", "Audit.lean")
    assert audit.returncode == 0, audit.stdout + audit.stderr
    axioms = re.findall(r"depends on axioms: \[(.*?)\]", audit.stdout)
    assert len(axioms) == 14, audit.stdout
    for dependencies in axioms:
        assert set(dependencies.split(", ")) <= {"propext", "Classical.choice", "Quot.sound"}, dependencies
    print("PASS: all audited proofs use only Lean's standard logical axioms.")
    for name, expected in [("RejectRaw", "no affine fit on the discovery samples"),
                           ("RejectUnseen", "`grind` failed")]:
        result = run("lake", "env", "lean", f"tests/{name}.lean")
        assert result.returncode != 0, f"{name} was incorrectly accepted"
        assert expected in result.stdout + result.stderr, result.stdout + result.stderr
        print(f"PASS: {name} rejected at the expected stage.")
    demo = run("lake", "exe", "pcv-demo")
    assert demo.returncode == 0, demo.stdout + demo.stderr
    print(demo.stdout, end="")


if __name__ == "__main__":
    main()

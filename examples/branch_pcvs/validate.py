"""Build, run, and validate conditional PCVs using actual DynamoRIO counts."""
import csv
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lib"))
import derive  # noqa: E402
import runner  # noqa: E402


def main():
    binary = HERE / "bin" / "branch_pcvs"
    binary.parent.mkdir(exist_ok=True)
    subprocess.run([
        "gcc", "-O2", "-g", "-Wall", "-Wextra", "-Werror",
        str(HERE / "app.c"), "-o", str(binary),
        "-L" + str(ROOT / "build"), "-lperfmark",
        "-Wl,-rpath," + str(ROOT / "build"),
    ], check=True)
    subprocess.run([str(binary)], check=True)

    (ROOT / "out").mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="branch_pcvs-", dir=ROOT / "out"))
    rc, log, files = runner.run([str(binary)], str(output), timeout=60)
    print(log.strip())
    assert rc == 0 and files, "instrumented execution failed"
    runs = runner.load_runs(str(output))
    problems = runner.validity(runs)
    assert not problems, problems
    keys, slots = runner.blocks_of_set(runs)
    recs = runner.load_traces_all(runs)
    report = []
    models = {}
    with (output / "points.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "pcv_1", "pcv_2", "calls", "measured_mean",
                         "fitted", "unexplained", "residual"])
        for region in ("raw", "conditional"):
            vecs, calls, names, _, dropped = derive.inclusive_vectors(keys, region, recs)
            assert dropped == 0
            assert sum(calls.values()) == 60
            regimes = derive.derive(vecs, slots)
            assert len(regimes) == 1, "expected one joint two-variable fit"
            model = regimes[0]
            assert not model.dependent, "PCVs must vary independently"
            models[region] = model
            errors = []
            for v in model.values:
                measured = sum(vecs[v].values())
                residual = measured - model.total(v)
                errors.append(abs(residual))
                writer.writerow([region, *v, calls[v], measured, model.formula(v),
                                 model.irr.get(v, 0), residual])
            report.append(
                "%s: %d states; %.9g*%s + %.9g*%s + %.9g; "
                "unexplained %.6f%%; max residual %.6f instructions/call"
                % (region, len(model.values), model.a[0], names[0],
                   model.a[1], names[1], model.c, 100 * model.irr_share(), max(errors)))
    assert models["raw"].irr_share() > 0.5, "raw inputs should leave substantial cost unexplained"
    assert models["conditional"].n_irr == 0, "conditional PCVs should explain every block"
    assert all(a > 0 for a in models["conditional"].a)
    cli = runpy.run_path(str(ROOT / "bin" / "drperf"))
    report.extend(["", "CLI output from the same measurements:"])
    report.extend(cli["cost_lines"](runs, keys, slots, recs))
    report.append("\nPASS: raw inputs fail; conditional PCVs explain all blocks within tolerance.")
    text = "\n".join(report) + "\n"
    (output / "validation.txt").write_text(text)
    print(text)
    print("Measurements and per-state CSV:", output)


if __name__ == "__main__":
    main()

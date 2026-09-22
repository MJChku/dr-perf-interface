"""Late takeover must cover workers which existed with SIGILL blocked."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
import derive
import runner


class LateBlocked(unittest.TestCase):
    def test_options(self):
        self.assertEqual(runner.late_attach_options({}), ["-attach_unmask_suspend_signal"])
        self.assertEqual(runner.late_attach_options({"DRPERF_ATTACH_UNMASK_SIGNAL": "0"}), [])
        self.assertEqual(runner.late_attach_options({"DRPERF_ATTACH_UNMASK_SIGNAL": "1"}),
                         ["-attach_unmask_suspend_signal"])
        with self.assertRaises(ValueError):
            runner.late_attach_options({"DRPERF_ATTACH_UNMASK_SIGNAL": "yes"})

    def test_blocked_worker_coverage(self):
        probe = subprocess.run([runner.DRRUN, "-attach_unmask_suspend_signal", "--", "/bin/true"],
                               capture_output=True, text=True, timeout=15)
        if "unknown option -attach_unmask_suspend_signal" in probe.stderr:
            self.skipTest("runtime lacks strict signal-unmasking takeover support")
        self.assertEqual(probe.returncode, 0, probe.stderr)
        with tempfile.TemporaryDirectory(prefix="drperf-blocked-") as tmp:
            directory = Path(tmp)
            program = directory / "app"
            subprocess.run(["gcc", "-O2", "-pthread", str(ROOT / "tests/late_blocked.c"),
                            "-L" + str(ROOT / "build"), "-lperfmark",
                            "-Wl,-rpath," + str(ROOT / "build"), "-o", str(program)], check=True)
            results = []
            for mode in ("early", "blocked", "unmask"):
                out = directory / mode
                out.mkdir()
                env = runner.build_env()
                opts = ["-o", str(out / "run.%p.json"), "-blocks", "-no_follow_threads"]
                if mode == "early":
                    command = [runner.DRRUN, "-c", runner.CLIENT] + opts + ["--", str(program)]
                else:
                    env["DRPERF_LATE"] = "1"
                    env["DRPERF_ATTACH_UNMASK_SIGNAL"] = "1" if mode == "unmask" else "0"
                    core = " ".join(runner.late_attach_options(env))
                    env["DYNAMORIO_OPTIONS"] = (
                        f"-code_api -takeover_timeout_ms 100 {core} "
                        f"-client_lib '{runner.CLIENT};0;{' '.join(opts)}'")
                    env["LD_PRELOAD"] = runner.ATTACH + " " + env.get("LD_PRELOAD", "")
                    command = ["setarch", "-R", str(program)]
                run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
                if mode == "blocked":
                    self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
                    self.assertIn("Failed to take over", run.stderr)
                    continue
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                self.assertIn("BLOCKED_WORKERS_PASS", run.stdout)
                runs = runner.load_runs(str(out))
                self.assertEqual(runner.validity(runs), [])
                keys, _ = runner.blocks_of_set(runs)
                states, _, dropped = derive.per_state(keys, "worker")
                self.assertEqual(dropped, 0)
                vectors, calls = derive.per_trigger(states)
                self.assertEqual(len(vectors), 4)
                self.assertEqual(set(calls.values()), {2})
                results.append({v: sum(cs.values()) for v, cs in vectors.items()})
            self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()

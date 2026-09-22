"""End-to-end state-budget regressions; requires a built DynamoRIO client."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
import derive
import runner


class ManyStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "out").mkdir(exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(prefix="state-budget-", dir=ROOT / "out")
        cls.directory = Path(cls.tmp.name)
        cls.program = cls.directory / "cstates"
        subprocess.run(["gcc", "-O2", "-Wall", "-Wextra", "-Werror",
                        str(ROOT / "tests/cstates.c"), "-o", str(cls.program),
                        "-L" + str(ROOT / "build"), "-lperfmark",
                        "-Wl,-rpath," + str(ROOT / "build")], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_capture(self, out, points, regions, budget):
        runs = runner.load_runs(str(out))
        self.assertEqual(runner.validity(runs), [])
        self.assertTrue(runs["runs"])
        for run in runs["runs"]:
            self.assertEqual(run["data"]["drperf"]["max_states_per_region"], budget)
        keys, _ = runner.blocks_of_set(runs)
        recs = runner.load_traces_all(runs)
        kept = min(points, budget)
        for i in range(regions):
            vecs, calls, names, nested, dropped = derive.inclusive_vectors(
                keys, f"states_{i}", recs)
            self.assertEqual(tuple(names), ("n",))
            self.assertEqual(set(vecs), {(n,) for n in range(kept)})
            # Revisit retained states after reaching the cap: no loss/merging.
            self.assertEqual(set(calls.values()), {2})
            self.assertEqual((nested, dropped), (0, 2 * (points - kept)))
            if kept > 128:
                self.assertGreater(sum(vecs[(128,)].values()), sum(vecs[(127,)].values()))

    def capture(self, label, points, regions, setting, expected_budget):
        out = self.directory / label
        env = dict(os.environ)
        env.pop("DRPERF_MAX_STATES_PER_REGION", None)
        if setting is not None:
            env["DRPERF_MAX_STATES_PER_REGION"] = str(setting)
        with patch.dict(os.environ, env, clear=True):
            rc, log, files = runner.run([str(self.program), str(points), str(regions)],
                                        str(out), timeout=120)
        self.assertEqual(rc, 0, log)
        self.assertIn("STATE_CARDINALITY_PASS", log)
        self.assertTrue(files, log)
        self.check_capture(out, points, regions, expected_budget)

    def test_default_and_overflow(self):
        self.capture("default", 4098, 1, None, 4096)

    def test_raise_budget(self):
        self.capture("raised", 5000, 1, 8192, 8192)

    def test_budget_applies_after_64_regions(self):
        self.capture("many-regions", 4, 70, 2, 2)

    def test_native_client_option(self):
        out = self.directory / "direct"
        out.mkdir()
        result = subprocess.run([runner.DRRUN, "-c", runner.CLIENT, "-o", str(out / "run.json"),
                                 "-blocks", "-max_states_per_region", "300",
                                 "--", str(self.program), "301", "1"],
                                env=runner.build_env(), capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.check_capture(out, 301, 1, 300)

    def test_invalid_environment(self):
        for value in ["", "0", "-1", "65536", "100000000000000000000", "1.5", "8 -trace 0"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.state_budget_options({"DRPERF_MAX_STATES_PER_REGION": value})

    def test_invalid_client_option(self):
        result = subprocess.run([runner.DRRUN, "-c", runner.CLIENT,
                                 "-max_states_per_region", "65536", "--", str(self.program), "1"],
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be an integer between 1 and 65535", result.stderr)


if __name__ == "__main__":
    unittest.main()

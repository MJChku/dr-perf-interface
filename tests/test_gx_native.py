"""Native emulation must execute its body and return to host instrumentation."""
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


class GXNative(unittest.TestCase):
    def test_configuration(self):
        self.assertEqual(runner.native_gx_options({}), [])
        self.assertEqual(runner.native_gx_options({"DRPERF_NATIVE_GX": "0"}), ["-no_auto_gx"])
        self.assertEqual(runner.native_exec_options({"DRPERF_NATIVE_EXEC_MODULES": "a.so,b.so"}),
                         ["-native_exec_list", "a.so;b.so"])
        with self.assertRaises(ValueError):
            runner.native_exec_options({"DRPERF_NATIVE_EXEC_MODULES": "gx_cuda.so"})
        self.assertEqual(runner.native_gx_options({"DRPERF_NATIVE_GX": "1",
                                                  "DRPERF_EXCLUDE_CUDA_MODULE": "gx_cuda.so"}),
                         ["-native_gx"])
        for env in ({"DRPERF_NATIVE_GX": "yes"}, {"DRPERF_NATIVE_GX": "1"},
                    {"DRPERF_NATIVE_GX": "1", "DRPERF_EXCLUDE_CUDA_MODULE": "gx_cuda.so",
                     "DRPERF_NATIVE_EXEC_MODULES": "gx_cuda.so"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                runner.native_gx_options(env)
        self.assertIn("requested GX native-work boundary was not found",
                      runner.validity({"runs": [{"data": {"drperf": {"native_gx": True}}}]}))

    def test_native_boundary_and_resume(self):
        (ROOT / "out").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="gx-native-", dir=ROOT / "out") as tmp:
            directory = Path(tmp)
            library, program = directory / "fake_gx.so", directory / "app"
            source = str(ROOT / "tests/gx_native.c")
            subprocess.run(["gcc", "-O2", "-fPIC", "-shared", "-DEMULATOR", source,
                            "-o", str(library)], check=True)
            subprocess.run(["gcc", "-O2", "-pthread", source, str(library),
                            "-L" + str(ROOT / "build"), "-lperfmark",
                            "-Wl,-rpath," + str(ROOT / "build"), "-o", str(program)], check=True)
            results = []
            for attach in ("early", "late"):
                for native in ("0", "1"):
                    with self.subTest(attach=attach, native=native):
                        out = directory / (attach + native)
                        out.mkdir()
                        with patch.dict(os.environ, {"DRPERF_NATIVE_GX": native,
                                                     "DRPERF_NATIVE_EXEC_MODULES": "",
                                                     "DRPERF_FOLLOW_THREADS": "0",
                                                     "DRPERF_EXCLUDE_CUDA_MODULE": library.name}):
                            if attach == "late":
                                rc, log, _ = runner.run([str(program)], str(out), timeout=30)
                            else:
                                command = [runner.DRRUN, "-c", runner.CLIENT,
                                           "-o", str(out / "run.json"), "-blocks",
                                           "-no_follow_threads", "-exclude_cuda_module", library.name]
                                command += runner.native_gx_options(os.environ)
                                run = subprocess.run(command + ["--", str(program)],
                                                     env=runner.build_env(), capture_output=True,
                                                     text=True, timeout=30)
                                rc, log = run.returncode, run.stdout + run.stderr
                        self.assertEqual(rc, 0, log)
                        self.assertIn("GX_NATIVE_PASS", log)
                        runs = runner.load_runs(str(out))
                        self.assertEqual(runner.validity(runs), [])
                        stats = runs["runs"][0]["data"]["drperf"]
                        self.assertEqual(stats["native_gx_hooks"], int(native))
                        self.assertEqual(stats["native_gx_calls"], 8 * int(native))
                        self.assertEqual(stats["excluded_cuda_calls"], 8)
                        keys, _ = runner.blocks_of_set(runs)
                        states, _, dropped = derive.per_state(keys, "after")
                        self.assertEqual(dropped, 0)
                        vecs, calls = derive.per_trigger(states)
                        self.assertEqual(set(calls.values()), {2})
                        self.assertEqual(len(vecs), 4)
                        results.append({v: sum(cs.values()) for v, cs in vecs.items()})
            self.assertTrue(all(r == results[0] for r in results))

    def test_detect_loaded_gx_without_options(self):
        """Detect the module even if it wasn't in the launcher's environment."""
        with tempfile.TemporaryDirectory(prefix="drperf-auto-gx-") as tmp:
            directory = Path(tmp)
            library, program = directory / "gx_cuda.so", directory / "app"
            source = str(ROOT / "tests/gx_native.c")
            subprocess.run(["gcc", "-O2", "-fPIC", "-shared", "-DEMULATOR", source,
                            "-o", str(library)], check=True)
            subprocess.run(["gcc", "-O2", "-pthread", source, str(library),
                            "-L" + str(ROOT / "build"), "-lperfmark",
                            "-Wl,-rpath," + str(ROOT / "build"), "-o", str(program)], check=True)
            results = []
            for mode in ("off", "early", "late"):
                out = directory / mode
                out.mkdir()
                with patch.dict(os.environ):
                    for name in ("DRPERF_NATIVE_GX", "DRPERF_EXCLUDE_CUDA_MODULE",
                                 "DRPERF_NATIVE_EXEC_MODULES"):
                        os.environ.pop(name, None)
                    os.environ["DRPERF_FOLLOW_THREADS"] = "0"
                    if mode == "late":
                        rc, log, _ = runner.run([str(program)], str(out), timeout=30)
                    else:
                        opts = ["-no_auto_gx"] if mode == "off" else []
                        run = subprocess.run([runner.DRRUN, "-c", runner.CLIENT,
                                              "-o", str(out / "run.json"), "-blocks",
                                              "-no_follow_threads"] + opts + ["--", str(program)],
                                             env=runner.build_env(), capture_output=True,
                                             text=True, timeout=30)
                        rc, log = run.returncode, run.stdout + run.stderr
                self.assertEqual(rc, 0, log)
                self.assertIn("GX_NATIVE_PASS", log)
                runs = runner.load_runs(str(out))
                self.assertEqual(runner.validity(runs), [])
                stats = runs["runs"][0]["data"]["drperf"]
                self.assertEqual(stats["native_gx_calls"], 0 if mode == "off" else 8)
                self.assertEqual(stats["excluded_cuda_calls"], 0 if mode == "off" else 8)
                keys, slots = runner.blocks_of_set(runs)
                gx_count = sum(n for k in keys.values() for b, n in k["vec"].items()
                               if slots[b][0] == "gx_cuda.so")
                if mode == "off":
                    self.assertGreater(gx_count, 0)
                else:
                    self.assertEqual(gx_count, 0)
                states, _, dropped = derive.per_state(keys, "after")
                self.assertEqual(dropped, 0)
                vectors, calls = derive.per_trigger(states)
                self.assertEqual(set(calls.values()), {2})
                self.assertEqual(len(vectors), 4)
                results.append({v: sum(cs.values()) for v, cs in vectors.items()})
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[1], results[2])


if __name__ == "__main__":
    unittest.main()

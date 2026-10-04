"""CPU-only tests for the benchmark runner, including a fake engine subprocess."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import bench_prefill_suite as suite


LOG = """strata generate: GPU 0: test GPU, compute capability 8.6
strata generate: expert cache 123 slots, 1 GiB of VRAM
strata generate: prompt path borrows 8 cache slots
strata prefill: GDN_HASH abcd ef01
strata generate: prefill 128 tokens in 2 chunks, 100.0 ms (1280 tok/s); experts streamed 40 (30 by DMA, host 2.0 ms), resident 10; PLE 3.0 ms
output  : 99
prefill                  128 tokens in 120.0 ms -> 1066 tok/s (time to first token 130.0 ms)
"""


class PlanTests(unittest.TestCase):
    def args(self, *extra):
        return suite.parser().parse_args(["--config", "unused.json", *extra])

    def test_presets_boundaries_and_fixed_context(self):
        smoke, _, _ = suite.build_plan(self.args("--preset", "smoke"))
        self.assertEqual(len(smoke), 6)
        standard, _, _ = suite.build_plan(self.args())
        self.assertEqual(len(standard), 63)
        self.assertEqual({c["context"] for c in standard}, {8256})
        full, _, _ = suite.build_plan(self.args("--preset", "full"))
        self.assertEqual(len(full), 720)
        self.assertTrue({1023, 1024, 1025, 1041}.issubset({c["tokens"] for c in full}))

    def test_matrix_and_skips(self):
        cases, skipped, _ = suite.build_plan(self.args(
            "--lengths", "17,65", "--chunks", "16,auto", "--memory", "arena,mmap-ring2",
            "--kv-modes", "fp16,int8", "--expert-caches", "auto,128", "--contexts", "32,128"))
        self.assertEqual(len(cases), 48)
        self.assertEqual(len(skipped), 16)
        self.assertEqual({c["reason"] for c in skipped}, {"context too small for prompt and output"})

    def test_seeded_prefixes_and_distinct_workloads(self):
        args = self.args("--lengths", "17,65", "--chunks", "16,32", "--workloads", "ramp,random,repeat")
        cases, _, prompts = suite.build_plan(args)
        self.assertEqual(suite.build_plan(args), (cases, [], prompts))
        for name in ("ramp", "random", "repeat"):
            matching = [c for c in cases if c["workload"] == name]
            short = next(c for c in matching if c["tokens"] == 17)
            long = next(c for c in matching if c["tokens"] == 65)
            self.assertEqual(prompts[short["prompt_sha256"]].strip().split(","),
                             prompts[long["prompt_sha256"]].strip().split(",")[:18])
            self.assertEqual(len({c["prompt_sha256"] for c in matching}), 2)

    def test_real_prompt_is_never_repeated_to_fill(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "prompt.tokens"
            path.write_text("1, 2\n3 4 5")
            cases, skipped, prompts = suite.build_plan(self.args(
                "--tokens-file", str(path), "--lengths", "4,8", "--chunks", "256"))
            self.assertEqual(len(cases), 1)
            self.assertEqual(skipped[0]["reason"], "prompt file too short")
            self.assertEqual(next(iter(prompts.values())), "1,2,3,4,5\n")

    def test_command_overrides_and_config_memory_preservation(self):
        base = ["--pack", "weights", "--tokens", "1,2", "--prefill", "auto", "--seed", "9",
                "--max-context", "99999", "--kv", "int8", "--expert-cache", "auto",
                "--resident-experts", "--shared-expert-arena", "arena.bin", "--logits-stride", "2"]
        case = dict(chunk="256", context=2048, memory="mmap", kv="fp16", expert_cache="128")
        command = suite.make_command(base, "engine", case, "prompt.tokens", 0)
        self.assertEqual(command.count("--prefill"), 1)
        self.assertNotIn("--seed", command)
        self.assertNotIn("--tokens", command)
        self.assertNotIn("--resident-experts", command)
        self.assertNotIn("--shared-expert-arena", command)
        self.assertNotIn("--logits-stride", command)
        self.assertIn("--mmap-experts", command)
        self.assertEqual(command[command.index("--kv") + 1], "fp16")
        command = suite.make_command(base, "engine", dict(case, memory="config"), "prompt.tokens", 0)
        self.assertIn("--resident-experts", command)


class ResultTests(unittest.TestCase):
    def test_parse_and_reject_partial_invalid_or_duplicate_results(self):
        parsed = suite.parse_log(LOG, 128)
        self.assertEqual(parsed["ms"], 100)
        self.assertEqual(parsed["engine_ttft_ms"], 130)
        self.assertEqual((parsed["cache_slots"], parsed["borrowed_slots"]), (123, 8))
        for broken in (LOG.replace("100.0 ms", "0.0 ms"), LOG.replace("100.0 ms", "1e309 ms"),
                       LOG.replace("output  : 99", "output  :"), LOG + LOG, "engine ran out of memory"):
            with self.assertRaises(ValueError):
                suite.parse_log(broken, 128)
        with self.assertRaises(ValueError):
            suite.parse_log(LOG, 129)

    def test_pair_validation_catches_correctness_and_placement_drift(self):
        record = suite.parse_log(LOG, 128)
        suite.validate_pair(record, dict(record))
        for field in ("chunks", "gpu", "gdn_hashes", "output", "streamed", "dma", "resident", "cache_slots"):
            with self.assertRaisesRegex(ValueError, field):
                suite.validate_pair(record, dict(record, **{field: "different"}))

    def test_statistics_and_deterministic_bootstrap(self):
        old = suite.parse_log(LOG, 128)
        pairs = [{"A": old, "B": dict(old, ms=80)} for _ in range(3)]
        summary = suite.summarize(pairs, 9)
        self.assertAlmostEqual(summary["throughput_gain_pct"], 25)
        self.assertAlmostEqual(summary["time_saved_pct"], 20)
        self.assertEqual(summary["new_tps"], 1600)
        self.assertEqual(summary["paired_gain_ci95_low"], 25)
        self.assertEqual(summary, suite.summarize(pairs, 9))
        self.assertIsNone(suite.summarize(pairs[:2], 9)["paired_gain_ci95_low"])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.counter = self.root / "counter"
        self.fake = self.root / "fake_engine.py"
        self.fake.write_text(textwrap.dedent('''
            import argparse, math, os
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument('--counter', type=Path)
            p.add_argument('--tokens-file', type=Path)
            p.add_argument('--prefill')
            p.add_argument('--fail-b', action='store_true')
            a, rest = p.parse_known_args()
            i = int(a.counter.read_text()) + 1 if a.counter.exists() else 1
            a.counter.write_text(str(i))
            enabled = os.environ['STRATA_PREFILL_STREAM_AHEAD'] == '1'
            if a.fail_b and enabled:
                print('simulated GPU OOM', flush=True)
                raise SystemExit(7)
            n = len(a.tokens_file.read_text().strip().split(',')) - 1
            chunks = 1 if a.prefill == 'auto' else math.ceil(n / int(a.prefill))
            base = 1000 if i <= 2 else 100
            dump = os.environ.get('STRATA_PREFILL_DUMP_R')
            if dump:
                base = 10000
                Path(dump).write_bytes(b'checked residual bytes')
            ms = base * (0.8 if enabled else 1)
            print('strata generate: GPU 0: fake GPU')
            print('strata generate: expert cache 100 slots, 1 GiB')
            print('strata generate: prompt path borrows 8 cache slots')
            print('strata prefill: GDN_HASH abcd ef01')
            print(f'strata generate: prefill {n} tokens in {chunks} chunks, {ms} ms (1 tok/s); '
                  'experts streamed 40 (30 by DMA, host 2.0 ms), resident 10; PLE 3.0 ms')
            print('output  : 99')
        '''), encoding="utf-8")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"exe": sys.executable, "cwd": str(self.root),
                                          "args": [str(self.fake), "--counter", str(self.counter)]}))
        self.output = self.root / "results"
        self.argv = ["--config", str(self.config), "--lengths", "65", "--chunks", "256", "--rounds", "3",
                     "--warmup-pairs", "1", "--check-residuals", "--output", str(self.output)]

    def run_suite(self, extra=()):
        with mock.patch.object(suite, "probe", return_value="test metadata"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return suite.main([*self.argv, *extra])

    def test_end_to_end_excludes_warmup_and_checks_then_resumes(self):
        self.assertEqual(self.run_suite(), 0)
        self.assertEqual(self.counter.read_text(), "10")  # warmup 2, check 2, measured 6
        summary = json.loads((self.output / "summary.json").read_text())["cases"][0]
        self.assertEqual((summary["old_ms"], summary["new_ms"]), (100, 80))
        self.assertEqual(summary["pairs"], 3)
        self.assertTrue((self.output / "summary.csv").exists())
        self.assertIn("+25.00%", (self.output / "summary.md").read_text())
        self.assertEqual(self.run_suite(["--resume"]), 0)
        self.assertEqual(self.counter.read_text(), "10")
        # Half an interrupted pair is not mixed with a freshly executed arm.
        next((self.output / "cases").glob("*/measure-2-B.json")).unlink()
        self.assertEqual(self.run_suite(["--resume"]), 0)
        self.assertEqual(self.counter.read_text(), "12")

    def test_failure_preserves_log_and_excludes_case_from_performance(self):
        config = json.loads(self.config.read_text())
        config["args"].append("--fail-b")
        self.config.write_text(json.dumps(config))
        self.assertEqual(self.run_suite(), 1)
        row = json.loads((self.output / "summary.json").read_text())["cases"][0]
        self.assertEqual(row["status"], "failed")
        self.assertNotIn("old_ms", row)
        failed = next((self.output / "cases").glob("*/warmup-0-B.log"))
        self.assertIn("simulated GPU OOM", failed.read_text())

    def test_dry_run_never_launches_or_creates_results(self):
        self.assertEqual(self.run_suite(["--dry-run"]), 0)
        self.assertFalse(self.counter.exists())
        self.assertFalse(self.output.exists())

    def test_resume_rejects_changed_settings_and_corrupted_logs(self):
        self.assertEqual(self.run_suite(), 0)
        with self.assertRaises(SystemExit) as exc:
            self.run_suite(["--resume", "--rounds", "4"])
        self.assertEqual(exc.exception.code, 2)
        log = next((self.output / "cases").glob("*/measure-0-A.log"))
        log.write_text(log.read_text() + "modified\n")
        self.assertEqual(self.run_suite(["--resume"]), 1)
        row = json.loads((self.output / "summary.json").read_text())["cases"][0]
        self.assertIn("saved log changed", row["error"])

    def test_timeout_leaves_failed_record(self):
        record = suite.run_engine([sys.executable, "-c", "import time; time.sleep(5)"], self.root,
                                  dict(os.environ, STRATA_PREFILL_STREAM_AHEAD="0"), self.root / "timeout",
                                  65, .05, False, False)
        self.assertEqual(record["status"], "failed")
        self.assertIn("timed out", record["error"])
        self.assertTrue((self.root / "timeout.json").exists())


if __name__ == "__main__":
    unittest.main()

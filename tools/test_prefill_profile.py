"""CPU-only tests: python tools/test_prefill_profile.py."""
import json
from pathlib import Path
import tempfile
import unittest

from prefill_profile import compare, load, signature, summarize


class ProfileTest(unittest.TestCase):
    def records(self, run="a", complete=True):
        base = {"schema": 1, "run": run}
        return [dict(base, type="run", gpu="test GPU", device=0, position=0, tokens=512,
                     layer_begin=0, layer_end=2),
                dict(base, type="gpu", device=0, stream="compute", phase="gemm", chunk=0,
                     tokens=512, layer=1, implementation="MMQ", gu_type=1, down_type=2,
                     ms=6.0, intervals=3, max_ms=4.0),
                dict(base, type="gpu", device=0, stream="compute", phase="gemm", chunk=0,
                     tokens=512, layer=1, implementation="MMQ", gu_type=1, down_type=2,
                     ms=2.0, intervals=1, max_ms=2.0),
                dict(base, type="host", phase="grouping", ms=3),
                dict(base, type="end", complete=complete, valid=True, wall_ms=10)]

    def load_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            return load(path)

    def test_aggregation_across_folds(self):
        runs, skipped = self.load_rows(self.records())
        result = summarize(runs, "implementation")
        row, = result["rows"]
        self.assertEqual((row["ms"], row["intervals"], row["max_ms"], row["mean_ms"]), (8, 4, 4, 2))
        self.assertEqual(row["percent"], 100)
        self.assertEqual(result["host_ms"], {"grouping": 3})
        self.assertEqual(skipped, 0)

    def test_excludes_failed_cancelled_and_unfinished(self):
        invalid = self.records("invalid")
        invalid[-1]["valid"] = False
        runs, skipped = self.load_rows(self.records() + self.records("cancelled", False) +
                                       self.records("unfinished")[:-1] + invalid)
        self.assertEqual((len(runs), skipped), (1, 3))
        with self.assertRaisesRegex(ValueError, "no complete"):
            self.load_rows(invalid)

    def test_comparison_and_workload_matching(self):
        before, _ = self.load_rows(self.records())
        after_rows = self.records("b")
        after_rows[1]["ms"] = 3
        after, _ = self.load_rows(after_rows)
        self.assertEqual(signature(before), signature(after))
        delta, = compare(summarize(after), summarize(before))
        self.assertEqual(delta["delta_ms"], -3)
        after[0]["metadata"]["tokens"] = 1024
        self.assertNotEqual(signature(before), signature(after))

    def test_streams_remain_separate(self):
        rows = self.records()
        rows.insert(3, dict(rows[1], device=1, stream="peer compute", ms=5))
        runs, _ = self.load_rows(rows)
        result = summarize(runs)
        self.assertEqual([r["percent"] for r in result["rows"]], [100, 100])
        self.assertEqual(result["stage_wall_ms"], 10)

    def test_malformed_input_has_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_text('{"schema":1', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r":1:"):
                load(path)


if __name__ == "__main__":
    unittest.main()

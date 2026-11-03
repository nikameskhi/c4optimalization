"""Tests for repeated experiments and generated result files."""

import csv
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from run_experiments import STRATEGIES, run_experiments


class ExperimentRunnerTests(unittest.TestCase):
    """Check that the runner produces parseable data and charts."""

    def test_runs_all_strategies_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            rows, summaries = run_experiments(
                strategies=STRATEGIES,
                runs=1,
                games_per_side=1,
                seed=31,
                output_dir=output_dir,
                depth=1,
                simulations=1,
                iterations=2,
            )

            self.assertEqual(len(rows), len(STRATEGIES))
            self.assertEqual(set(summaries), set(STRATEGIES))

            with (output_dir / "strategy_results.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                csv_rows = list(csv.DictReader(file))
            self.assertEqual(len(csv_rows), len(STRATEGIES))

            report = json.loads(
                (output_dir / "strategy_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["runs_per_strategy"], 1)
            self.assertEqual(len(report["runs"]), len(STRATEGIES))

            with (output_dir / "strategy_summary.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                summary_rows = list(csv.DictReader(file))
            self.assertEqual(len(summary_rows), len(STRATEGIES))

            chart = ET.parse(output_dir / "strategy_scores.svg").getroot()
            self.assertEqual(chart.tag, "{http://www.w3.org/2000/svg}svg")
            runtime_chart = ET.parse(output_dir / "strategy_runtime.svg").getroot()
            self.assertEqual(runtime_chart.tag, "{http://www.w3.org/2000/svg}svg")


if __name__ == "__main__":
    unittest.main()

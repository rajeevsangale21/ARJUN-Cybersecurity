import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from pipeline import run_pipeline
from generate_sample_data import generate_sample_csv, generate_sample_zeek


class TestPipelineIngestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("data/sample")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.csv_file = cls.test_dir / "sample_network_flows.csv"
        cls.zeek_file = cls.test_dir / "sample_zeek_conn.log"

        if not cls.csv_file.exists():
            generate_sample_csv(cls.csv_file, num_windows=12)
        if not cls.zeek_file.exists():
            generate_sample_zeek(cls.zeek_file, num_windows=12)

    def test_csv_pipeline_execution(self):
        result = run_pipeline(
            str(self.csv_file),
            window_seconds=5,
            sequence_length=5,
        )
        self.assertIn("states", result)
        self.assertIn("graph_sequences", result)
        self.assertIn("feature_names", result)

        states = np.asarray(result["states"])
        self.assertGreater(len(states), 5)
        self.assertEqual(states.shape[1], 33)

    def test_zeek_pipeline_execution(self):
        result = run_pipeline(
            str(self.zeek_file),
            window_seconds=5,
            sequence_length=5,
        )
        self.assertIn("states", result)
        self.assertIn("graph_sequences", result)

        states = np.asarray(result["states"])
        self.assertGreater(len(states), 5)
        self.assertEqual(states.shape[1], 33)


if __name__ == "__main__":
    unittest.main()

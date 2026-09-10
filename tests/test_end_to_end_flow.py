import unittest
from pathlib import Path
from run_end_to_end import run_end_to_end
from database.repository import get_recent_analyses, get_recent_risks


class TestEndToEndFlow(unittest.TestCase):
    def test_end_to_end_csv_pipeline(self):
        sample_csv = Path("data/sample/sample_network_flows.csv")
        self.assertTrue(sample_csv.exists(), "Sample CSV file must exist")

        result = run_end_to_end(
            input_path=str(sample_csv),
            model_path="saved_models/hybrid_world_model.pt",
            steps=5,
            window_seconds=5,
            sequence_length=5,
            save_db=True,
        )

        self.assertIsNotNone(result)
        self.assertIn("analysis_id", result)
        self.assertIn("forecast_id", result)
        self.assertIn("summary", result)
        self.assertIn("timeline", result)

        # Verify timeline structure
        timeline = result["timeline"]
        self.assertEqual(len(timeline), 5)
        for item in timeline:
            self.assertIn("step", item)
            self.assertIn("risk_score", item)
            self.assertIn("risk_level", item)
            self.assertIn("mitre_stage", item)

        # Verify database record creation
        recent_analyses = get_recent_analyses(limit=1)
        self.assertGreaterEqual(len(recent_analyses), 1)
        self.assertEqual(recent_analyses[0]["id"], result["analysis_id"])


if __name__ == "__main__":
    unittest.main()

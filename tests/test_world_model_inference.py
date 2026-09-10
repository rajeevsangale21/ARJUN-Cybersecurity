import unittest
from pathlib import Path
import numpy as np
import torch

from world_model.hybrid_wrapper import HybridWorldModelWrapper
from forecasting.hybrid_k_step_forecaster import HybridKStepForecaster


class TestWorldModelInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = Path("saved_models/hybrid_world_model.pt")
        if not cls.model_path.exists():
            raise FileNotFoundError(f"Model not found: {cls.model_path}")

        cls.wrapper = HybridWorldModelWrapper.from_checkpoint(
            cls.model_path,
            state_dimension=33,
            device="cpu",
        )
        cls.forecaster = HybridKStepForecaster(cls.wrapper, device="cpu")

    def test_model_loaded(self):
        self.assertIsNotNone(self.wrapper.model)
        self.assertEqual(self.wrapper.state_dimension, 33)

    def test_predict_single_step(self):
        # Create dummy sequence of 5 states
        seq = np.random.randn(5, 33).astype(np.float32)
        pred = self.wrapper.predict(seq, graph_sequence=None)
        self.assertEqual(pred.shape, (33,))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_k_step_forecast(self):
        seq = np.random.randn(5, 33).astype(np.float32)
        forecast = self.forecaster.forecast(seq, graph_sequence=None, steps=5)
        self.assertEqual(forecast.shape, (5, 33))
        self.assertTrue(np.all(np.isfinite(forecast)))


if __name__ == "__main__":
    unittest.main()

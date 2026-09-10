import unittest
from fastapi.testclient import TestClient
from backend.app import app


class TestAPIService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("application"), "ARJUN")
        self.assertEqual(data.get("status"), "running")

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("database", data)

    def test_records_stats(self):
        response = self.client.get("/api/records/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_analyses", data)
        self.assertIn("total_forecasts", data)
        self.assertIn("total_risks", data)

    def test_records_analyses(self):
        response = self.client.get("/api/records/analyses?limit=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_records_risks(self):
        response = self.client.get("/api/records/risks?limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_alerts_endpoint(self):
        response = self.client.get("/api/alerts?limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)


if __name__ == "__main__":
    unittest.main()

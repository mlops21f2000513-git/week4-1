import unittest
import subprocess
import pandas as pd
from sklearn import metrics
import os
import json

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        This runs once before all tests.
        It ensures DVC data is pulled from Google Cloud Storage (GCS).
        Works in GitHub Actions with authentication via GCP service account.
        """
        print("\nSetting up DVC + GCS for data pull...")

        # Expect GOOGLE_APPLICATION_CREDENTIALS JSON content to be in env variable
        gcp_key = os.getenv("GCP_SA_KEY")
        if not gcp_key:
            raise EnvironmentError(
                "Missing GCP_SA_KEY environment variable (add it to GitHub secrets)."
            )

        # Write the service account key to a temporary file
        key_path = "gcp-key.json"
        with open(key_path, "w") as f:
            f.write(gcp_key)

        # Authenticate gcloud CLI (optional but safe)
        subprocess.run(
            ["gcloud", "auth", "activate-service-account", "--key-file", key_path],
            check=True,
        )

        # Tell DVC and GCS which credentials to use
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

        # Verify the account
        subprocess.run(["gcloud", "auth", "list"], check=True)
        
        # Pull data using DVC
        print("Running 'dvc pull' ...")
        result = subprocess.run(["dvc", "pull"], capture_output=True, text=True)
        if result.returncode != 0:
            print("DVC pull failed!")
            print(result.stderr)
            raise RuntimeError(f"DVC pull failed:\n{result.stderr}")
        print("DVC data successfully pulled.")


    def test_data_columns_present(self):
        """
        Validate that iris.csv has all required columns.
        """
        expected_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

        file_path = "data/iris.csv"
        self.assertTrue(os.path.exists(file_path), f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing expected column: {col}")

        print("✅ All expected columns are present in iris.csv.")

    # def test_model_evaluation(self):
    #     """
    #     Test that model evaluation (accuracy) works correctly.
    #     """
    #     y_true = ["setosa", "versicolor", "virginica"]
    #     y_pred = ["setosa", "versicolor", "virginica"]
    #     acc = metrics.accuracy_score(y_true, y_pred)

    #     self.assertEqual(acc, 1.0, "Model accuracy should be 100% for identical predictions.")
    #     print("✅ Model evaluation passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

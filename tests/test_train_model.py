import unittest

import pandas as pd

from src.train_model import DATA, build_preprocessor, load_data


class TrainModelTests(unittest.TestCase):
    def test_dataset_path_exists(self):
        self.assertTrue(DATA.exists(), f"Dataset não encontrado em {DATA}")

    def test_load_data_removes_invalid_extremes(self):
        data = load_data()
        self.assertTrue((data["person_age"] <= 100).all())
        self.assertTrue((data["person_emp_length"].fillna(0) <= 80).all())
        self.assertIn("loan_status", data.columns)

    def test_preprocessor_handles_numeric_and_categorical_columns(self):
        data = pd.DataFrame({"income": [1000, None], "home": ["RENT", None]})
        transformed = build_preprocessor(data).fit_transform(data)
        self.assertEqual(transformed.shape[0], 2)
        self.assertGreater(transformed.shape[1], 0)


if __name__ == "__main__":
    unittest.main()

import unittest
import pandas as pd
from preprocessing.data_preprocessing import load_data, clean_data, normalize_data

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, None],
            'B': [5, 6, None, 8, 9],
            'C': [10, 11, 12, 13, 14]
        })
    
    def test_load_data(self):
        df = load_data('test_data.csv')
        self.assertIsInstance(df, pd.DataFrame)

    def test_clean_data(self):
        cleaned_df = clean_data(self.df)
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)

    def test_normalize_data(self):
        cleaned_df = clean_data(self.df)
        normalized_df = normalize_data(cleaned_df, columns=['A', 'B'])
        self.assertAlmostEqual(normalized_df['A'].mean(), 0, delta=1e-6)
        self.assertAlmostEqual(normalized_df['A'].std(), 1, delta=1e-6)

if __name__ == '__main__':
    unittest.main()

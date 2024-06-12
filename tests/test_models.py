import unittest
import numpy as np
from sklearn.datasets import make_regression
from stacking.stacking import train_stacking_model

class TestModels(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_regression(n_samples=100, n_features=4, noise=0.1)
    
    def test_train_stacking_model(self):
        model, val_predictions, y_val = train_stacking_model(self.X, self.y)
        self.assertEqual(len(val_predictions), len(y_val))
        self.assertTrue(np.isfinite(val_predictions).all())

if __name__ == '__main__':
    unittest.main()

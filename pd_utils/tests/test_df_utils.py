
import unittest
import pandas as pd
import numpy as np

import sys
sys.path.append('..')

# from pd_utils.series_utils import qtl_clip
# import pd_utils

from pd_utils import Series

# -------------------------------------------------------------------------
# Tests for the DataFrame weighted accessor methods
# -------------------------------------------------------------------------
class TestWtdDataFrameAccessor(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.df = pd.DataFrame({
            "A": np.random.normal(size=100),
            "B": np.random.uniform(0, 10, size=100),
            "C": np.random.choice([1, 2, 3, 4, 5], size=100)
        })
        self.wts = pd.Series(np.random.rand(100), index=self.df.index)

    def test_mean(self):
        weighted_means = self.df.wtd.mean(self.wts)
        self.assertIsInstance(weighted_means, pd.Series)
        for col in ["A", "B", "C"]:
            self.assertIn(col, weighted_means.index)

    def test_std(self):
        weighted_std = self.df.wtd.std(self.wts)
        self.assertIsInstance(weighted_std, pd.Series)
        for col in ["A", "B", "C"]:
            self.assertIn(col, weighted_std.index)

    def test_quantile(self):
        quantiles = self.df.wtd.quantile(q=0.5, wts=self.wts)
        self.assertIsInstance(quantiles, pd.Series)
        for col in ["A", "B", "C"]:
            self.assertIn(col, quantiles.index)

    def test_corr(self):
        corr_matrix = self.df.wtd.corr(self.wts)
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertTrue(set(["A", "B", "C"]).issubset(corr_matrix.index))

    def test_describe(self):
        desc = self.df.wtd.describe(percentiles=[0.25, 0.5, 0.75], wts=self.wts, )
        # The returned object should be a DataFrame with rows corresponding to statistics and
        # columns corresponding to each numeric column.
        self.assertIsInstance(desc, pd.DataFrame)
        for col in ["A", "B", "C"]:
            self.assertIn(col, desc.columns)
        self.assertIn("mean", desc.index)
        self.assertIn("std", desc.index)


# -------------------------------------------------------------------------
# Main block to run tests
# -------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
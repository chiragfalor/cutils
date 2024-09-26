import unittest
import pandas as pd
import numpy as np

import sys
sys.path.append('..')

# from pd_utils.series_utils import qtl_clip
# import pd_utils

from pd_utils import Series

# write tests for qtl_clip
class TestQtlClip(unittest.TestCase):
    def setUp(self):
        # seed
        np.random.seed(42)

        s1: Series = pd.Series(np.random.normal(size=1000))
        s1.iloc[0] = 100
        s1.iloc[1] = -200
        s1.iloc[2] = np.inf
        s1.iloc[3] = -np.inf
        s1.iloc[4] = np.nan
        s1.iloc[5] = np.nan

        s2: Series = pd.Series(np.random.choice([1, 2, 3, 4, 5], size=1000))
        s2.iloc[0] = 100
        s2.iloc[1] = -200

        s3: Series = pd.Series(np.random.normal(size=100_000))

        self.series_list = [s1, s2, s3]

    def test_no_mutation(self):
        for s in self.series_list:
            original = s.copy()
            s.qtl_clip()
            pd.testing.assert_series_equal(s, original, check_exact=True)

    def test_default_qtl_clip(self):
        for s in self.series_list:
            clipped = s.qtl_clip()
            self.assertTrue(clipped.min() >= s.quantile(0.01))
            self.assertTrue(clipped.max() <= s.quantile(0.99))
            self.assertAlmostEqual(clipped.min(), s.quantile(0.01), places=2)
            self.assertAlmostEqual(clipped.max(), s.quantile(0.99), places=2)

    def test_custom_qtl_clip(self):
        for s in self.series_list:
            clipped = s.qtl_clip(qtl=0.05)
            self.assertTrue(clipped.min() >= s.quantile(0.05))
            self.assertTrue(clipped.max() <= s.quantile(0.95))
            self.assertAlmostEqual(clipped.min(), s.quantile(0.05, interpolation='higher'), places=2)
            self.assertAlmostEqual(clipped.max(), s.quantile(0.95, interpolation='lower'), places=2)

    def test_lower_qtl_clip(self):
        for s in self.series_list:
            clipped = s.qtl_clip(lower_qtl=0.1)
            self.assertTrue(clipped.min() >= s.quantile(0.1))
            self.assertEqual(clipped.max(), s.max())
            self.assertAlmostEqual(clipped.min(), s.quantile(0.1, interpolation='higher'), places=2)

    def test_upper_qtl_clip(self):
        for s in self.series_list:
            clipped = s.qtl_clip(upper_qtl=0.9)
            self.assertEqual(clipped.min(), s.min())
            self.assertTrue(clipped.max() <= s.quantile(0.9))
            self.assertAlmostEqual(clipped.max(), s.quantile(0.9, interpolation='lower'), places=2)

    def test_both_qtl_clip(self):
        for s in self.series_list:
            clipped = s.qtl_clip(lower_qtl=0.1, upper_qtl=0.9)
            self.assertTrue(clipped.min() >= s.quantile(0.1))
            self.assertTrue(clipped.max() <= s.quantile(0.9))
            self.assertAlmostEqual(clipped.min(), s.quantile(0.1, interpolation='higher'), places=2)
            self.assertAlmostEqual(clipped.max(), s.quantile(0.9, interpolation='lower'), places=2)

    def test_no_effect(self):
        for s in self.series_list:
            clipped = s.qtl_clip(lower_qtl=0, upper_qtl=1)
            pd.testing.assert_series_equal(clipped, s)

    def test_invalid_qtl(self):
        for s in self.series_list:
            with self.assertRaises(ValueError):
                s.qtl_clip(qtl=1.5)
            with self.assertRaises(ValueError):
                s.qtl_clip(lower_qtl=-0.1)
            with self.assertRaises(ValueError):
                s.qtl_clip(upper_qtl=1.1)

class TestWtdMean(unittest.TestCase):
    def setUp(self):
        # seed
        np.random.seed(1)

        self.s1: Series = pd.Series(np.random.normal(size=1000))
        self.s1.iloc[0] = np.nan
        self.s1.iloc[1] = np.nan

        self.s2: Series = pd.Series(np.random.choice([1, 2, 3, 4, 5], size=1000))
        self.s3: Series = pd.Series(np.random.normal(size=100_000))

        self.wts1 = pd.Series(np.random.rand(1000)).abs()
        self.wts2 = pd.Series(np.random.rand(1000)).abs()
        self.wts3 = pd.Series(np.random.rand(100_000)).abs()

        self.series_list = [self.s1, self.s2, self.s3]
        self.wts_list = [self.wts1, self.wts2, self.wts3]



    def test_no_weights(self):
        for s in self.series_list:
            self.assertAlmostEqual(s.wtd.mean(), s.mean())

    def test_wtd_mean_with_zero_weights(self):
        s: Series = pd.Series([1, 2, 3])
        wts = pd.Series([0, 0, 0])
        with self.assertRaises(ZeroDivisionError):
            s.wtd.mean(wts)

    def test_with_weights(self):
        for s, wts in zip(self.series_list, self.wts_list):
            df = pd.DataFrame({'s': s, 'wts': wts})
            df = df.dropna()
            self.assertAlmostEqual(s.wtd.mean(wts), np.dot(df['s'], df['wts']) / df['wts'].sum())

    def test_missing_values(self):
        s = self.s1.copy()
        s.iloc[0] = np.nan
        wts = self.wts1.copy()
        wts.iloc[1] = np.nan
        self.assertAlmostEqual(s.wtd.mean(wts), np.average(s[~s.isnull() & ~wts.isnull()], weights=wts[~s.isnull() & ~wts.isnull()]))

    def test_no_mutation(self):
        for s, wts in zip(self.series_list, self.wts_list):
            original = s.copy()
            original_wts = wts.copy()
            s.wtd.mean(wts)
            pd.testing.assert_series_equal(s, original, check_exact=True)
            pd.testing.assert_series_equal(wts, original_wts, check_exact=True)



class TestWtdStd(unittest.TestCase):
    def setUp(self):
        # Set a seed for reproducibility
        np.random.seed(42)

        # Sample Series
        self.s1: Series = pd.Series(np.random.normal(size=1000))
        self.s2: Series = pd.Series([1] * 1000)
        self.s3: Series = pd.Series(range(1, 11), dtype=float)

        # Weights
        self.wts1 = pd.Series(np.random.uniform(0, 1, size=1000))
        self.wts2 = pd.Series(range(1, 11), dtype=float)
        self.wts3 = pd.Series([0] * 9 + [1], dtype=float)  # All zeros except last

        self.series_list = [self.s1, self.s2, self.s3]
        self.weights_list = [self.wts1, self.wts2, self.wts3]

    def test_no_weights(self):
        # Test without weights (should be equal to standard deviation)
        for s in self.series_list:
            result = s.wtd.std()
            expected = s.std(ddof=0)
            self.assertAlmostEqual(result, expected, places=5)

    def test_with_weights(self):
        # Test with weights
        result = self.s2.wtd.std(self.wts1)
        expected = 0.0  # Since the weighted variance of equal numbers is zero
        self.assertAlmostEqual(result, expected, places=5)

    def test_weighted_std_with_one_datapoint(self):
        # Test with weights having zeros
        # with self.assertRaises(ZeroDivisionError):
        result = self.s3.wtd.std(self.wts3)
        expected = 0.0  # Since all wts are concentrated on the last element
        self.assertAlmostEqual(result, expected, places=5)


    def test_inconsistent_length(self):
        # Test with different length of weights
        with self.assertRaises(ValueError):
            self.s1.wtd.std(pd.Series([1, 2, 3]))

    def test_nan_handling(self):
        # Test when the Series contains NaN values
        s_with_nan: Series = pd.Series([1, 2, np.nan, 4, 5])
        wts = pd.Series([1, 1, 1, 1, 1])
        result = s_with_nan.wtd.std(wts)
        expected = s_with_nan.dropna().std(ddof=0)
        self.assertAlmostEqual(result, expected, places=5)

    def test_weights_sum_to_zero(self):
        # Test with weights summing to zero
        s: Series = pd.Series([1, 2, 3])
        wts = pd.Series([0, 0, 0])
        with self.assertRaises(ZeroDivisionError):
            s.wtd.std(wts)

    def test_no_mutation(self):
        # Test that the original series is not mutated
        for s in self.series_list:
            original = s.copy()
            s.wtd.std()
            pd.testing.assert_series_equal(s, original, check_exact=True)


if __name__ == '__main__':
    unittest.main()
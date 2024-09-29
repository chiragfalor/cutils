import unittest
import pandas as pd
import numpy as np
import pathlib
import os
import shutil
from .cachify import CachifyManager, cachify_factory


class TestCachifyManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup a temporary cache directory
        # get current working directory
        cur_dir = pathlib.Path(__file__).parent
        cls.cache_dir = cur_dir / "test_cache"
        if not cls.cache_dir.exists():
            cls.cache_dir.mkdir()

    @classmethod
    def tearDownClass(cls):
        # Remove the cache directory after tests
        if cls.cache_dir.exists():
            shutil.rmtree(cls.cache_dir)

    def setUp(self):
        # Setup test function for caching
        self.test_dir = self.cache_dir / "test_func"
        if not self.test_dir.exists():
            self.test_dir.mkdir()

        self.test_cachify = cachify_factory(self.cache_dir)

        self.call_count = 0

        @self.test_cachify()
        def test_func(x, y):
            self.call_count += 1
            return x + y
        
        self.test_func = test_func

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_cache_update(self):
        # Test cache update with new arguments
        result = self.test_func(1, 2)
        cache_file = self.test_dir / "v0" / "1__2.pkl"
        self.assertTrue(cache_file.exists())
        self.assertEqual(result, 3)

    def test_cache_retrieval(self):
        # Test that cache is correctly retrieved on subsequent calls
        result1 = self.test_func(3, 4)
        result2 = self.test_func(3, 4)
        self.assertEqual(result1, result2)

    def test_cache_info(self):
        # Test cache info to retrieve cache details
        self.test_func(1, 2)
        cache_info = self.test_func.cache_info()
        self.assertTrue(len(cache_info) == 1)

    def test_cache_clear(self):
        # Test that cache is cleared properly
        self.test_func(5, 6)
        self.test_func.cache_clear()
        cache_files = os.listdir(self.test_dir / "v0")
        self.assertEqual(len(cache_files), 0)

    def test_cache_nonexistent_file(self):
        # Test for file not found scenario in cache retrieval
        with self.assertRaises(FileNotFoundError):
            CachifyManager(self.test_func, self.test_dir)._load_result(self.test_dir / "nonexistent.pkl")

    def test_cache_different_args(self):
        # Test caching works for different arguments
        result1 = self.test_func(1, 2)
        result2 = self.test_func(3, 4)
        self.assertNotEqual(result1, result2)

    def test_cache_with_ndarray(self):
        # Test caching with numpy arrays as arguments
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        result = self.test_func(arr1, arr2)
        self.assertTrue(isinstance(result, np.ndarray))

    def test_cache_with_dataframe(self):
        # Test caching with pandas DataFrame as arguments
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        result = self.test_func(df1, df2)
        self.assertTrue(isinstance(result, pd.DataFrame))

    def test_cache_update_function(self):
        # Test that cache is updated when cache_update is called
        result1 = self.test_func(10, 20)
        result2 = self.test_func.cache_update(10, 20)
        self.assertEqual(result1, result2)

    def test_cache_with_kwargs(self):
        # Test caching with keyword arguments
        result = self.test_func(5, y=6)
        cache_file = self.test_dir / "v0" / "5__'y'=6.pkl"
        self.assertTrue(cache_file.exists())
        self.assertEqual(result, 11)

    def test_cache_with_multiple_data_types(self):
        # Test caching with a mix of data types
        df = pd.DataFrame({'x': [1, 2, 3]})
        arr = np.array([10, 20, 30]).reshape(-1, 1)
        result = self.test_func(df, arr)
        self.assertTrue(isinstance(result, pd.DataFrame))

    def test_cache_load_objects(self):
        # Test cache info with load_objs=True to retrieve cached objects
        self.test_func(7, 8)
        cache_info = self.test_func.cache_info(load_objs=True)
        cached_result = list(cache_info.values())[0]
        self.assertEqual(cached_result, 15)

    def test_avoid_recomputation_for_same_args(self):
        # Call the function twice with the same arguments
        self.test_func(10, 20)
        first_call_count = self.call_count
        self.test_func(10, 20)
        second_call_count = self.call_count

        # Assert the function was only computed once
        self.assertEqual(first_call_count, 1)
        self.assertEqual(second_call_count, 1)

    def test_recomputation_for_different_args(self):
        # Call the function twice with different arguments
        self.test_func(10, 20)
        first_call_count = self.call_count
        self.test_func(30, 40)
        second_call_count = self.call_count

        # Assert the function was computed twice (once for each unique argument)
        self.assertEqual(first_call_count, 1)
        self.assertEqual(second_call_count, 2)


    def test_recomputation_with_cache_update(self):
        # Call the function twice with the same arguments
        self.test_func(10, 20)
        first_call_count = self.call_count
        self.test_func.cache_update(10, 20)
        second_call_count = self.call_count
        self.test_func(10, 20)
        third_call_count = self.call_count
        self.test_func(x=10, y=20) # Different call signature
        fourth_call_count = self.call_count

        # Assert the function was only computed once
        self.assertEqual(first_call_count, 1)
        self.assertEqual(second_call_count, 2)
        self.assertEqual(third_call_count, 2)
        self.assertEqual(fourth_call_count, 3)


if __name__ == '__main__':
    unittest.main()

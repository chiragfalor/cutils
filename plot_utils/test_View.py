import unittest
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
# Assuming the original code is in a file named 'plotting_module.py'
from plot_utils.View import View, Plot, PlotArgs

class TestViewOperations(unittest.TestCase):
    def setUp(self):
        # Create mock plots with different hash values

        def linear_plot(ax):
            x = np.linspace(0, 10, 100)
            y = x
            ax.plot(x, y, label="Linear")

            return ax
        self.plot1 = Plot(linear_plot, title="Linear Plot", xlabel="X", ylabel="Y", legend=['lin'])
        self.plot1_hash = hash(self.plot1)

        def sine_plot(ax):
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y)
            return ax
        self.plot2 = Plot(sine_plot, title="Sine Plot", xlabel="X", ylabel="Y", legend=['sin'], xlim=(0, 5))
        self.plot2_hash = hash(self.plot2)

        def cosine_plot(ax):
            x = np.linspace(0, 10, 100)
            y = np.cos(x)
            ax.plot(x, y)
            return ax
        self.plot3 = Plot(cosine_plot, title="Cosine Plot", xlabel="X_new", ylabel="Y_new")
        self.plot3_hash = hash(self.plot3)

        def exponential_plot(ax):
            x = np.linspace(0, 20, 100)
            y = 1.1**x
            ax.plot(x, y)
            return ax
        self.plot4 = Plot(exponential_plot, title="Exponential Plot", xlabel="X_new", ylabel="Y_new")
        self.plot4_hash = hash(self.plot4)


    def test_add_simple(self):
        view1 = View([self.plot1])
        view2 = View([self.plot2])
        combined_view = view1 + view2
        np.testing.assert_array_equal(combined_view.arrangement, np.array([[self.plot1_hash, self.plot2_hash]]))
        self.assertEqual(len(combined_view.plots), 2)

    def test_add_complex(self):
        view1 = View([self.plot1, self.plot2], np.array([[self.plot1_hash, self.plot2_hash]]))
        view2 = View([self.plot3, self.plot4], np.array([[self.plot3_hash], [self.plot4_hash]]))
        combined_view = view1 + view2
        expected_arrangement = np.array([[self.plot1_hash, self.plot2_hash, self.plot3_hash],
                                            [self.plot1_hash, self.plot2_hash, self.plot4_hash]])
        np.testing.assert_array_equal(combined_view.arrangement, expected_arrangement)
        self.assertEqual(len(combined_view.plots), 4)

    def test_truediv_simple(self):
        view1 = View([self.plot1])
        view2 = View([self.plot2])
        combined_view = view1 / view2
        np.testing.assert_array_equal(combined_view.arrangement, np.array([[self.plot1_hash], [self.plot2_hash]]))
        self.assertEqual(len(combined_view.plots), 2)

    def test_truediv_complex(self):
        view1 = View([self.plot1, self.plot2], np.array([[1, 2]]))
        view2 = View([self.plot3, self.plot4], np.array([[3], [4]]))
        combined_view = view1 / view2
        expected_arrangement = np.array([[1, 2],
                                         [3, 3],
                                         [4, 4]])
        np.testing.assert_array_equal(combined_view.arrangement, expected_arrangement)
        self.assertEqual(len(combined_view.plots), 4)

    def test_mul_simple(self):
        view1 = View([self.plot1])
        view2 = View([self.plot2])
        
        # Mock the multiplication of plots

        combined_view = view1 * view2
        np.testing.assert_array_equal(combined_view.arrangement.shape, (1, 1))
        self.assertEqual(len(combined_view.plots), 1)

    def test_mul_error_multiple_plots(self):
        view1 = View([self.plot1, self.plot2])
        view2 = View([self.plot3])
        with self.assertRaises(AssertionError):
            _ = view1 * view2

    def test_reshape_2d_to_1d(self):
        view = View([self.plot1, self.plot2, self.plot3, self.plot4], np.array([[self.plot1_hash, self.plot2_hash], [self.plot3_hash, self.plot4_hash]]))
        reshaped_view = view.reshape(1, 4)
        np.testing.assert_array_equal(reshaped_view.arrangement, np.array([[self.plot1_hash, self.plot2_hash, self.plot3_hash, self.plot4_hash]]))
        self.assertEqual(len(reshaped_view.plots), 4)

    def test_reshape_1d_to_2d(self):
        view = View([self.plot1, self.plot2, self.plot3, self.plot4])
        reshaped_view = view.reshape(2, 2)
        np.testing.assert_array_equal(reshaped_view.arrangement, np.array([[self.plot1_hash, self.plot2_hash], [self.plot3_hash, self.plot4_hash]]))
        self.assertEqual(len(reshaped_view.plots), 4)

    def test_transpose_2d(self):
        view = View([self.plot1, self.plot2, self.plot3, self.plot4], np.array([[self.plot1_hash, self.plot2_hash], [self.plot3_hash, self.plot4_hash]]))
        transposed_view = view.T
        np.testing.assert_array_equal(transposed_view.arrangement, np.array([[self.plot1_hash, self.plot3_hash], [self.plot2_hash, self.plot4_hash]]))
        self.assertEqual(len(transposed_view.plots), 4)

    def test_transpose_1d(self):
        view = View([self.plot1, self.plot2, self.plot3])
        transposed_view = view.T
        np.testing.assert_array_equal(transposed_view.arrangement, np.array([[self.plot1_hash], [self.plot2_hash], [self.plot3_hash]]))
        self.assertEqual(len(transposed_view.plots), 3)

    def test_complex_operations(self):
        view1 = View({1: self.plot1, 2: self.plot2}, np.array([[1], [2]]))
        view2 = View({3: self.plot3, 4: self.plot4}, np.array([[3, 4]]))
        
        # Combine views horizontally, then vertically, then transpose
        combined_view = (view1 + view2) / View({5: self.plot1})
        final_view = combined_view.T

        expected_arrangement = np.array([[1, 3, 4],
                                         [2, 3, 4],
                                         [5, 5, 5]]).T
        np.testing.assert_array_equal(final_view.arrangement, expected_arrangement)
        self.assertEqual(len(final_view.plots), 5)

if __name__ == '__main__':
    unittest.main()
from automl_zero.hierarchical.evo import get_ops_from_meta_level
import unittest
import numpy as np

class TestGenerateCachedMetalevels(unittest.TestCase):

    def setUp(self):
        self.combine_OPs = np.array([
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]
            ],
            [
                [9, 10, 11],
                [12, 13, 14],
                [15, 16, 17]
            ],
            [
                [18, 19, 20],
                [21, 22, 23],
                [24, 25, 26]
            ]
        ])

    def test_generate_cached_metalevels(self):
        expected_result = np.zeros(shape=(27, 27, 81))
        result = _generate_cached_metalevels(self.combine_OPs, 3)

        np.testing.assert_array_equal(result, expected_result)

    def test_modified_metalevel_op(self):
        # Modify the combine_OPs array to set the second op of the second metalevel to [0, 1, 2]
        self.combine_OPs[1][1] = [0, 1, 2]

        # Generate the cached metalevels for the modified combine_OPs
        expected_result = np.zeros(shape=(27, 27, 81))
        expected_result[2, 1] = self.combine_OPs[2][1][0]
        expected_result[2, 1, :3] = self.combine_OPs[1][1]
        result = _generate_cached_metalevels(self.combine_OPs, 3, modified_metalevel_idx=1, modified_op_idx=1)

        np.testing.assert_array_equal(result, expected_result)
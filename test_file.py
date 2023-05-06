from automl_zero.hierarchical.evo import get_ops_from_meta_level, _generate_cached_metalevels
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
                [0,0,1],
                [1,2,0],
                [1,1,1]
            ],
            [
                [0,1,0],
                [2,2,2],
                [0,1,0]
            ]
        ])

    def test_get_ops_from_meta_level(self):
        meta_level = 3
        ops_to_resolve = [0, 1]
        target_level = 0
        
        result = get_ops_from_meta_level(meta_level, ops_to_resolve, self.combine_OPs, target_level)
        expected_result = [0,1,2,0,1,2,3,4,5,3,4,5,6,7,8,0,1,2,0,1,2,0,1,2,3,4,5,3,4,5,3,4,5,3,4,5,3,4,5,3,4,5,3,4,5,3,4,5,3,4,5,3,4,5]
        self.assertTrue( (expected_result == result).all())


    def test_generate_cached_metalevels(self):
        METALEVEL_COUNT = 3
        NUMBER_OF_OPS = 3

        expected_result = np.array(
            [
                [
                    [ 0,  1 , 2] + [-1] * 24,
                    [ 3,  4,  5] + [-1] * 24,
                    [ 6,  7,  8]  + [-1] * 24
                ],

                [
                    [ 0,  1, 2,  0,  1,  2,  3,  4,  5 ]
                    [ 3  4  5  6  7  8  0  1  2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
                    -1 -1 -1 -1]
                    [ 3  4  5  3  4  5  3  4  5 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
                    -1 -1 -1 -1]],

                [
                    [ 0  1  2  0  1  2  3  4  5  3  4  5  6  7  8  0  1  2  0  1  2  0  1
                    2  3  4  5]
                    [ 3  4  5  3  4  5  3  4  5  3  4  5  3  4  5  3  4  5  3  4  5  3  4
                    5  3  4  5]
                    [ 0  1  2  0  1  2  3  4  5  3  4  5  6  7  8  0  1  2  0  1  2  0  1
                    2  3  4  5]
                ]
        ])
        result = _generate_cached_metalevels(self.combine_OPs, 0)
        result = result.astype(np.int64)
        print(result)

        self.assertTrue( (expected_result == result).all())

    def test_modified_metalevel_op(self):
        # Modify the combine_OPs array to set the second op of the second metalevel to [0, 1, 2]
        self.combine_OPs[1][1] = [0, 1, 2]

        # Generate the cached metalevels for the modified combine_OPs
        expected_result = np.zeros(shape=(27, 27, 81))
        expected_result[2, 1] = self.combine_OPs[2][1][0]
        expected_result[2, 1, :3] = self.combine_OPs[1][1]
        result = _generate_cached_metalevels(self.combine_OPs, 0)
        #print(result)
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
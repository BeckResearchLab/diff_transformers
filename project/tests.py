import unittest
import definitions
from norm_def import normalize_data, normalize_point, normalize_points_list, unnormalize_data, unnormalize_point, unnormalize_points_list
import sql_def
## python -m unittest tests.py

class TestDefinitions(unittest.TestCase):

    def test_add(self):
        self.assertEqual(definitions.add(1, 2), 3)

    def test_subtract(self):
        self.assertEqual(definitions.subtract(5, 3), 2)

    def test_seperate_trajectory(self):
        self.assertEqual(definitions.separate_trajectories([]), [])

        self.assertEqual(
            definitions.separate_trajectories([
                ('exp1', 'sl1', 'vid1', 'track1', 1, 10, 10),
                ('exp1', 'sl1', 'vid1', 'track1', 2, 20, 20),
                ('exp1', 'sl1', 'vid1', 'track1', 3, 30, 30)
            ]),
            [[(1, 10, 10), (2, 20, 20), (3, 30, 30)]]
        )

        self.assertEqual(
            definitions.separate_trajectories([
                ('exp1', 'sl1', 'vid1', 'track1', 1, 10, 10),
                ('exp1', 'sl1', 'vid1', 'track2', 1, 15, 15),
                ('exp1', 'sl1', 'vid1', 'track1', 2, 20, 20),
                ('exp1', 'sl1', 'vid1', 'track2', 2, 25, 25)
            ]),
            [[(1, 10, 10), (2, 20, 20)], [(1, 15, 15), (2, 25, 25)]]
        )

        self.assertEqual(
           definitions.separate_trajectories([
                ('exp1', 'sl1', 'vid1', 'track1', 1, 10, 10),
                ('exp2', 'sl2', 'vid2', 'track1', 1, 50, 50),
                ('exp1', 'sl1', 'vid1', 'track1', 2, 20, 20),
                ('exp2', 'sl2', 'vid2', 'track1', 2, 60, 60)
            ]),
            [[(1, 10, 10), (2, 20, 20)], [(1, 50, 50), (2, 60, 60)]]
        )

        self.assertEqual(
            definitions.separate_trajectories([
                ('exp1', 'sl1', 'vid1', 'track1', 2, 20, 20),
                ('exp1', 'sl1', 'vid1', 'track1', 1, 10, 10),
                ('exp1', 'sl1', 'vid1', 'track1', 3, 30, 30)
            ]),
            [[(2, 20, 20), (1, 10, 10), (3, 30, 30)]]
        )

        self.assertEqual(
            definitions.separate_trajectories([
                ('exp1', 'sl1', 'vid1', 'track1', 1, 10, 10),
                ('exp1', 'sl1', 'vid2', 'track1', 1, 50, 50),
                ('exp1', 'sl1', 'vid1', 'track1', 2, 20, 20),
                ('exp1', 'sl1', 'vid2', 'track1', 2, 60, 60)
            ]),
            [[(1, 10, 10), (2, 20, 20)], [(1, 50, 50), (2, 60, 60)]]
        )


    def test_mask_point_at_index(self):
        # 0 - 1 - many test:

        # 0 test
        self.assertEqual(definitions.mask_point_at_index([], 1), ([], []))

        # 1 test
        self.assertEqual(definitions.mask_point_at_index([[(2, 2)]], 0), ([[None]], [(2, 2)]))
        self.assertEqual(definitions.mask_point_at_index([[(2, 2), (45, 12)]], 1), ([[(2, 2), None]], [(45, 12)]))

        # many test
        self.assertEqual(definitions.mask_point_at_index([[(2, 2), (45, 12)], [(1,1), (2, 3)]], 1), ([[(2, 2), None], [(1, 1), None]], [(45, 12), (2, 3)]))
        self.assertEqual(definitions.mask_point_at_index([[(2, 2)], [(1,1)]], 0), ([[None], [None]], [(2,2), (1,1)]))

    def test_find_min_length(self):
        # 0 - 1 - many test:

        # 0 test
        self.assertEqual(definitions.find_min_length([]), 0)

        # 1 test
        self.assertEqual(definitions.find_min_length([[(2, 2)]]), 1)
        self.assertEqual(definitions.find_min_length([[(2, 2), (2, 2), (2, 2), (2, 2)]]), 4)

        # many test
        self.assertEqual(definitions.find_min_length([[(2, 2), (2, 2), (2, 2), (2, 2)], [(1, 1)]]), 1)
        self.assertEqual(definitions.find_min_length([[(2, 2), (2, 2), (2, 2), (2, 2)], [(1, 1), (1, 1)], [(1, 1), (1, 1), (1, 1)]]), 2)

    def test_find_max_length(self):
        # 0 - 1 - many test:

        # 0 test
        self.assertEqual(definitions.find_max_length([]), 0)

        # 1 test
        self.assertEqual(definitions.find_max_length([[(2, 2)]]), 1)
        self.assertEqual(definitions.find_max_length([[(2, 2), (2, 2), (2, 2), (2, 2)]]), 4)

        # many test
        self.assertEqual(definitions.find_max_length([[(2, 2), (2, 2), (2, 2), (2, 2)], [(1, 1)]]), 4)
        self.assertEqual(definitions.find_max_length([[(2, 2), (2, 2), (2, 2), (2, 2)], [(1, 1), (1, 1)], [(1, 1), (1, 1), (1, 1)]]), 4)

    # def test_line_generator(self):
    #     # Test cases for length = 0 with default parameters
    #     self.assertEqual(definitions.random_trajectory_straight(0), [])

    #     # Test cases for length = 0 with non-default parameters, including rotation
    #     self.assertEqual(definitions.random_trajectory_straight(0, 2, 3, 10, True, 45), [])
    #     self.assertEqual(definitions.random_trajectory_straight(0, -1, 4, 5, True, 30), [])

    #     # Test cases for length = 1 with default parameters
    #     self.assertEqual(definitions.random_trajectory_straight(1), [(0, 0)])
    #     self.assertEqual(definitions.random_trajectory_straight(1), [(0, 0)])

    #     # Test cases for length = 1 with varying start positions and no rotation
    #     self.assertEqual(definitions.random_trajectory_straight(1, 1, 1), [(1, 1)])
    #     self.assertEqual(definitions.random_trajectory_straight(1, 3, -2), [(3, -2)])

    #     # Test case for length = 1 with rotation enabled, but should have no effect as it's a single point
    #     self.assertEqual(definitions.random_trajectory_straight(1, 2, 3, 1, True, 90), [(2, 3)])
    #     self.assertEqual(definitions.random_trajectory_straight(1, 0, 5, 1, True, 45), [(0, 5)])

    #     # Test cases for length > 1 with default parameters (no rotation)
    #     self.assertEqual(definitions.random_trajectory_straight(3, 0, 0, 1), [(0, 0), (1, 0), (2, 0)])
    #     self.assertEqual(definitions.random_trajectory_straight(3, 0, 0, 1), [(0, 0), (1, 0), (2, 0)])

    #     # Test cases for length > 1 with non-default parameters, including spacing
    #     self.assertEqual(definitions.random_trajectory_straight(3, 1, 1, 2), [(1, 1), (3, 1), (5, 1)])
    #     self.assertEqual(definitions.random_trajectory_straight(3, 2, 2, 3), [(2, 2), (5, 2), (8, 2)])

    #     # Test cases for length > 1 with rotation enabled, assuming rotation should affect the trajectory
    #     self.assertEqual(definitions.random_trajectory_straight(3, 1, 1, 1, True, 90, 0), [(1.0, 1.0), (1.0, 2.0), (1.0, 3.0)])
    #     self.assertEqual(definitions.random_trajectory_straight(3, 2, 2, 1, True, 180, 0), [(2.0, 2.0), (1.0, 2.0), (0.0, 2.0)])

    #     # Test cases for length > 1 with negative spacing, ensuring the function can handle negative increments
    #     self.assertEqual(definitions.random_trajectory_straight(3, 5, 5, -1), [(5, 5), (4, 5), (3, 5)])
    #     self.assertEqual(definitions.random_trajectory_straight(3, 6, 4, -2), [(6, 4), (4, 4), (2, 4)])

    #     # Test cases with non-integer start positions, checking for rounding behavior if applicable
    #     self.assertEqual(definitions.random_trajectory_straight(2, 1.5, 2.5, 1), [(1.5, 2.5), (2.5, 2.5)])
    #     self.assertEqual(definitions.random_trajectory_straight(2, 3.7, 1.2, 1), [(3.7, 1.2), (4.7, 1.2)])

    #     # Test cases with decimal precision, ensuring the function rounds the coordinates as specified
    #     self.assertEqual(definitions.random_trajectory_straight(2, 0, 0, 1, decimals=2), [(0, 0), (1, 0)])
    #     self.assertEqual(definitions.random_trajectory_straight(2, 0.123456, 0.654321, 1, decimals=3), [(0.123, 0.654), (1.123, 0.654)])


    def test_data_cut(self):
        self.assertEqual(definitions.listTrim([], 0), [])

        self.assertEqual(definitions.listTrim([(1, 2), (3, 2)], 1), [(1, 2)])

        self.assertEqual(definitions.listTrim([(1, 2), (3, 2), (4, 5)], 2), [(1, 2), (3, 2)])

    # def test_separate_data(self):
    #     # Test case for empty input
    #     self.assertEqual(definitions.separate_data([], True), ([], [], []))

    #     # Test case for a single point
    #     # self.assertEqual(
    #     #     definitions.separate_data([(1, 10, 100)], True),
    #     #     ([1], [10], [100])
    #     # )

    #     # Test case for multiple points
    #     self.assertEqual(
    #         definitions.separate_data([[(1, 10, 100), (2, 20, 200), (3, 30, 300)]], True),
    #         ([1, 2, 3], [10, 20, 30], [100, 200, 300])
    #     )

    #     # Test case for non-sequential frames
    #     self.assertEqual(
    #         definitions.separate_data([(3, 30, 300), (1, 10, 100), (2, 20, 200)], True),
    #         ([3, 1, 2], [30, 10, 20], [300, 100, 200])
    #     )

    #     # Test case for negative coordinates
    #     self.assertEqual(
    #         definitions.separate_data([(1, -10, -100), (2, -20, -200), (3, -30, -300)], True),
    #         ([1, 2, 3], [-10, -20, -30], [-100, -200, -300])
    #     )

    #     # Test case with decimal points
    #     self.assertEqual(
    #         definitions.separate_data([(1, 10.5, 100.5), (2, 20.5, 200.5), (3, 30.5, 300.5)], True),
    #         ([1, 2, 3], [10.5, 20.5, 30.5], [100.5, 200.5, 300.5])
    #     )

    #     # Test case for large number of points
    #     self.assertEqual(
    #         definitions.separate_data([(i, i*10, i*100) for i in range(1, 101)], True),
    #         (list(range(1, 101)), [i*10 for i in range(1, 101)], [i*100 for i in range(1, 101)])
    #     )


    def test_cut_frame(self):
        self.assertEqual(definitions.cut_frame_data([]), [])
        self.assertEqual(definitions.cut_frame_data([[(1, 2, 3)]]), [[(2, 3)]])
        self.assertEqual(definitions.cut_frame_data([[(1, 2, 3), (2, 1, 1)]]), [[(2, 3), (1, 1)]])
        self.assertEqual(definitions.cut_frame_data([[(1, 2, 3), (2, 1, 1)], [(200, 2, 3)]]), [[(2, 3), (1, 1)], [(2, 3)]])

    def test_train_split(self):
        self.assertEqual(definitions.split_test_train([], 50), ([], []))

        self.assertEqual(definitions.split_test_train([[(1,2), (2,3), (3, 4)], [(1, 1), (2,2), (3,3)], [(2,4), (3,4), (4,4)]], 90), ([[(1,2), (2,3), (3, 4)], [(1, 1), (2,2), (3,3)]], [[(2,4), (3,4), (4,4)]]))


class TestSQL(unittest.TestCase):

    def test_sql(self):
        sql1_command = "SELECT Track_ID, Frame, X, Y FROM TRACKMATEDATA WHERE slide = 1 AND video = 1 LIMIT 4;"
        sql1_data = [(0, 63, 424.4435,1711.9775), (0,64,402.177,1711.2914), (0,65,424.0598,1709.0296), (0,66,430.233,1703.6202)]
        self.assertEqual(sql_def.data_from_sql("data/database.db", sql1_command), sql1_data)


class TestNormalizationFunctions(unittest.TestCase):

    def test_normalize_point(self):
        self.assertEqual(normalize_point((10, 20), 10, 20, 90, 80), (0, 0))
        self.assertEqual(normalize_point((100, 100), 10, 20, 90, 80), (1, 1))
        self.assertEqual(normalize_point(None, 10, 20, 90, 80), None)

    def test_normalize_points_list(self):
        self.assertEqual(normalize_points_list([(10, 20), (100, 100)], 10, 20, 90, 80), [(0, 0), (1, 1)])
        self.assertEqual(normalize_points_list([], 10, 20, 90, 80), [])
        self.assertEqual(normalize_points_list([None], 10, 20, 90, 80), [None])

    def test_normalize_data(self):
            data = [[(10, 20), (100, 100)], [(50, 70)]]
            expected = [[(0, 0), (1, 1)], [(0.4444, 0.625)]]
            normalized_data, min_x, min_y, range_x, range_y = normalize_data(data)
            
            for seq_index, sequence in enumerate(normalized_data):
                for point_index, point in enumerate(sequence):
                    with self.subTest(seq_index=seq_index, point_index=point_index):
                        self.assertAlmostEqual(point[0], expected[seq_index][point_index][0], places=4)
                        self.assertAlmostEqual(point[1], expected[seq_index][point_index][1], places=4)

    def test_unnormalize_point(self):
        self.assertEqual(unnormalize_point((0, 0), 10, 20, 90, 80), (10, 20))
        self.assertEqual(unnormalize_point((1, 1), 10, 20, 90, 80), (100, 100))
        self.assertEqual(unnormalize_point(None, 10, 20, 90, 80), None)

    def test_unnormalize_points_list(self):
        self.assertEqual(unnormalize_points_list([(0, 0), (1, 1)], 10, 20, 90, 80), [(10, 20), (100, 100)])
        self.assertEqual(unnormalize_points_list([], 10, 20, 90, 80), [])
        self.assertEqual(unnormalize_points_list([None], 10, 20, 90, 80), [None])

    def test_unnormalize_data(self):
        normalized_data = [[(0, 0), (1, 1)], [(0.4444, 0.625)]]
        expected = [[(10, 20), (100, 100)], [(49.996, 70)]]
        unnormalized_data = unnormalize_data(normalized_data, 10, 20, 90, 80)
        self.assertEqual(unnormalized_data, expected)








if __name__ == '__main__':
    unittest.main()
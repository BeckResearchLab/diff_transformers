import unittest
import definitions
import sql_def
## python -m unittest tests.py

class TestDefinitions(unittest.TestCase):

    def test_add(self):
        self.assertEqual(definitions.add(1, 2), 3)

    def test_subtract(self):
        self.assertEqual(definitions.subtract(5, 3), 2)

    def test_seperate_trajectory(self):
        # 0 - 1 - many test:

        # 0 test
        self.assertEqual(definitions.separate_trajectories([]), [])

        # 1 test
        self.assertEqual(definitions.separate_trajectories([(0, 1, 2, 2)]), [[(1, 2, 2)]])
        self.assertEqual(definitions.separate_trajectories([(0, 1, 2, 2), (0, 2, 45, 12)]), [[(1, 2, 2), (2, 45, 12)]])

        # many test
        self.assertEqual(definitions.separate_trajectories([(0, 1, 2, 2), (1, 52, 412, 21)]), [[(1, 2, 2)], [(52, 412, 21)]])
        self.assertEqual(definitions.separate_trajectories([(0, 1, 2, 2), (0, 2, 45, 12), (1, 1, 2, 2), (1, 2, 45, 12)]), [[(1, 2, 2), (2, 45, 12)], [(1, 2, 2), (2, 45, 12)]])

    def test_mask_point_at_index(self):
        # 0 - 1 - many test:

        # 0 test
        self.assertEqual(definitions.mask_point_at_index([], 1), ([], []))

        # 1 test
        self.assertEqual(definitions.mask_point_at_index([[(2, 2)]], 0), ([[(0, 0)]], [(2, 2)]))
        self.assertEqual(definitions.mask_point_at_index([[(2, 2), (45, 12)]], 1), ([[(2, 2), (0, 0)]], [(45, 12)]))

        # many test
        self.assertEqual(definitions.mask_point_at_index([[(2, 2), (45, 12)], [(1,1), (2, 3)]], 1), ([[(2, 2), (0, 0)], [(1, 1), (0, 0)]], [(45, 12), (2, 3)]))
        self.assertEqual(definitions.mask_point_at_index([[(2, 2)], [(1,1)]], 0), ([[(0, 0)], [(0, 0)]], [(2,2), (1,1)]))

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

    def test_line_generator(self):
        # 0 - 1 - many test:

        # 0 test
        self.assertEqual(definitions.random_trajectory_straight(0), [])
        self.assertEqual(definitions.random_trajectory_straight(0, 2, 3, 10, True), [])

        # 1 test
        self.assertEqual(definitions.random_trajectory_straight(1, 1, 1, 1), [(1,1)])
        self.assertEqual(definitions.random_trajectory_straight(1, 2, 13, 1), [(2,13)])
        self.assertEqual(definitions.random_trajectory_straight(1, 2, 13, 1, True), [(2,13)])

        # 2 test
        self.assertEqual(definitions.random_trajectory_straight(3, 1, 1, 1), [(1,1), (2,1), (3,1)])
        # self.assertEqual(definitions.random_trajectory_straight(3, 1, 1, 1, False, 180), [(1,1), (0,1), (-1,1)])
    def test_data_cut(self):
        self.assertEqual(definitions.listTrim([], 0), [])

        self.assertEqual(definitions.listTrim([(1, 2), (3, 2)], 1), [(1, 2)])

        self.assertEqual(definitions.listTrim([(1, 2), (3, 2), (4, 5)], 2), [(1, 2), (3, 2)])

    def test_separate_data(self):
        self.assertEqual(definitions.separate_data([]), ([], [], []))

        self.assertEqual(definitions.separate_data([(1, 1, 2), (2, 3, 2)]), ([1, 2], [1, 3], [2, 2]))

        # self.assertEqual(definitions.separate_data([(1, 2), (3, 2), (4, 5)], 2), [(1, 2), (3, 2)])

    def test_cut_frame(self):
        self.assertEqual(definitions.cut_frame_data([]), [])
        self.assertEqual(definitions.cut_frame_data([[(1, 2, 3)]]), [[(2, 3)]])
        self.assertEqual(definitions.cut_frame_data([[(1, 2, 3), (2, 1, 1)]]), [[(2, 3), (1, 1)]])
        self.assertEqual(definitions.cut_frame_data([[(1, 2, 3), (2, 1, 1)], [(200, 2, 3)]]), [[(2, 3), (1, 1)], [(2, 3)]])

class TestSQL(unittest.TestCase):

    def test_sql(self):
        sql1_command = "SELECT Track_ID, Frame, X, Y FROM TRACKMATEDATA WHERE slide = 1 AND video = 1 LIMIT 4;"
        sql1_data = [(0, 63, 424.4435,1711.9775), (0,64,402.177,1711.2914), (0,65,424.0598,1709.0296), (0,66,430.233,1703.6202)]
        self.assertEqual(sql_def.data_from_sql("data/database.db", sql1_command), sql1_data)

if __name__ == '__main__':
    unittest.main()
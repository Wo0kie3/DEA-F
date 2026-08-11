import math
import unittest

import pandas as pd

from candidate_refinement import _local_random_candidates, global_stratified_candidates


class StratifiedLocalSearchTests(unittest.TestCase):
    def test_two_dimensional_search_covers_each_grid_cell(self):
        centers = pd.DataFrame(
            [{"name": "center", "i1": 4.0, "i2": 5.0, "o1": 5.0}]
        )
        target = pd.Series({"i1": 8.2, "i2": 9.1, "o1": 5.0})

        points = _local_random_candidates(
            centers=centers,
            target_row=target,
            io_cols=["i1", "i2", "o1"],
            inputs=["i1", "i2"],
            outputs=["o1"],
            search_columns=["i1", "i2"],
            step_by_column={"i1": 1.0, "i2": 1.0},
            samples_per_center=400,
            step_multiplier=1.5,
            random_state=42,
            name_prefix="test",
            sampling_strategy="stratified",
        )

        self.assertEqual(len(points), 400)
        self.assertTrue(points["i1"].between(2.5, 5.5).all())
        self.assertTrue(points["i2"].between(3.5, 6.5).all())
        self.assertEqual(set(points["local_search_sampling_strategy"]), {"stratified"})

        x_bins = ((points["i1"] - 2.5) / 3.0 * 20).apply(math.floor)
        y_bins = ((points["i2"] - 3.5) / 3.0 * 20).apply(math.floor)
        occupied_cells = set(zip(x_bins, y_bins))
        self.assertEqual(len(occupied_cells), 400)

    def test_global_search_covers_full_two_dimensional_range(self):
        reference = pd.DataFrame(
            [
                {"name": "target", "i1": 8.0, "i2": 9.0, "o1": 5.0},
                {"name": "best", "i1": 2.0, "i2": 3.0, "o1": 8.0},
            ]
        )
        target = reference.iloc[0]

        points = global_stratified_candidates(
            reference=reference,
            target_row=target,
            io_cols=["i1", "i2", "o1"],
            inputs=["i1", "i2"],
            outputs=["o1"],
            search_columns=["i1", "i2"],
            pct_above=0,
            samples=900,
            random_state=42,
            name_prefix="test",
        )

        self.assertEqual(len(points), 900)
        x_bins = ((points["i1"] - 2.0) / 6.0 * 30).apply(math.floor)
        y_bins = ((points["i2"] - 3.0) / 6.0 * 30).apply(math.floor)
        self.assertEqual(len(set(zip(x_bins, y_bins))), 900)
        self.assertTrue(points["i1"].between(2.0, 8.0).all())
        self.assertTrue(points["i2"].between(3.0, 9.0).all())


if __name__ == "__main__":
    unittest.main()

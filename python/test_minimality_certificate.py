import unittest

import pandas as pd

from minimality_certificate import dominating_mask, find_selected_path, pareto_front_mask


class MinimalityCertificateTests(unittest.TestCase):
    def test_tc_values_within_tolerance_use_msc_tie_break(self):
        metrics = pd.DataFrame(
            [
                {
                    "path_id": "numerically_lower_tc",
                    "tc": 0.4817597899414578,
                    "msc": 0.35,
                    "attainable_transition_violations": 0,
                },
                {
                    "path_id": "lower_msc",
                    "tc": 0.4817597899414579,
                    "msc": 0.22,
                    "attainable_transition_violations": 0,
                },
            ]
        )

        selected, _ = find_selected_path(metrics, None, tolerance=1e-9)

        self.assertEqual(selected["path_id"], "lower_msc")

    def test_larger_inputs_mean_less_reduction(self):
        pool = pd.DataFrame(
            [
                {"name": "selected", "i1": 5.0, "i2": 4.0},
                {"name": "cheaper", "i1": 5.5, "i2": 4.0},
                {"name": "tradeoff", "i1": 4.5, "i2": 4.5},
            ]
        )

        mask = dominating_mask(
            pool,
            {"i1": 5.0, "i2": 4.0},
            ["i1", "i2"],
            1e-9,
        )

        self.assertEqual(pool.loc[mask, "name"].tolist(), ["cheaper"])

    def test_tradeoffs_are_both_pareto_minimal(self):
        pool = pd.DataFrame(
            [
                {"name": "left", "i1": 6.0, "i2": 3.0},
                {"name": "right", "i1": 4.0, "i2": 5.0},
            ]
        )

        mask = pareto_front_mask(pool, ["i1", "i2"], 1e-9)

        self.assertEqual(pool.loc[mask, "name"].tolist(), ["left", "right"])

    def test_dominated_point_is_removed_from_front(self):
        pool = pd.DataFrame(
            [
                {"name": "front", "i1": 6.0, "i2": 5.0},
                {"name": "dominated", "i1": 5.0, "i2": 4.0},
            ]
        )

        mask = pareto_front_mask(pool, ["i1", "i2"], 1e-9)

        self.assertEqual(pool.loc[mask, "name"].tolist(), ["front"])


if __name__ == "__main__":
    unittest.main()

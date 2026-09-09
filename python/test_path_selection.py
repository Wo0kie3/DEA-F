import unittest

import pandas as pd

from path_selection import (
    pareto_minimal_mask,
    select_publication_paths,
    tolerance_lexicographic_choice,
)


class PathSelectionTests(unittest.TestCase):
    def test_lexicographic_tolerance_uses_next_metric(self):
        metrics = pd.DataFrame(
            [
                {"path_id": "numerically_first", "md": 0.1, "msc": 0.4},
                {"path_id": "lower_msc", "md": 0.1 + 5e-10, "msc": 0.2},
            ]
        )

        selected = tolerance_lexicographic_choice(metrics, ["md", "msc"])

        self.assertEqual(selected["path_id"], "lower_msc")

    def test_pareto_keeps_tradeoffs_and_removes_dominated_path(self):
        metrics = pd.DataFrame(
            [
                {"path_id": "cheap", "msc": 0.1, "fw": 0.8},
                {"path_id": "stable", "msc": 0.3, "fw": 0.2},
                {"path_id": "dominated", "msc": 0.4, "fw": 0.9},
            ]
        )

        mask = pareto_minimal_mask(metrics, ["msc", "fw"])

        self.assertEqual(metrics.loc[mask, "path_id"].tolist(), ["cheap", "stable"])

    def test_selection_emits_single_lex_weighted_and_pareto_results(self):
        metrics = pd.DataFrame(
            [
                {"path_id": "a", "md": 0.1, "msc": 0.5, "swbp": 0.2, "fw": 0.8},
                {"path_id": "b", "md": 0.2, "msc": 0.2, "swbp": 0.1, "fw": 0.2},
            ]
        )

        selection, pareto = select_publication_paths(
            metrics,
            method_name="best_efficiency_path",
            weights={"md": 0.1, "msc": 0.2, "swbp": 0.2, "fw": 0.5},
        )

        self.assertIn("lexicographic", selection["selection_type"].tolist())
        self.assertIn("normalized_weighted", selection["selection_type"].tolist())
        tc_rows = selection[
            (selection["selection_type"] == "single_metric")
            & (selection["selection_detail"] == "tc")
        ]
        self.assertEqual(len(tc_rows), 0)
        weighted = selection[selection["selection_type"] == "normalized_weighted"].iloc[0]
        self.assertEqual(weighted["path_id"], "b")
        self.assertEqual(set(pareto["path_id"]), {"a", "b"})

    def test_selection_always_emits_minimum_tc_when_available(self):
        metrics = pd.DataFrame(
            [
                {"path_id": "milestone", "md": 0.0, "msc": 0.4, "swbp": 0.2, "fw": 0.3, "tc": 0.8},
                {"path_id": "cheap", "md": 0.2, "msc": 0.2, "swbp": 0.1, "fw": 0.4, "tc": 0.3},
            ]
        )

        selection, _ = select_publication_paths(metrics, method_name="best_efficiency_path")

        minimum_tc = selection[
            (selection["selection_type"] == "single_metric")
            & (selection["selection_detail"] == "tc")
        ].iloc[0]
        self.assertEqual(minimum_tc["path_id"], "cheap")


if __name__ == "__main__":
    unittest.main()

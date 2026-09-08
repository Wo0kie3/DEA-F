import unittest

import pandas as pd

from generate_dea_input_snapshot import build_snapshot_tables, peel_efficiency_fronts


class DeaInputSnapshotTests(unittest.TestCase):
    def test_builds_efficiency_fronts_and_necessary_ranking(self):
        input_frame = pd.DataFrame(
            [
                {"name": "A", "i1": 1.0, "o1": 3.0},
                {"name": "B", "i1": 2.0, "o1": 2.0},
                {"name": "C", "i1": 3.0, "o1": 1.0},
            ]
        )
        relation_rows = []
        necessary_pairs = {("A", "A"), ("B", "B"), ("C", "C"), ("A", "B"), ("A", "C"), ("B", "C")}
        possible_pairs = necessary_pairs | {("B", "A")}
        for source in ["A", "B", "C"]:
            for target in ["A", "B", "C"]:
                relation_rows.append(
                    {
                        "source_dmu": source,
                        "target_dmu": target,
                        "necessary_preferred": (source, target) in necessary_pairs,
                        "possible_preferred": (source, target) in possible_pairs,
                    }
                )

        tables = build_snapshot_tables(
            input_frame=input_frame,
            relations=pd.DataFrame(relation_rows),
            efficiencies=pd.DataFrame(
                [
                    {"name": "A", "best_efficiency": 1.0, "worst_efficiency": 0.9},
                    {"name": "B", "best_efficiency": 0.8, "worst_efficiency": 0.6},
                    {"name": "C", "best_efficiency": 0.5, "worst_efficiency": 0.2},
                ]
            ),
            ranks=pd.DataFrame(
                [
                    {"name": "A", "best_rank": 1, "worst_rank": 1},
                    {"name": "B", "best_rank": 2, "worst_rank": 2},
                    {"name": "C", "best_rank": 3, "worst_rank": 3},
                ]
            ),
            efficiency_fronts=pd.DataFrame(
                [
                    {
                        "name": "A",
                        "efficiency_front": 1,
                        "front_best_efficiency": 1.0,
                        "front_worst_efficiency": 0.9,
                    },
                    {
                        "name": "B",
                        "efficiency_front": 2,
                        "front_best_efficiency": 1.0,
                        "front_worst_efficiency": 0.8,
                    },
                    {
                        "name": "C",
                        "efficiency_front": 3,
                        "front_best_efficiency": 1.0,
                        "front_worst_efficiency": 1.0,
                    },
                ]
            ),
        )

        results = tables["reference_dmu_results"].set_index("name")
        self.assertEqual(tables["necessary_matrix"].loc["A", "C"], 1)
        self.assertEqual(tables["possible_matrix"].loc["B", "A"], 1)
        self.assertAlmostEqual(results.loc["B", "score_width"], 0.2)
        self.assertEqual(results.loc["B", "rank_width"], 0)
        self.assertEqual(results.loc["A", "efficiency_front"], 1)
        self.assertEqual(results.loc["B", "efficiency_front"], 2)
        self.assertEqual(results.loc["C", "efficiency_front"], 3)
        self.assertEqual(results.loc["A", "necessary_rank"], 1)
        self.assertEqual(results.loc["B", "necessary_rank"], 2)
        self.assertEqual(results.loc["C", "necessary_rank"], 3)
        self.assertEqual(results.loc["A", "necessary_over_refs"], "B|C")
        self.assertEqual(results.loc["C", "necessary_under_refs"], "A|B")
        self.assertEqual(len(tables["necessary_cover_edges"]), 2)

    def test_peels_efficiency_fronts_by_recomputing_after_removal(self):
        input_frame = pd.DataFrame(
            [
                {"name": "A", "i1": 1.0, "o1": 3.0},
                {"name": "B", "i1": 2.0, "o1": 2.0},
                {"name": "C", "i1": 3.0, "o1": 1.0},
            ]
        )
        calls = []

        def evaluator(remaining, front):
            names = remaining["name"].tolist()
            calls.append((front, names))
            efficient_name = names[0]
            return pd.DataFrame(
                [
                    {
                        "name": name,
                        "best_efficiency": 1.0 if name == efficient_name else 0.5,
                        "worst_efficiency": 0.8 if name == efficient_name else 0.2,
                    }
                    for name in names
                ]
            )

        fronts = peel_efficiency_fronts(input_frame, evaluator).set_index("name")

        self.assertEqual(calls, [(1, ["A", "B", "C"]), (2, ["B", "C"]), (3, ["C"])])
        self.assertEqual(fronts["efficiency_front"].to_dict(), {"A": 1, "B": 2, "C": 3})


if __name__ == "__main__":
    unittest.main()

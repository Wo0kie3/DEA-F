import unittest

import pandas as pd

from path_metrics import summarize_paths


class PathMetricsTests(unittest.TestCase):
    def setUp(self):
        self.ranges = {"i1": 10.0, "i2": 10.0}

    def test_md_requires_every_recommended_stage_gap(self):
        paths = pd.DataFrame(
            [
                {
                    "path_id": "incomplete_gap",
                    "path_length": 2,
                    "stage_00_name": "start",
                    "stage_00_i1": 10.0,
                    "stage_00_i2": 10.0,
                    "stage_01_name": "middle",
                    "stage_01_i1": 9.0,
                    "stage_01_i2": 9.0,
                    "stage_01_milestone_gap": 0.1,
                    "stage_02_name": "final",
                    "stage_02_i1": 8.0,
                    "stage_02_i2": 8.0,
                }
            ]
        )

        metrics = summarize_paths(paths, normalization_ranges=self.ranges)

        self.assertTrue(pd.isna(metrics.iloc[0].get("md")))

    def test_start_summary_requires_complete_robust_series(self):
        paths = pd.DataFrame(
            [
                {
                    "path_id": "missing_start",
                    "path_length": 1,
                    "stage_00_name": "start",
                    "stage_00_i1": 10.0,
                    "stage_01_name": "final",
                    "stage_01_i1": 9.0,
                    "stage_01_best_rank": 4,
                }
            ]
        )

        metrics = summarize_paths(paths, normalization_ranges={"i1": 10.0})

        self.assertTrue(pd.isna(metrics.iloc[0].get("best_rank_start")))

    def test_path_kind_excludes_real_start(self):
        paths = pd.DataFrame(
            [
                {
                    "path_id": "fictive_path",
                    "path_length": 2,
                    "stage_00_name": "start",
                    "stage_00_state_type": "real",
                    "stage_00_i1": 10.0,
                    "stage_01_name": "f1",
                    "stage_01_state_type": "fictive",
                    "stage_01_i1": 9.0,
                    "stage_02_name": "f2",
                    "stage_02_state_type": "fictive",
                    "stage_02_i1": 8.0,
                }
            ]
        )

        metrics = summarize_paths(paths, normalization_ranges={"i1": 10.0})

        self.assertEqual(metrics.iloc[0]["path_kind"], "fictive")
        self.assertFalse(bool(metrics.iloc[0]["mixed_state_path"]))
        self.assertEqual(metrics.iloc[0]["rr"], 0.0)

    def test_peer_continuity_is_not_substituted_with_relation_continuity(self):
        paths = pd.DataFrame(
            [
                {
                    "path_id": "relation_only",
                    "path_length": 1,
                    "stage_00_name": "start",
                    "stage_00_i1": 10.0,
                    "stage_00_candidate_necessary_over_refs": "A|B",
                    "stage_01_name": "final",
                    "stage_01_i1": 9.0,
                    "stage_01_candidate_necessary_over_refs": "B|C",
                }
            ]
        )

        metrics = summarize_paths(paths, normalization_ranges={"i1": 10.0})

        self.assertTrue(pd.isna(metrics.iloc[0].get("pc")))
        self.assertAlmostEqual(metrics.iloc[0]["rc"], 2.0 / 3.0)
        self.assertEqual(metrics.iloc[0]["rc_pair_count"], 1)

    def test_generic_progress_and_width_follow_method_semantics(self):
        paths = pd.DataFrame(
            [
                {
                    "path_id": "rank",
                    "path_length": 2,
                    "stage_00_name": "start",
                    "stage_00_i1": 10.0,
                    "stage_00_best_efficiency": 0.2,
                    "stage_00_best_rank": 10,
                    "stage_00_score_width": 0.2,
                    "stage_00_rank_width": 8,
                    "stage_01_name": "middle",
                    "stage_01_i1": 9.0,
                    "stage_01_best_efficiency": 0.8,
                    "stage_01_best_rank": 9,
                    "stage_01_score_width": 0.7,
                    "stage_01_rank_width": 5,
                    "stage_02_name": "final",
                    "stage_02_i1": 7.0,
                    "stage_02_best_efficiency": 0.9,
                    "stage_02_best_rank": 5,
                    "stage_02_score_width": 0.8,
                    "stage_02_rank_width": 2,
                }
            ]
        )

        metrics = summarize_paths(
            paths,
            method_name="best_rank_path",
            normalization_ranges={"i1": 10.0},
        )

        self.assertEqual(metrics.iloc[0]["fw"], 2)
        self.assertEqual(metrics.iloc[0]["pym"], metrics.iloc[0]["pym_best_rank"])

    def test_weights_create_distinct_weighted_metrics(self):
        paths = pd.DataFrame(
            [
                {
                    "path_id": "weighted",
                    "path_length": 3,
                    "stage_00_name": "start",
                    "stage_00_i1": 10.0,
                    "stage_00_i2": 10.0,
                    "stage_00_score_width": 1.0,
                    "stage_01_name": "one",
                    "stage_01_i1": 9.0,
                    "stage_01_i2": 8.0,
                    "stage_01_score_width": 0.8,
                    "stage_02_name": "two",
                    "stage_02_i1": 7.0,
                    "stage_02_i2": 7.5,
                    "stage_02_score_width": 0.4,
                    "stage_03_name": "three",
                    "stage_03_i1": 6.0,
                    "stage_03_i2": 6.0,
                    "stage_03_score_width": 0.1,
                }
            ]
        )

        metrics = summarize_paths(
            paths,
            normalization_ranges=self.ranges,
            factor_weights={"i1": 0.8, "i2": 0.2},
            stage_weights=[0.6, 0.3, 0.1],
            width_state_weights=[0.1, 0.2, 0.3, 0.4],
        )

        self.assertNotAlmostEqual(metrics.iloc[0]["bp"], metrics.iloc[0]["wbp"])
        self.assertNotAlmostEqual(metrics.iloc[0]["sbp"], metrics.iloc[0]["swbp"])
        self.assertNotAlmostEqual(metrics.iloc[0]["apw"], metrics.iloc[0]["ww"])

    def test_cdir_is_shared_pool_minimum_and_endpoint_cost_is_preserved(self):
        paths = pd.DataFrame(
            [
                {
                    "path_id": "far",
                    "path_length": 1,
                    "stage_00_name": "start",
                    "stage_00_i1": 10.0,
                    "stage_01_name": "far_final",
                    "stage_01_i1": 7.0,
                },
                {
                    "path_id": "near",
                    "path_length": 1,
                    "stage_00_name": "start",
                    "stage_00_i1": 10.0,
                    "stage_01_name": "near_final",
                    "stage_01_i1": 8.0,
                },
            ]
        )

        metrics = summarize_paths(paths, normalization_ranges={"i1": 10.0})

        self.assertEqual(metrics["cdir_scope"].unique().tolist(), ["complete_path_endpoints"])
        self.assertAlmostEqual(metrics.iloc[0]["cdir_endpoint"], 0.3)
        self.assertAlmostEqual(metrics.iloc[0]["cdir"], 0.2)
        self.assertAlmostEqual(metrics.iloc[0]["dr"], 1.5)


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from path_metrics import summarize_paths
from path_pipeline_common import (
    add_effort_columns,
    enumerate_state_paths,
    filter_attainable_named_paths,
    filter_states_to_modified_columns,
    normalization_ranges_from_frame,
    select_transition_candidates,
    state_paths_to_frame,
)


class TransitionCandidateTests(unittest.TestCase):
    def setUp(self):
        self.ranges = {"i1": 10.0, "i2": 10.0}

    def test_candidate_is_compared_with_actual_predecessor(self):
        previous = pd.Series({"name": "stage_1", "i1": 7.0, "i2": 7.0})
        candidates = pd.DataFrame(
            [
                {"name": "globally_cheaper_but_unattainable", "i1": 8.0, "i2": 8.0},
                {"name": "attainable_next_step", "i1": 6.0, "i2": 6.0},
            ]
        )

        selected = select_transition_candidates(
            previous=previous,
            candidates=candidates,
            inputs=["i1", "i2"],
            outputs=[],
            normalization_ranges=self.ranges,
        )

        self.assertEqual(selected["name"].tolist(), ["attainable_next_step"])
        self.assertEqual(
            selected["transition_reference_name"].tolist(),
            ["stage_1"],
        )
        self.assertAlmostEqual(selected.iloc[0]["effort_from_previous"], 0.1)

    def test_transition_front_is_recomputed_for_each_predecessor(self):
        candidates = pd.DataFrame(
            [
                {"name": "right", "i1": 6.0, "i2": 4.0},
                {"name": "left", "i1": 4.0, "i2": 6.0},
                {"name": "deep", "i1": 3.0, "i2": 3.0},
            ]
        )

        from_right = select_transition_candidates(
            previous=pd.Series({"name": "right_parent", "i1": 7.0, "i2": 5.0}),
            candidates=candidates,
            inputs=["i1", "i2"],
            outputs=[],
            normalization_ranges=self.ranges,
        )
        from_left = select_transition_candidates(
            previous=pd.Series({"name": "left_parent", "i1": 5.0, "i2": 7.0}),
            candidates=candidates,
            inputs=["i1", "i2"],
            outputs=[],
            normalization_ranges=self.ranges,
        )

        self.assertEqual(from_right["name"].tolist(), ["right"])
        self.assertEqual(from_left["name"].tolist(), ["left"])

    def test_path_stores_incremental_not_cumulative_effort(self):
        start = pd.Series({"name": "start", "i1": 10.0, "i2": 10.0})
        stages = [
            pd.DataFrame([{"name": "stage_1", "i1": 8.0, "i2": 8.0}]),
            pd.DataFrame([{"name": "stage_2", "i1": 7.0, "i2": 7.0}]),
        ]

        paths = enumerate_state_paths(
            start_row=start,
            stage_candidates=stages,
            inputs=["i1", "i2"],
            outputs=[],
            max_paths=None,
            normalization_ranges=self.ranges,
        )
        frame = state_paths_to_frame(paths, ["i1", "i2"])

        self.assertEqual(len(paths), 1)
        self.assertAlmostEqual(frame.iloc[0]["stage_01_effort_from_previous"], 0.2)
        self.assertAlmostEqual(frame.iloc[0]["stage_02_effort_from_previous"], 0.1)

    def test_normalization_uses_reference_data_range(self):
        reference = pd.DataFrame(
            [
                {"i1": 2.0, "i2": 3.0},
                {"i1": 8.0, "i2": 9.0},
            ]
        )

        ranges = normalization_ranges_from_frame(reference, ["i1", "i2"])

        self.assertEqual(ranges, {"i1": 6.0, "i2": 6.0})

    def test_tc_sums_consecutive_stage_efforts(self):
        paths = pd.DataFrame(
            [
                {
                    "path_id": "path",
                    "path_length": 2,
                    "stage_00_name": "start",
                    "stage_00_i1": 10.0,
                    "stage_00_i2": 10.0,
                    "stage_01_name": "middle",
                    "stage_01_i1": 8.0,
                    "stage_01_i2": 8.0,
                    "stage_02_name": "final",
                    "stage_02_i1": 7.0,
                    "stage_02_i2": 7.0,
                }
            ]
        )

        metrics = summarize_paths(
            paths,
            io_columns=["i1", "i2"],
            normalization_ranges=self.ranges,
        )

        self.assertAlmostEqual(metrics.iloc[0]["tc"], 0.3)
        self.assertAlmostEqual(metrics.iloc[0]["msc"], 0.2)
        self.assertAlmostEqual(metrics.iloc[0]["cdir"], 0.3)
        self.assertAlmostEqual(metrics.iloc[0]["dr"], 1.0)

    def test_cost_uses_only_adjustable_factors_and_same_convention(self):
        start = pd.Series({"name": "start", "i1": 10.0, "i2": 10.0, "o1": 4.0})
        candidate = pd.DataFrame(
            [{"name": "candidate", "i1": 8.0, "i2": 9.0, "o1": 4.0}]
        )
        ranges = {"i1": 10.0, "i2": 10.0, "o1": 10.0}

        with_start_effort = add_effort_columns(
            candidate,
            start,
            ["i1", "i2"],
            normalization_ranges=ranges,
        )
        selected = select_transition_candidates(
            previous=start,
            candidates=with_start_effort,
            inputs=["i1", "i2"],
            outputs=["o1"],
            effort_inputs=["i1", "i2"],
            effort_outputs=[],
            normalization_ranges=ranges,
        )

        self.assertAlmostEqual(with_start_effort.iloc[0]["effort_from_start"], 0.15)
        self.assertAlmostEqual(selected.iloc[0]["effort_from_previous"], 0.15)

    def test_factor_weights_and_stage_budgets_are_applied_before_front(self):
        previous = pd.Series({"name": "start", "i1": 10.0, "i2": 10.0})
        candidates = pd.DataFrame(
            [
                {"name": "i1_heavy", "i1": 8.0, "i2": 10.0},
                {"name": "balanced", "i1": 9.0, "i2": 9.0},
            ]
        )

        selected = select_transition_candidates(
            previous=previous,
            candidates=candidates,
            inputs=["i1", "i2"],
            outputs=[],
            normalization_ranges=self.ranges,
            factor_weights={"i1": 0.75, "i2": 0.25},
            max_stage_effort=0.11,
            max_factor_changes={"i1": 0.15},
        )

        self.assertEqual(selected["name"].tolist(), ["balanced"])
        self.assertAlmostEqual(selected.iloc[0]["effort_from_previous"], 0.1)

    def test_real_states_respect_fixed_columns_and_stepwise_attainability(self):
        target = pd.Series({"name": "start", "i1": 10.0, "o1": 5.0})
        states = pd.DataFrame(
            [
                target.to_dict(),
                {"name": "fixed_output", "i1": 9.0, "o1": 5.0},
                {"name": "changed_output", "i1": 9.0, "o1": 6.0},
                {"name": "backward", "i1": 11.0, "o1": 5.0},
            ]
        )

        fixed = filter_states_to_modified_columns(
            states,
            target,
            ["i1", "o1"],
            ["i1"],
        )
        paths = filter_attainable_named_paths(
            [["start", "fixed_output"], ["start", "backward"]],
            fixed,
            ["i1"],
            ["o1"],
        )

        self.assertEqual(fixed["name"].tolist(), ["start", "fixed_output", "backward"])
        self.assertEqual(paths, [["start", "fixed_output"]])


if __name__ == "__main__":
    unittest.main()

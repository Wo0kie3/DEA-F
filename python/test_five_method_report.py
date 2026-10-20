import unittest

import pandas as pd

from generate_five_method_report import hasse_figure, joint_weight_section, transition_front
from path_pipeline_common import state_paths_to_frame


class FiveMethodReportTests(unittest.TestCase):
    def test_joint_weights_are_separate_numeric_columns_and_keep_method_identity(self):
        witnesses = pd.DataFrame([dict(name='end', joint_top=True, joint_score=1.,
                                      joint_input_weights='i1=0.0|i2=1.225371580191|i3=0.003676093293',
                                      joint_output_weights='o1=1.10497237569')])
        result = joint_weight_section(witnesses, {'hasse_path': 'end', 'best_rank_path': 'end'},
                                      ['A', 'B'], ['i1', 'i2', 'i3'], ['o1'])
        tables = pd.read_html(result)
        self.assertEqual(len(tables[0]), 2)
        self.assertEqual(tables[0]['Waga i1'].tolist(), [0., 0.])
        self.assertAlmostEqual(tables[0].iloc[0]['Waga i3'], 0.0036761)
        self.assertEqual(tables[0]['Wspólny cel'].tolist(), ['Spełniony', 'Spełniony'])
        self.assertIn('A, B', result)
        self.assertNotIn('i1=0.0|', result)

    def test_pos_front_does_not_mix_routes_with_the_same_component(self):
        reference = pd.DataFrame([dict(name="a", i1=1, i2=1, o1=1), dict(name="b", i1=10, i2=10, o1=1)])
        previous = pd.Series(dict(name="start", i1=8, i2=8, o1=1))
        pool = pd.DataFrame([
            dict(name="chosen", i1=6, i2=6, o1=1, milestone_component_id="C", milestone_route_id="r1", milestone_target_name="T1", milestone_goal_kind="pairwise_single_target"),
            dict(name="wrong_route", i1=7, i2=7, o1=1, milestone_component_id="C", milestone_route_id="r2", milestone_target_name="T2", milestone_goal_kind="pairwise_single_target")])
        filtered, front = transition_front(pool, previous, pool.iloc[0], "hasse_path", reference, ["i1", "i2"], {})
        self.assertEqual(filtered.name.tolist(), ["chosen"])
        self.assertEqual(front.name.tolist(), ["chosen"])

    def test_milestone_identity_survives_path_export(self):
        states = [pd.Series(dict(name="start", i1=3, i2=3, i3=3)),
                  pd.Series(dict(name="next", i1=2, i2=2, i3=2,
                                 milestone_component_id="C2", milestone_members="A|B", milestone_front=2))]
        row = state_paths_to_frame([states], ["i1", "i2", "i3"]).iloc[0]
        self.assertEqual(row["stage_01_milestone_component_id"], "C2")
        self.assertEqual(row["stage_01_milestone_members"], "A|B")
        self.assertEqual(row["stage_01_milestone_front"], 2)

    def test_front_uses_third_input_and_actual_branch(self):
        reference = pd.DataFrame([dict(name="a", i1=1, i2=1, i3=1, o1=1), dict(name="b", i1=10, i2=10, i3=10, o1=1)])
        previous = pd.Series(dict(name="parent", i1=8, i2=8, i3=8, o1=1))
        candidates = pd.DataFrame([
            dict(name="x", i1=7, i2=7, i3=3, o1=1, milestone_component_id="A"),
            dict(name="y", i1=6, i2=6, i3=7, o1=1, milestone_component_id="A"),
            dict(name="other_goal", i1=8, i2=8, i3=8, o1=1, milestone_component_id="B"),
            dict(name="unattainable", i1=9, i2=8, i3=8, o1=1, milestone_component_id="A"),
        ])
        _, front = transition_front(candidates, previous, candidates.iloc[0], "hasse_path", reference, ["i1", "i2", "i3"], {})
        self.assertEqual(set(front["name"]), {"x", "y"})

    def test_hasse_counts_paths_on_necessary_edges(self):
        snapshot = {"components": [dict(component_id="S", members="start", necessary_rank=3),
                                   dict(component_id="A", members="a", necessary_rank=2),
                                   dict(component_id="B", members="b", necessary_rank=2),
                                   dict(component_id="T", members="top", necessary_rank=1)],
                    "edges": [dict(source_component=s, target_component=t) for s, t in [("S", "A"), ("S", "B"), ("A", "T"), ("B", "T")]]}
        _, count = hasse_figure(snapshot, "start")
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()

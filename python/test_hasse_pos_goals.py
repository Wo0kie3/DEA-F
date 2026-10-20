import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from hasse_pos_goals import build_pos_routes, goal_pools, enumerate_pos_variant
from java_runner import export_possible_goals_with_java, export_candidate_robust_metrics_with_java
from compare_hasse_pos_variants import audit_weights


class PosGoalTests(unittest.TestCase):
    def test_global_maxima_include_unreachable_top(self):
        names = ["S", "U", "T", "OTHER_TOP"]
        matrix = pd.DataFrame(np.eye(4, dtype=int), index=names, columns=names)
        matrix.loc["U", "S"] = matrix.loc["T", "S"] = matrix.loc["T", "U"] = 1
        routes, top, _, _ = build_pos_routes(matrix, "S")
        self.assertEqual(top, ["OTHER_TOP", "T"])
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["intermediate"][0]["target"], "U")

    def test_sink_equivalence_class_is_not_deleted_by_mutual_nec(self):
        names = ["S", "A", "B"]
        matrix = pd.DataFrame([[1, 0, 0], [1, 1, 1], [1, 1, 1]], index=names, columns=names)
        routes, top, _, _ = build_pos_routes(matrix, "S")
        self.assertEqual(top, ["A", "B"])
        self.assertEqual(routes[0]["intermediate"], [])
        at_top, _, _, _ = build_pos_routes(matrix, "A")
        self.assertEqual(at_top[0]["intermediate"], [])

    def test_intermediate_uses_single_pos_not_necessary(self):
        candidates = pd.DataFrame([dict(name="p", possible_over_refs="U", candidate_necessary_over_refs="[]", pairwise_top=False, joint_top=False)])
        route = dict(route_id="r", intermediate=[dict(component="C", target="U")])
        for variant in ("pairwise", "joint"):
            stages = goal_pools(candidates, route, ["T"], variant)
            self.assertEqual(stages[0].name.tolist(), ["p"])
            self.assertEqual(stages[0].milestone_members.iloc[0], "U")
            self.assertTrue(stages[-1].empty)

    def test_both_goals_keep_stepwise_minimality_and_zero_steps(self):
        start = pd.Series(dict(name="s", i1=10., o1=1., state_type="real"))
        candidates = pd.DataFrame([
            dict(name="p", i1=8., o1=1., state_type="fictive", possible_over_refs="U|T|T2", pairwise_top=True, joint_top=False),
            dict(name="q", i1=5., o1=1., state_type="fictive", possible_over_refs="U|T|T2", pairwise_top=True, joint_top=True)])
        routes = [dict(route_id="r", intermediate=[dict(component="C", target="U")])]
        for variant, end, cost in [("pairwise", "p", .2), ("joint", "q", .5)]:
            paths, metrics, pools, log, cap = enumerate_pos_variant(start, candidates, routes, ["T", "T2"], variant,
                ["i1"], ["o1"], ["i1"], dict(i1=10., o1=1.), dict(i1=1.))
            self.assertEqual(paths.stage_01_name.iloc[0], "p")
            self.assertEqual(paths.stage_02_name.iloc[0], end)
            self.assertEqual(paths.stage_02_transition_reference_name.iloc[0], "p")
            self.assertAlmostEqual(metrics.tc.iloc[0], cost)
            self.assertFalse(cap)
            self.assertEqual(pools.loc[pools.stage.eq(2), "milestone_members"].unique().tolist(), ["T|T2"])

    def test_different_route_lengths_do_not_create_phantom_stages(self):
        start = pd.Series(dict(name="s", i1=10., o1=1., state_type="real"))
        candidates = pd.DataFrame([dict(name="p", i1=8., o1=1., state_type="fictive", possible_over_refs="U|T", pairwise_top=True, joint_top=True)])
        routes = [dict(route_id="short", intermediate=[]), dict(route_id="long", intermediate=[dict(component="C", target="U")])]
        paths, metrics, *_ = enumerate_pos_variant(start, candidates, routes, ["T"], "joint", ["i1"], ["o1"], ["i1"], dict(i1=10., o1=1.), dict(i1=1.))
        self.assertEqual(set(metrics.path_length), {1, 2})
        self.assertTrue(np.allclose(metrics.tc, .2))
        self.assertTrue(np.allclose(metrics.cdir, .2))


@unittest.skipUnless(os.environ.get("DEA_JAVA_TESTS") == "1", "Set DEA_JAVA_TESTS=1 for LP integration")
class PossibleGoalJavaTests(unittest.TestCase):
    def test_pairwise_counterexample_and_joint_weight_certificate(self):
        with tempfile.TemporaryDirectory(prefix="pos goals ") as directory:
            root = Path(directory)
            reference = pd.DataFrame([dict(name="A", i1=1., i2=3., o1=1.), dict(name="B", i1=3., i2=1., o1=1.)])
            candidates = pd.DataFrame([dict(name="pairwise_only", i1=2.5, i2=2.5, o1=1.),
                                       dict(name="joint", i1=2., i2=2., o1=1.),
                                       dict(name="neither", i1=4., i2=4., o1=1.)])
            reference.to_csv(root / "reference.csv", index=False)
            candidates.to_csv(root / "candidates.csv", index=False)
            pd.DataFrame(dict(name=["A", "B"])).to_csv(root / "targets.csv", index=False)
            java = Path(__file__).resolve().parents[1] / "java"
            export_possible_goals_with_java(root / "reference.csv", root / "candidates.csv", root / "targets.csv", root / "goals.csv", java)
            results = pd.read_csv(root / "goals.csv").set_index("name")
            self.assertTrue(results.loc["pairwise_only", "pairwise_top"])
            self.assertFalse(results.loc["pairwise_only", "joint_top"])
            self.assertAlmostEqual(results.loc["pairwise_only", "joint_score"], .8)
            self.assertTrue(results.loc["joint", "joint_top"])
            self.assertFalse(results.loc["neither", "pairwise_top"])
            audit = audit_weights(reference, candidates, results.reset_index(), ["A", "B"])
            self.assertTrue((audit.loc[audit.joint_top, "margin"] >= -1e-8).all())
            export_candidate_robust_metrics_with_java(root / "reference.csv", root / "candidates.csv", root / "robust.csv", java, "org.example.CsvCandidateRobustMetricsExporter")
            robust = pd.read_csv(root / "robust.csv").set_index("name")
            self.assertAlmostEqual(robust.loc["pairwise_only", "best_efficiency"], .8)
            self.assertAlmostEqual(robust.loc["joint", "best_efficiency"], 1.)


if __name__ == "__main__":
    unittest.main()

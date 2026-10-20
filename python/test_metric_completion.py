import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from complete_experiment_metrics import enrich_paths
from java_runner import export_peer_sets_with_java
from path_metrics import _add_set_continuity_metrics
from metric_reporting import metric_coverage, metric_status, selected_metric_table


class ContinuityTests(unittest.TestCase):
    def test_report_keeps_every_defined_metric(self):
        metrics = pd.Series(dict(path_id="p", tc=.5))
        path = pd.Series(dict(stage_00_name="start", stage_01_name="end", stage_01_effort_from_previous=.5))
        result = selected_metric_table({"m": (metrics, path, "blue")}, {"m": "Method"})
        self.assertEqual(len(result), 20)
        self.assertEqual(result.set_index("metric").loc["pc", "Method"], "missing_required_data")

    def test_zero_cost_is_undefined_not_missing_or_zero(self):
        path = pd.Series(dict(path_id="p", stage_00_name="start", stage_01_name="end", stage_01_effort_from_previous=0.))
        metrics = pd.Series(dict(path_id="p", cdir=0.))
        self.assertEqual(metric_status("pyv", float("nan"), path, metrics), "undefined_zero_step_cost")
        self.assertEqual(metric_status("dr", float("nan"), path, metrics), "undefined_zero_direct_cost")
        coverage = metric_coverage(pd.DataFrame([metrics]), pd.DataFrame([path]))
        self.assertEqual(len(coverage), 20)
        self.assertEqual(coverage.set_index("metric").loc["pc", "status"], "missing_required_data")

    def test_all_steps_required(self):
        summary = {}
        row = pd.Series({"stage_01_peer_refs": "A|B", "stage_02_peer_refs": "B"})
        _add_set_continuity_metrics(summary, row, [0, 1, 2], ["peer_refs"], "pc")
        self.assertNotIn("pc", summary)

    def test_empty_set_roundtrip_and_jaccard(self):
        row = pd.Series({"stage_00_peer_refs": "A|B", "stage_01_peer_refs": "B", "stage_02_peer_refs": "[]"})
        summary = {}
        _add_set_continuity_metrics(summary, row, [0, 1, 2], ["peer_refs"], "pc")
        self.assertAlmostEqual(summary["pc"], .75)
        self.assertEqual(summary["pc_pair_count"], 2)

    def test_enrichment_includes_start_and_known_empty_relations(self):
        reference = pd.DataFrame([dict(name="A", i1=2, o1=1)])
        paths = pd.DataFrame([dict(path_id="p", stage_00_name="A", stage_00_i1=2, stage_00_o1=1,
                                  stage_01_name="f", stage_01_i1=1, stage_01_o1=1)])
        pool = pd.DataFrame([dict(name="f", i1=1, o1=1, candidate_necessary_over_refs="A", candidate_possible_over_refs="A",
                                 reference_necessary_over_candidate_refs="", reference_possible_over_candidate_refs="")])
        matrix = pd.DataFrame([[1]], index=["A"], columns=["A"])
        peers = {(2., 1.): dict(peer_refs="A", peer_theta=1), (1., 1.): dict(peer_refs="A", peer_theta=2)}
        result = enrich_paths(paths, pool, reference, dict(necessary=matrix, possible=matrix), peers, ["i1", "o1"])
        self.assertEqual(result.iloc[0].stage_00_candidate_necessary_over_refs, "A")
        self.assertEqual(result.iloc[0].stage_01_reference_necessary_over_candidate_refs, "[]")
        self.assertEqual(result.iloc[0].stage_01_peer_theta, 2)


@unittest.skipUnless(os.environ.get("DEA_JAVA_TESTS") == "1", "Set DEA_JAVA_TESTS=1 for Java LP integration")
class PeerJavaTests(unittest.TestCase):
    def test_union_of_alternative_optima_and_fixed_reference(self):
        with tempfile.TemporaryDirectory(prefix="dea peer test ") as directory:
            root = Path(directory)
            pd.DataFrame([dict(name="A", i1=1, i2=2, o1=1), dict(name="B", i1=2, i2=1, o1=1),
                          dict(name="C", i1=1.5, i2=1.5, o1=1), dict(name="D", i1=4, i2=4, o1=1)]).to_csv(root / "reference.csv", index=False)
            pd.DataFrame([dict(name="inefficient", i1=3, i2=3, o1=1),
                          dict(name="unique", i1=2, i2=4, o1=1),
                          dict(name="beyond", i1=.5, i2=1, o1=1)]).to_csv(root / "states.csv", index=False)
            export_peer_sets_with_java(root / "reference.csv", root / "states.csv", root / "peers.csv",
                                      Path(__file__).resolve().parents[1] / "java")
            results = pd.read_csv(root / "peers.csv").set_index("name")
            self.assertEqual(set(results.loc["inefficient", "peer_refs"].split("|")), {"A", "B", "C"})
            self.assertAlmostEqual(results.loc["inefficient", "peer_theta"], .5)
            self.assertEqual(results.loc["unique", "peer_refs"], "A")
            self.assertAlmostEqual(results.loc["beyond", "peer_theta"], 2)
            self.assertEqual(results.loc["beyond", "peer_refs"], "A")


if __name__ == "__main__":
    unittest.main()

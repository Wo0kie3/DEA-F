import json
from pathlib import Path
import shutil
import subprocess
import unittest

import pandas as pd

from metric_calculation_details import explain_metrics, render_metric_table
from path_metrics import summarize_paths


class CalculationDetailsTests(unittest.TestCase):
    def fixture(self, method="best_efficiency_path", zero_step=False):
        states = [dict(i1=10., i2=10., o1=1., best_efficiency=.3, worst_efficiency=.1, best_rank=5, worst_rank=8),
                  dict(i1=8., i2=9., o1=1., best_efficiency=.5, worst_efficiency=.2, best_rank=3, worst_rank=6),
                  dict(i1=6., i2=6., o1=1., best_efficiency=.9, worst_efficiency=.4, best_rank=1, worst_rank=4)]
        if zero_step:
            states[1] = states[0].copy()
        row = dict(path_id="path-test", path_length=2)
        for h, state in enumerate(states):
            state.update(name=f"s{h}", state_type="real" if h < 2 else "fictive", peer_refs="A|B" if h == 0 else "B",
                         candidate_necessary_over_refs="[]" if h == 0 else "A", milestone_gap=0,
                         milestone_target=state["best_rank" if method == "best_rank_path" else "best_efficiency"],
                         score_width=state["best_efficiency"]-state["worst_efficiency"], rank_width=state["worst_rank"]-state["best_rank"])
            row.update({f"stage_{h:02d}_{k}": v for k, v in state.items()})
        path = pd.Series(row)
        ranges = dict(i1=10., i2=10., o1=1.)
        metrics = summarize_paths(pd.DataFrame([path]), method_name=method, normalization_ranges=ranges,
            adjustable_columns=["i1", "i2"], factor_weights=dict(i1=3., i2=1.), stage_weights=[1, 2],
            width_state_weights=[1, 2, 3], direct_target_cost=.2, cdir_scope="evaluated_final_stage_pool").iloc[0]
        params = dict(dimensions="i1,i2", factor_weights="i1=3,i2=1", stage_weights="1,2", width_state_weights="1,2,3", normalization_ranges=json.dumps(ranges))
        pool = pd.DataFrame([dict(name="other-goal", direct_cost=.2, i1=8., i2=8.)])
        return metrics, path, params, pool, pd.DataFrame()

    def test_twenty_formulas_with_nonuniform_weights_match_calculation(self):
        for method in ("best_efficiency_path", "best_rank_path"):
            args = self.fixture(method)
            details = explain_metrics(*args)
            self.assertEqual(len(details), 20)
            for key, detail in details.items():
                self.assertAlmostEqual(detail["computed_value"], args[0][key])
                self.assertTrue(detail["steps"])
            self.assertNotEqual(details["cdir"]["computed_value"], details["cdir_endpoint"]["computed_value"])
            self.assertIn("best_rank" if method == "best_rank_path" else "best_efficiency", details["pyv"]["steps"][0])

    def test_incorrect_saved_metric_is_rejected(self):
        metrics, *rest = self.fixture()
        metrics["tc"] = 100.
        with self.assertRaisesRegex(ValueError, "disagrees"):
            explain_metrics(metrics, *rest)

    def test_zero_cost_explains_undefined_yield(self):
        details = explain_metrics(*self.fixture(zero_step=True))
        self.assertIsNone(details["pyv"]["computed_value"])
        self.assertIsNone(details["pym"]["computed_value"])

    def test_html_is_escaped_and_every_value_has_a_template(self):
        metrics, path, *rest = self.fixture()
        path["path_id"] = '<script>alert("x")</script>'
        details = explain_metrics(metrics, path, *rest)
        result = render_metric_table({"m": (metrics, path, "blue")}, {"m": "<b>Method</b>"}, {"m": details})
        self.assertEqual(result.count('<template id='), 20)
        self.assertEqual(result.count('class="metric-value"'), 20)
        self.assertNotIn('<script>', result)
        self.assertIn('&lt;script&gt;', result)
        self.assertIn('aria-haspopup="dialog"', result)

    @unittest.skipUnless(shutil.which("node"), "Node required for popup event tests")
    def test_popup_events(self):
        result = subprocess.run(["node", str(Path(__file__).with_name("test_metric_calculations_ui.cjs"))], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

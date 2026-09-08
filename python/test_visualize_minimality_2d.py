import tempfile
import unittest
from pathlib import Path

import pandas as pd

from visualize_minimality_2d import load_evaluated_pool


class VisualizeMinimality2DTests(unittest.TestCase):
    def test_evaluated_pool_includes_probes_and_records_their_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            pd.DataFrame([{"name": "grid", "best_efficiency": 0.4}]).to_csv(
                run_dir / "fictive_candidate_metrics.csv",
                index=False,
            )
            pd.DataFrame([{"name": "global", "best_efficiency": 0.6}]).to_csv(
                run_dir / "global_search_metrics.csv",
                index=False,
            )
            pd.DataFrame([{"name": "probe", "best_efficiency": 0.7}]).to_csv(
                run_dir / "stage_01_eff_refine_iter_01_metrics.csv",
                index=False,
            )
            pd.DataFrame([{"name": "boundary", "best_efficiency": 0.8}]).to_csv(
                run_dir / "stage_01_eff_refined_final_metrics.csv",
                index=False,
            )
            pd.DataFrame([{"name": "local", "best_efficiency": 0.9}]).to_csv(
                run_dir / "stage_01_eff_local_search_metrics.csv",
                index=False,
            )

            pool = load_evaluated_pool(run_dir, stage=1).set_index("name")

            self.assertEqual(len(pool), 5)
            self.assertEqual(pool.loc["grid", "sampling_source"], "siatka bazowa")
            self.assertEqual(pool.loc["global", "sampling_source"], "próbkowanie globalne")
            self.assertEqual(pool.loc["probe", "sampling_source"], "iteracja refinementu")
            self.assertEqual(
                pool.loc["boundary", "sampling_source"],
                "granica po refinemencie",
            )
            self.assertEqual(pool.loc["local", "sampling_source"], "przeszukiwanie lokalne")


if __name__ == "__main__":
    unittest.main()

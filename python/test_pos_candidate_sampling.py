import unittest

import numpy as np
import pandas as pd

from pos_candidate_sampling import expand_grid
from compare_hasse_pos_variants import evaluate_batches
from generate_five_method_report import evaluated_pool
from pathlib import Path
from tempfile import TemporaryDirectory


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.source = pd.DataFrame(dict(name=["a", "b"], state_type=["fictive"]*2,
            i1=[1., 3.], i2=[2., 6.], o1=[.9, .9]))

    def test_reproducible_bounds_centers_and_fixed_output(self):
        result = expand_grid(self.source, ["i1", "i2"], 5, 3, .2, 17)
        pd.testing.assert_frame_equal(result, expand_grid(self.source, ["i1", "i2"], 5, 3, .2, 17))
        self.assertEqual(len(result), 100)
        self.assertTrue(result.o1.eq(.9).all())
        self.assertTrue(result.i1.between(1, 3).all() and result.i2.between(2, 6).all())
        indexed = result.set_index("name")
        for _, row in result[result.sampling_source.eq("jitter")].iterrows():
            center = indexed.loc[row.sampling_center]
            self.assertLessEqual(abs(row.i1-center.i1), .1+1e-12)
            self.assertLessEqual(abs(row.i2-center.i2), .2+1e-12)
        self.assertEqual(result.name.nunique(), len(result))
        pd.testing.assert_frame_equal(result.iloc[:2][self.source.columns], self.source)
        other = expand_grid(self.source, ["i1", "i2"], 5, 3, .2, 18)
        self.assertFalse(np.allclose(result.i1, other.i1))

    def test_invalid_options(self):
        for args in [(1, 2, .2), (3, -1, .2), (3, 2, 0), (3, 2, 2)]:
            with self.assertRaises(ValueError):
                expand_grid(self.source, ["i1", "i2"], *args)

    def test_batch_assembly(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)/"results.csv"
            calls = []
            def fake(src, dst):
                frame = pd.read_csv(src)
                calls.append(len(frame))
                frame.to_csv(dst, index=False)
                frame[["name"]].to_csv(str(dst)+".witnesses.csv", index=False)
            evaluate_batches(self.source, output, fake, batch_size=1, witnesses=True)
            self.assertEqual(calls, [1, 1])
            pd.testing.assert_frame_equal(pd.read_csv(output), self.source)
            self.assertEqual(len(pd.read_csv(str(output)+".witnesses.csv")), 2)

    def test_plot_preserves_sampling_source(self):
        with TemporaryDirectory() as folder:
            run = Path(folder)
            frame = self.source.assign(source=["jitter: center_1", None])
            frame.to_csv(run/"fictive_candidate_metrics.csv", index=False)
            result = evaluated_pool(run, 1)
            self.assertEqual(result.source.tolist(), ["jitter: center_1", "fictive_candidate_metrics"])

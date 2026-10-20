import tempfile
import unittest
from pathlib import Path

import pandas as pd

from shared_candidate_pool import load_shared_pool


class SharedPoolTests(unittest.TestCase):
    def test_validate_pool_and_reject_changed_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'dea_input_snapshot').mkdir()
            ref = pd.DataFrame(dict(name=['A','B'], i1=[2.,1.], i2=[3.,2.], o1=[1.,1.]))
            ref.to_csv(root/'dea_input_snapshot/decision_data.csv',index=False)
            pd.DataFrame([('target','A'),('dimensions','i1,i2')],columns=['parameter','value']).to_csv(root/'experiment_params.csv',index=False)
            pool = ref.iloc[[0]].copy().assign(name='candidate', i1=1.5)
            pool.to_csv(root/'evaluated_candidates.csv',index=False)
            loaded, manifest = load_shared_pool(root, ref, ref.iloc[0], ['i1','i2'])
            self.assertEqual(len(loaded),1)
            self.assertEqual(manifest['count'],1)
            for column,value in [('i1',2.1),('o1',1.1)]:
                changed = pool.copy()
                changed[column] = value
                changed.to_csv(root/'evaluated_candidates.csv',index=False)
                with self.assertRaises(ValueError):
                    load_shared_pool(root,ref,ref.iloc[0],['i1','i2'])
            with self.assertRaises(AssertionError):
                load_shared_pool(root,ref.assign(i1=7.),ref.iloc[0],['i1','i2'])

import base64
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np
import pandas as pd

from metric_path_explorer import metric_choices, pack_bundle


class MetricExplorerTests(unittest.TestCase):
    def test_directions_missing_values_and_tolerance_ties(self):
        metrics = pd.DataFrame(dict(path_id=['a','b','c','d'],tc=[2.,1.,1.,4.],msc=[.5,.7,.3,.1],
            pym=[1.,3.,3.-5e-10,np.nan],pyv=[np.nan,np.inf,2.,1.],rr=[0.,0.,0.,0.],pc=[np.nan]*4))
        catalog = pd.DataFrame(dict(metric=['tc','pym','pyv','rr','pc'],
            direction=['lower_better','higher_better','lower_better','higher_better','lower_better'],
            plain_description=['test']*5))
        rows,ties = metric_choices(metrics,catalog)
        selected = rows.set_index('metric')
        self.assertEqual(selected.loc['tc','path_id'],'c')
        self.assertEqual(selected.loc['pym','path_id'],'c')
        self.assertEqual(selected.loc['pym','tie_count'],2)
        self.assertEqual(selected.loc['pyv','path_id'],'d')
        self.assertEqual(selected.loc['pyv','excluded_paths'],2)
        self.assertTrue(selected.loc['rr','constant'])
        self.assertEqual(selected.loc['rr','tie_count'],4)
        self.assertEqual(selected.loc['pc','finite_paths'],0)
        self.assertIsNone(selected.loc['pc','path_id'])
        self.assertEqual(set(ties.loc[ties.metric.eq('pym'),'path_id']),{'b','c'})

    def test_reject_duplicate_path_identity(self):
        with self.assertRaises(ValueError):
            metric_choices(pd.DataFrame(dict(path_id=['p','p'])))

    def test_bundle_round_trip_and_no_nonfinite_json(self):
        bundle = dict(path_id='p',stages_html='<table>test</table>',figure=dict(data=[],layout={}))
        encoded = pack_bundle(bundle)
        self.assertEqual(json.loads(gzip.decompress(base64.b64decode(encoded))),bundle)
        self.assertEqual(encoded,pack_bundle(bundle))
        with self.assertRaises(ValueError):
            pack_bundle(dict(value=float('nan')))

    @unittest.skipUnless(shutil.which('node'), 'Node required for interaction tests')
    def test_interactive_metric_switch(self):
        result = subprocess.run(['node',str(Path(__file__).with_name('test_metric_path_explorer_ui.cjs'))],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

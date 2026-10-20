import unittest
import pandas as pd
from joint_final_goal import filter_joint_final
from path_pipeline_common import enumerate_state_paths


class JointFinalTests(unittest.TestCase):
    def test_filter_before_pareto_and_only_at_last_stage(self):
        start = pd.Series(dict(name='s',i1=10.,o1=1.))
        pool = pd.DataFrame([dict(name='cheap',i1=8.,o1=1.,joint_top=False),
                             dict(name='joint',i1=5.,o1=1.,joint_top=True)])
        paths = enumerate_state_paths(start,[pool,filter_joint_final(pool,['T'])],['i1'],['o1'],
            max_paths=None,
            normalization_ranges={'i1':10.,'o1':1.},minimal_inputs=['i1'],minimal_outputs=[],
            effort_inputs=['i1'],effort_outputs=[])
        self.assertEqual(len(paths),1)
        self.assertEqual([p['name'] for p in paths[0]],['s','cheap','joint'])
        self.assertEqual(paths[0][-1]['transition_reference_name'],'cheap')

    def test_intersection_not_replacement(self):
        # Only rows already satisfying the method milestone are supplied.
        pool = pd.DataFrame(dict(name=['a','b'], joint_top=[False,True], milestone_target=[.95,.95]))
        actual = filter_joint_final(pool, ['T1','T2'])
        self.assertEqual(actual.name.tolist(), ['b'])
        self.assertEqual(actual.milestone_target.tolist(), [.95])
        self.assertEqual(actual.milestone_joint_targets.tolist(), ['T1|T2'])
        self.assertEqual(len(pool), 2)

    def test_empty_is_valid(self):
        pool = pd.DataFrame(dict(name=['a'], joint_top=[False]))
        self.assertTrue(filter_joint_final(pool, ['T']).empty)

    def test_never_treat_string_false_as_true(self):
        with self.assertRaises(ValueError):
            filter_joint_final(pd.DataFrame(dict(name=['a'], joint_top=['False'])), ['T'])
        with self.assertRaises(ValueError):
            filter_joint_final(pd.DataFrame(dict(name=['a'])), ['T'])

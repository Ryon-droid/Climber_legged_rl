"""Checkpoint selection evaluates both tasks and restores the training horizon."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from distill_getup_policy import short_score, selection_score


class ImitationSelectionTests(unittest.TestCase):
    def record(self,mode,success):
        return dict(mode=mode,success=success,acquired=True,leg_error_m=.001,position_m=.02,tilt_rad=.01)

    def test_losing_standing_cannot_replace_a_balanced_candidate(self):
        parent=[self.record('folded',x) for x in (True,True,False)]
        parent += [self.record('standing',True) for _ in range(5)]
        candidate=[self.record('folded',True) for _ in range(3)]
        candidate += [self.record('standing',False) for _ in range(5)]
        self.assertGreater(selection_score(parent),selection_score(candidate))
        for r in candidate:
            r['success']=True
        self.assertGreater(selection_score(candidate),selection_score(parent))

    def test_selection_forces_both_initial_states_and_restores_horizon(self):
        class Env:
            pose_bank=[{'rearward_offset_m':.36},{'rearward_offset_m':.37}]
            max_steps=777
            dt=.01
            modes=[]
            def reset(self,seed,options):
                self.modes.append(options['reset_mode'])
                return np.zeros(36),{}
            def step(self,action):
                if self.max_steps!=2000:raise AssertionError('Wrong evaluation horizon')
                return np.zeros(36),0.,False,True,dict(combined_success=True,getup_acquired=True,
                    getup_seconds=2.,planar_displacement_m=.01,termination_reason=None,
                    virtual_leg_lengths_m=[.135,.135],tilt_rad=.01)
        env=Env();original=env.pose_bank
        policy=SimpleNamespace(predict=lambda obs,deterministic:(np.zeros(6),None))
        result=short_score(env,policy,20.,3)
        self.assertEqual(env.modes,['folded','folded','standing','standing','standing'])
        self.assertEqual(len(result),5)
        self.assertEqual(env.max_steps,777)
        self.assertIs(env.pose_bank,original)


if __name__=='__main__':unittest.main()

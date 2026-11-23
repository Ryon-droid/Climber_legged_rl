"""New recovery quality scoring still protects task outcomes and holdouts."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_spring208_push_balance as trainer
from test_push_balance_selection import result

class DampedProfileTests(unittest.TestCase):
    def setUp(self):
        with patch.object(trainer.ab,'setup'):
            trainer.setup('balance','damped_recovery24',.4)

    def tearDown(self):
        with patch.object(trainer.ab,'setup'):
            trainer.setup('balance','focused_recovery24')

    def test_smoothness_cannot_compensate_for_failed_recovery_or_lost_standing(self):
        baseline=result(); baseline['summary']['exact24']=dict(success=1,count=2)
        for row in baseline['episodes']:
            row['recovery_quality']=dict(tilt_rate_rms_rad_s=1.,action_curvature_rms=.1)
        baseline['episodes'].append(dict(baseline['episodes'][0],spec=dict(group='exact24')))
        smoother=copy.deepcopy(baseline)
        for row in smoother['episodes']:
            row['recovery_quality'].update(tilt_rate_rms_rad_s=.5,action_curvature_rms=.05)
        self.assertGreater(trainer.score(smoother),trainer.score(baseline))
        smoother['summary']['exact24']['success']=0
        self.assertFalse(trainer.eligible(smoother,baseline))
        self.assertLess(trainer.score(smoother),trainer.score(baseline))
        smoother=copy.deepcopy(baseline);smoother['summary']['standing']['success']=0
        self.assertFalse(trainer.eligible(smoother,baseline))

    def test_diagnostic_discovery_selection_and_test_seeds_are_separate(self):
        assessor=trainer.Assessor.__new__(trainer.Assessor)
        assessor.pool=MagicMock();assessor.pool.map.return_value=[]
        seeds=[]
        for split in ('selection','test'):
            with patch.object(trainer,'sha',return_value='unused'):
                assessor.evaluate(Path('unused.zip'),split)
            specs=[job[1] for job in assessor.pool.map.call_args.args[1]]
            self.assertTrue(all(s['measure_recovery_quality'] for s in specs))
            self.assertTrue(all(s['standing_wheel_kv']==.4 for s in specs))
            seeds.append({s['seed'] for s in specs})
        self.assertFalse(seeds[0]&seeds[1])
        self.assertFalse((seeds[0]|seeds[1])&set(range(1610000,1621256)))

if __name__=='__main__':unittest.main()

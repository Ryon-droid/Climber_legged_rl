"""A bounded intermediate-force curriculum, with independent retention checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_spring208_push_balance as trainer
from test_push_balance_selection import result


class FocusedRecovery24Tests(unittest.TestCase):
    def setUp(self):
        with patch.object(trainer.ab, 'setup'):
            trainer.setup('balance', 'focused_recovery24')

    def tearDown(self):
        with patch.object(trainer.ab, 'setup'):
            trainer.setup('balance', 'standing_retention18')

    def test_parent_budget_and_target_reached_before_first_selection(self):
        self.assertEqual(trainer.PARENT.parent.name, 'push_progressive_balance30_20261116')
        self.assertEqual(trainer.PARENT_SHA, 'c8b5da09547f70957031d358ae067d0c00761b7b45b949e26971044683ddd314')
        self.assertEqual((trainer.TRAIN_SEED, trainer.common.SELECT_SEED, trainer.common.TEST_SEED), (1580000,1590000,1600000))
        self.assertEqual((trainer.common.TARGET,trainer.target_group()), (24.,'exact24'))
        self.assertEqual(trainer.force_schedule(), ((0,22.),(6144,24.)))
        self.assertLess(6144*8,trainer.INTERVAL)
        self.assertEqual(trainer.STEPS,294912)
        self.assertEqual(trainer.ab.LEARNING_RATE,3e-6)
        self.assertFalse(trainer.compare_original_controller())
        events=[dict(recovered=True,force_world_N=[24.,0.,0.]),dict(recovered=False,force_world_N=[-24.,0.,0.])]
        self.assertEqual(trainer.failure_sampling_weight(events),6)

    def test_specs_target_ranges_and_holdouts_disjoint(self):
        assessor=trainer.Assessor.__new__(trainer.Assessor)
        assessor.pool=MagicMock();assessor.pool.map.return_value=[]
        sets=[]
        for split in ('selection','test'):
            with patch.object(trainer,'sha',return_value='fake'):
                assessor.evaluate(Path('fake.zip'),split)
            specs=[a[1] for a in assessor.pool.map.call_args.args[1]]
            for group,force in [('exact24',[24.,24.]),('level',[22.,24.]),('exact18',[18.,18.])]:
                rows=[s for s in specs if s['group']==group]
                self.assertEqual(len(rows),24)
                self.assertEqual(len({s['seed'] for s in rows}),24)
                self.assertTrue(all(s['force_N']==force for s in rows))
            self.assertEqual(sum(s['group']=='standing' for s in specs),64 if split=='selection' else 32)
            sets.append({s['seed'] for s in specs})
        self.assertFalse(sets[0]&sets[1])
        self.assertFalse((sets[0]|sets[1])&set(range(1581000,1581256)))

    def test_target_improvement_requires_retention(self):
        baseline=result();baseline['summary']['exact24']=dict(count=2,success=1)
        after=copy.deepcopy(baseline);after['summary']['exact24']['success']=2
        self.assertGreater(trainer.score(after),trainer.score(baseline))
        self.assertTrue(trainer.eligible(after,baseline))
        for group in ('exact18','level','standing','accepted15','nominal_getup','difficult_getup'):
            bad=copy.deepcopy(after);bad['summary'][group]['success']-=1
            self.assertFalse(trainer.eligible(bad,baseline),group)

    def test_spawn_preserves_force_bands_motor_limits_and_cap_transition(self):
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from climber_legged_rl.recovery_force_curriculum import RecoveryForceCurriculum
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/'hard_training_resets.json').write_text(json.dumps(dict(cases=[dict(seed=17,force_N=[24.,24.],push=True)],probability=.5)))
            create=trainer.factory(folder,17,'balance')
            vector=SubprocVecEnv([create],start_method='spawn')
            try:
                vector.reset()
                self.assertEqual(vector.get_attr('push')[0].config['force_N'][1],22.)
                original=vector.get_attr('original')[0]
                self.assertEqual(original['force_N'][1],24.)
                self.assertEqual(original['force_bands'][2]['force_N'],[22.,24.])
                self.assertEqual(original['episode_probability'],.65)
                self.assertEqual(vector.get_attr('torque_limits')[0].tolist(),[20.]*4+[2.]*2)
                self.assertEqual(vector.get_attr('control')[0]['wheel_kv'],.4)
            finally:
                vector.close()
            env=create()
            try:
                wrapper=env
                while not isinstance(wrapper,RecoveryForceCurriculum):wrapper=wrapper.env
                wrapper.env.probability=1.
                for steps,cap in trainer.force_schedule():
                    wrapper.transitions=steps
                    _,info=env.reset()
                    self.assertEqual(info['training_force_cap_N'],cap)
                    self.assertEqual(env.unwrapped.push.config['force_N'],[cap,cap])
                    self.assertEqual(info['training_exact_force_replay'],cap==24.)
            finally:
                env.close()


if __name__=='__main__':unittest.main()

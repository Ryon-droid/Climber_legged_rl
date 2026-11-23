"""Thirty-newton progression protects eighteen-newton ability and physical limits."""
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


class ProgressiveBalance30Tests(unittest.TestCase):
    def tearDown(self):
        with patch.object(trainer.ab, 'setup'):
            trainer.setup('balance', 'standing_retention18')

    def configure(self):
        with patch.object(trainer.ab, 'setup'):
            trainer.setup('balance', 'progressive_balance30')

    def test_parent_budget_force_and_legacy_reset(self):
        self.configure()
        self.assertEqual(trainer.PARENT.parent.name, 'push_directional_recovery18_20261116')
        self.assertEqual(trainer.PARENT_SHA, 'ec7fee00619e4240a2c6d93c1a247ec74f972a47b34deb3a8ad8e2a1478f696e')
        self.assertEqual((trainer.STEPS, trainer.INTERVAL, trainer.MAX_SECONDS), (294912, 73728, 3000))
        self.assertEqual((trainer.TRAIN_SEED, trainer.common.SELECT_SEED, trainer.common.TEST_SEED), (1550000, 1560000, 1570000))
        self.assertEqual(trainer.common.TARGET, 30.)
        self.assertEqual(trainer.ab.LEARNING_RATE, 3e-6)
        self.assertLess(trainer.force_schedule()[-1][0] * 8, trainer.STEPS // 2)
        self.assertFalse(trainer.compare_original_controller())
        with patch.object(trainer.ab, 'setup'):
            trainer.setup('balance', 'standing_retention18')
        self.assertEqual((trainer.common.TARGET, trainer.target_group()), (18., 'exact18'))

    def test_selection_and_holdout_have_separate_target_and_retention_scenarios(self):
        self.configure()
        assessor = trainer.Assessor.__new__(trainer.Assessor)
        assessor.pool = MagicMock()
        assessor.pool.map.return_value = []
        sets = []
        for split, counts in [('selection', dict(level=24, exact30=24, exact18=24, standing=64, accepted15=24)),
                              ('test', dict(level=24, exact30=24, exact18=24, standing=32, accepted15=24))]:
            with patch.object(trainer, 'sha', return_value='fake'):
                assessor.evaluate(Path('fake.zip'), split)
            specs = [a[1] for a in assessor.pool.map.call_args.args[1]]
            for group, count in counts.items():
                group_specs = [s for s in specs if s['group'] == group]
                self.assertEqual(len(group_specs), count)
                self.assertEqual(len({s['seed'] for s in group_specs}), count)
            for s in specs:
                if s['group'] in ('exact30', 'exact18', 'level'):
                    self.assertEqual(s['force_N'], {'exact30': [30.,30.], 'exact18': [18.,18.], 'level': [18.,30.]}[s['group']])
            sets.append({s['seed'] for s in specs})
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse((sets[0] | sets[1]) & set(range(1551000,1551256)))

    def test_target_gain_cannot_trade_away_eighteen_or_standing(self):
        self.configure()
        baseline = result()
        baseline['summary']['exact30'] = dict(success=1, count=2)
        candidate = copy.deepcopy(baseline)
        candidate['summary']['exact30']['success'] += 1
        self.assertGreater(trainer.score(candidate), trainer.score(baseline))
        self.assertTrue(trainer.eligible(candidate, baseline))
        for group in ('exact18', 'standing', 'accepted15', 'nominal_getup', 'difficult_getup', 'level'):
            bad = copy.deepcopy(candidate)
            bad['summary'][group]['success'] -= 1
            self.assertFalse(trainer.eligible(bad, baseline), group)

    def test_spawn_keeps_new_force_schedule_and_motor_limits(self):
        self.configure()
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from climber_legged_rl.recovery_force_curriculum import RecoveryForceCurriculum
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'hard_training_resets.json').write_text(json.dumps(dict(cases=[
                dict(seed=17, force_N=[30., 30.], push=True)], probability=.5)))
            create = trainer.factory(folder, 17, 'balance')
            vector = SubprocVecEnv([create], start_method='spawn')
            try:
                vector.reset()
                self.assertEqual(vector.get_attr('push')[0].config['force_N'][1], 18.)
                self.assertEqual(vector.get_attr('original')[0]['force_N'][1], 30.)
                self.assertEqual(vector.get_attr('original')[0]['episode_probability'], .65)
                self.assertEqual(vector.get_attr('control')[0]['wheel_kv'], .4)
                self.assertEqual(vector.get_attr('torque_limits')[0].tolist(), [20.]*4+[2.]*2)
            finally:
                vector.close()
            env = create()
            try:
                wrapper = env
                while not isinstance(wrapper, RecoveryForceCurriculum):
                    wrapper = wrapper.env
                wrapper.env.probability = 1.
                for transitions, cap in trainer.force_schedule():
                    wrapper.transitions = transitions
                    _, info = env.reset()
                    self.assertEqual(info['training_force_cap_N'], cap)
                    self.assertEqual(env.unwrapped.push.config['force_N'], [cap,cap])
                    self.assertEqual(info['training_exact_force_replay'], cap == 30.)
            finally:
                env.close()


if __name__ == '__main__':
    unittest.main()

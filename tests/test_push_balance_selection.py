"""Stage selection permits strict-position loss, but protects balance/getup."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_spring208_push_balance import eligible, score
import train_spring208_push_balance as trainer


def result():
    summary = {g: dict(success=1, count=2) for g in
               ('exact18', 'level', 'accepted15', 'standing', 'nominal_getup', 'difficult_getup')}
    episodes = [dict(spec=dict(group=g), warnings=0, seconds=20.,
        push_metrics=dict(recovered_push_count=1),
        final_errors=dict(position_m=.01, heading_deg=1., speed_m_s=.01, yaw_rate_rad_s=.01))
        for g in summary]
    return dict(summary=summary, strict_summary=copy.deepcopy(summary), episodes=episodes)


class BalanceSelectionTests(unittest.TestCase):
    def test_standing_retention_expands_selection_without_touching_independent_test(self):
        from unittest.mock import MagicMock
        pool = MagicMock()
        pool.map.return_value = []
        assessor = trainer.Assessor.__new__(trainer.Assessor)
        assessor.pool = pool
        with patch.object(trainer, 'PROFILE', 'standing_retention18'), patch.object(trainer, 'sha', return_value='fake'):
            assessor.evaluate(Path('fake.zip'))
            specs = [a[1] for a in pool.map.call_args.args[1]]
            expected = dict(standing=64, accepted15=24, exact18=24, level=24)
            for group, count in expected.items():
                seeds = [s['seed'] for s in specs if s['group'] == group]
                self.assertEqual(len(seeds), count)
                self.assertEqual(len(set(seeds)), count)
            test_seeds = {s['seed'] for s in trainer.common.specifications(18., 'test')}
            self.assertFalse(test_seeds & {s['seed'] for s in specs})
            assessor.evaluate(Path('fake.zip'), 'test')
            test_specs = [a[1] for a in pool.map.call_args.args[1]]
            self.assertEqual(sum(s['group'] == 'standing' for s in test_specs), 32)

    def test_balanced_profile_rejects_exact_gain_with_mixed_force_regression(self):
        baseline = result()
        after = copy.deepcopy(baseline)
        after['summary']['exact18']['success'] += 1
        after['summary']['level']['success'] -= 1
        with patch.object(trainer, 'PROFILE', 'balanced_recovery18'):
            self.assertGreater(score(after), score(baseline))
            self.assertFalse(eligible(after, baseline))
            after['summary']['level']['success'] += 1
            self.assertTrue(eligible(after, baseline))
            after['summary']['accepted15']['success'] -= 1
            self.assertFalse(eligible(after, baseline))

    def test_expanded_selection_stays_disjoint_from_test(self):
        from unittest.mock import MagicMock
        pool = MagicMock()
        pool.map.return_value = []
        assessor = trainer.Assessor.__new__(trainer.Assessor)
        assessor.pool = pool
        with patch.object(trainer, 'PROFILE', 'balanced_recovery18'), patch.object(trainer, 'sha', return_value='fake'):
            assessor.evaluate(Path('fake.zip'))
        specs = [args[1] for args in pool.map.call_args.args[1]]
        for group in ('exact18', 'level'):
            seeds = [s['seed'] for s in specs if s['group'] == group]
            self.assertEqual(len(seeds), 24)
            self.assertEqual(len(set(seeds)), 24)
        test_seeds = {s['seed'] for s in trainer.common.specifications(18., 'test')}
        self.assertFalse(test_seeds & {s['seed'] for s in specs})

    def test_wheel_profile_keeps_both_force_groups_and_original_controller(self):
        from unittest.mock import MagicMock
        baseline = result()
        original = copy.deepcopy(baseline)
        original['summary']['level']['success'] += 1
        candidate = copy.deepcopy(baseline)
        candidate['summary']['exact18']['success'] += 1
        callback = trainer.Selection(Path('unused'), MagicMock(), None, baseline,
                                     MagicMock(), 0., original_controller=original)
        with patch.object(trainer, 'PROFILE', 'wheel_response18'):
            self.assertTrue(eligible(candidate, baseline))
            self.assertFalse(eligible(candidate, callback.original_controller))

    def test_directional_profile_preserves_force_groups_and_successful_retention(self):
        baseline, candidate = result(), result()
        candidate['summary']['exact18']['success'] += 1
        for profile in ('directional_recovery18', 'short_directional18', 'standing_retention18'):
            with patch.object(trainer, 'PROFILE', profile):
                self.assertTrue(eligible(candidate, baseline))
                for group in ('level', 'accepted15', 'standing', 'nominal_getup', 'difficult_getup'):
                    worse = copy.deepcopy(candidate)
                    worse['summary'][group]['success'] -= 1
                    self.assertFalse(eligible(worse, baseline), group)

    def test_strict_position_loss_does_not_veto_balance_gain(self):
        baseline = result()
        after = copy.deepcopy(baseline)
        after['summary']['exact18']['success'] = 2
        after['strict_summary']['exact18']['success'] = 0
        self.assertTrue(eligible(after, baseline))
        self.assertGreater(score(after), score(baseline))

    def test_displacement_and_heading_are_not_tiebreakers(self):
        baseline = result()
        after = copy.deepcopy(baseline)
        for e in after['episodes']:
            e['final_errors'].update(position_m=2., heading_deg=170.)
        self.assertEqual(score(after), score(baseline))

    def test_warnings_and_lost_balance_or_getup_reject(self):
        baseline = result()
        for group in ('standing', 'nominal_getup', 'difficult_getup'):
            after = copy.deepcopy(baseline)
            after['summary'][group]['success'] = 0
            self.assertFalse(eligible(after, baseline))
        after = copy.deepcopy(baseline)
        after['episodes'][0]['warnings'] = 1
        self.assertFalse(eligible(after, baseline))


if __name__ == '__main__':
    unittest.main()

"""Stage selection permits strict-position loss, but protects balance/getup."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_spring208_push_balance import eligible, score


def result():
    summary = {g: dict(success=1, count=2) for g in
               ('exact18', 'level', 'accepted15', 'standing', 'nominal_getup', 'difficult_getup')}
    episodes = [dict(spec=dict(group=g), warnings=0, seconds=20.,
        push_metrics=dict(recovered_push_count=1),
        final_errors=dict(position_m=.01, heading_deg=1., speed_m_s=.01, yaw_rate_rad_s=.01))
        for g in summary]
    return dict(summary=summary, strict_summary=copy.deepcopy(summary), episodes=episodes)


class BalanceSelectionTests(unittest.TestCase):
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

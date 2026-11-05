import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('distance_curriculum', ROOT / 'scripts/train_distance_curriculum.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DistanceCurriculumTests(unittest.TestCase):
    def groups(self):
        return {prefix + mode: dict(combined_success_fraction=1., episodes=[dict(
            getup_metrics=dict(acquired=True), simulator_warning_counts=[0],
            push_metrics=dict(push_count=int(prefix == 'push_')))] * 20)
            for prefix in ('baseline_', 'push_') for mode in ('folded', 'standing')}

    def test_all_required_groups_must_pass(self):
        groups = self.groups()
        self.assertTrue(module.advancement_check(groups)['passed'])
        groups['push_standing']['combined_success_fraction'] = .75
        self.assertFalse(module.advancement_check(groups)['passed'])

    def test_unpushed_or_missing_group_cannot_qualify(self):
        groups = self.groups()
        groups['push_folded']['episodes'] = [dict(getup_metrics=dict(acquired=True))]
        self.assertFalse(module.advancement_check(groups)['passed'])
        groups = self.groups()
        del groups['baseline_folded']
        self.assertFalse(module.advancement_check(groups)['passed'])

    def test_getup_loss_or_numerical_warning_blocks_advance(self):
        groups = self.groups()
        groups['baseline_folded']['episodes'][0]['getup_metrics']['acquired'] = False
        self.assertFalse(module.advancement_check(groups)['passed'])
        groups = self.groups()
        groups['push_standing']['episodes'][0]['simulator_warning_counts'] = [1]
        self.assertFalse(module.advancement_check(groups)['passed'])

    def test_distance_and_reward_tighten_without_changing_robot(self):
        import yaml
        base = yaml.safe_load((ROOT / 'configs/robot/cad_getup_stand_round19_recovery.yaml').read_text())
        for stage, limit, sigma, weight, transient in [
                ('loose', .20, .12, 2., .50), ('medium', .10, .06, 5., .30), ('tight', .05, .03, 10., .20)]:
            config = yaml.safe_load((ROOT / f'configs/robot/cad_getup_stand_round20_{stage}.yaml').read_text())
            self.assertEqual(config['task']['stationary_max_displacement_m'], limit)
            self.assertEqual(config['task']['position_sigma_m'], sigma)
            self.assertEqual(config['reward']['position_hold'], weight)
            self.assertEqual(config['push']['max_transient_displacement_m'], transient)
            for key in ('control', 'physics', 'mass', 'spring', 'getup'):
                self.assertEqual(config[key], base[key])
            for key in ('linear_tracking_sigma_m_s', 'yaw_tracking_sigma_rad_s', 'stationary_max_heading_rad'):
                self.assertEqual(config['task'][key], base['task'][key])
            self.assertTrue(config['training']['demonstration_replay']['preserve_successful_prone'])


if __name__ == '__main__':
    unittest.main()

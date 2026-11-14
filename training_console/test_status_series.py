"""The dashboard must preserve branch boundaries and missing measurements."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from remote_status import optimizer_series, paired_steps


class TrainingSeriesTests(unittest.TestCase):
    def test_independent_counters_and_sparse_first_row(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            a = folder / 'logs_18_0' / 'progress.csv'
            b = folder / 'logs_18_1' / 'progress.csv'
            contents = {
                a: 'time/total_timesteps,rollout/ep_rew_mean,train/approx_kl\n6144,-5,\n12288,8,0.0002\n',
                b: 'time/total_timesteps,rollout/ep_rew_mean,train/approx_kl\n6144,2,nan\n',
            }
            with patch.object(Path, 'glob', return_value=iter([a, b])), patch.object(Path, 'read_text', lambda path: contents[path]):
                rows = optimizer_series(folder, {'arms': ['standing_only', 'all_success']})
            self.assertEqual([r['arm'] for r in rows], ['standing_only', 'all_success'])
            self.assertEqual([r['latest']['steps'] for r in rows], [12288, 6144])
            self.assertIsNone(rows[0]['points'][0]['kl'])
            self.assertIsNone(rows[1]['latest']['kl'])
            self.assertEqual(rows[0]['points'][0]['reward'], -5)

    def test_final_checkpoint_counts_even_when_csv_lags_one_rollout(self):
        series = [dict(arm='a', latest=dict(steps=239616)), dict(arm='b', latest=dict(steps=12288))]
        curves = [dict(arm='a', steps=245760)]
        self.assertEqual(paired_steps(series, curves, ['a', 'b'], 245760), 258048)

    def test_second_branch_cumulative_checkpoint_is_counted_once(self):
        series = [dict(arm='a', latest=dict(steps=245760)), dict(arm='b', latest=dict(steps=61440))]
        curves = [dict(arm='a', steps=245760), dict(arm='b', steps=307200)]
        self.assertEqual(paired_steps(series, curves, ['a', 'b'], 245760), 307200)


if __name__ == '__main__':
    unittest.main()

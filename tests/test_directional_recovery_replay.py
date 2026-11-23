"""Check training-only replay balance and difficult-event prioritization."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from directional_recovery_replay import build_directional_replay, failure_weight


def event(recovered=False, x=-18.):
    return dict(recovered=recovered, force_world_N=[x, 0., 0.])


def row(seed, success, *, push=True, force=(15., 18.), events=None):
    return dict(spec=dict(seed=seed, push=push, force_N=list(force)), success=success,
                warnings=0, reason='timeout' if success else 'base_contact',
                push_metrics=dict(push_events=events if events is not None else [event()]))


class DirectionalReplayTests(unittest.TestCase):
    def rows(self):
        return [row(1, False), row(2, False, events=[event(True), event(False)]),
                row(3, False, events=[event(True, 18.)]),
                row(4, True, push=False, events=[]),
                row(5, True, force=(18., 18.), events=[event(True)]),
                row(6, True, events=[event(True)]),
                row(7, True, events=[event(True, 18.)])]

    def test_half_failure_half_retention_with_mixed_force_coverage(self):
        cases, metadata = build_directional_replay(self.rows(), 100, (1, 7))
        self.assertEqual(metadata['failure_replay_entries'], 16)
        self.assertEqual(metadata['successful_retention_entries'], 16)
        self.assertEqual(len(cases), 32)
        self.assertEqual(metadata['retention_entries_per_group'],
            dict(standing=2, fixed_negative_x=4, mixed_negative_x=6, mixed_other=4))
        self.assertEqual({c['seed'] for c in cases}, set(range(1, 8)))
        self.assertTrue(all(c['failure_reason'] is None for c in cases if c['successful_retention']))

    def test_weight_uses_maximum_not_product(self):
        self.assertEqual(failure_weight([event(True), event(False)]), 6)
        self.assertEqual(failure_weight([event(True)]), 6)
        self.assertEqual(failure_weight([event(True, 18.)]), 4)
        self.assertEqual(failure_weight([event(False, 18.)]), 1)

    def test_standing_retention_covers_half_success_pool_without_dropping_pushes(self):
        weights = dict(standing=7, fixed_negative_x=2, mixed_negative_x=3, mixed_other=2)
        rows = self.rows()
        before = copy.deepcopy(rows)
        cases, metadata = build_directional_replay(rows, 100, (1, 7), weights)
        self.assertEqual(rows, before)
        self.assertEqual(metadata['failure_replay_entries'], 16)
        self.assertEqual(metadata['successful_retention_entries'], 16)
        self.assertGreaterEqual(metadata['retention_entries_per_group']['standing'], 8)
        self.assertTrue(all(metadata['retention_entries_per_group'][k] > 0 for k in weights))
        self.assertEqual(metadata['retention_sampling_weights'], weights)
        self.assertEqual({c['seed'] for c in cases}, set(range(1, 8)))
        for invalid in (dict(standing=7), dict(weights, standing=0), dict(weights, standing=1.5)):
            with self.assertRaises(ValueError):
                build_directional_replay(rows, 100, (1, 7), invalid)
        with self.assertRaises(ValueError):
            build_directional_replay([rows[0], rows[4]], 100, (1, 7), weights)

    def test_inputs_unchanged_and_entries_independent(self):
        rows = self.rows()
        before = copy.deepcopy(rows)
        cases, metadata = build_directional_replay(rows, 100, (1, 7))
        self.assertEqual(rows, before)
        self.assertEqual((cases, metadata), build_directional_replay(rows, 100, (1, 7)))
        cases[0]['expected_push_events'][0]['force_world_N'][0] = 999.
        self.assertNotEqual(cases[1]['expected_push_events'][0]['force_world_N'][0], 999.)
        self.assertEqual(rows, before)

    def test_reject_nontraining_seeds_duplicates_and_numerical_warnings(self):
        for mutate in (lambda r: r[0]['spec'].update(seed=1000),
                       lambda r: r[0]['spec'].update(seed=2),
                       lambda r: r[0].update(warnings=1)):
            rows = self.rows()
            mutate(rows)
            with self.assertRaises(ValueError):
                build_directional_replay(rows, 100, (1, 7))

    def test_available_groups_fallback_without_dropping_failures(self):
        rows = [row(1, False), row(2, True, push=False, events=[])]
        cases, metadata = build_directional_replay(rows, 100, (1, 2))
        self.assertEqual(metadata['failure_replay_entries'], 6)
        self.assertEqual(metadata['retention_entries_per_group']['standing'], 6)
        self.assertEqual(len(cases), 12)
        for subset in ([rows[0]], [rows[1]]):
            with self.assertRaises(ValueError):
                build_directional_replay(subset, 100, (1, 2))


if __name__ == '__main__':
    unittest.main()

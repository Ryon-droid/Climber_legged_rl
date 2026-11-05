"""Collection windows include recovery tails without changing the legacy default."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from collect_standing_rehearsal import recovery_window


class RehearsalWindowTests(unittest.TestCase):
    def test_initial_window_can_extend_past_old_cutoff(self):
        self.assertFalse(recovery_window(190, .01, None, False))
        self.assertTrue(recovery_window(190, .01, None, False, 4., 4.))
        self.assertFalse(recovery_window(400, .01, None, False, 4., 4.))

    def test_settled_push_kept_until_tail_boundary(self):
        event = dict(end_step=310, recovered=True)
        self.assertFalse(recovery_window(360, .01, event, True))
        self.assertTrue(recovery_window(360, .01, event, True, 4., 4.))
        self.assertTrue(recovery_window(709, .01, event, True, 4., 4.))
        self.assertFalse(recovery_window(710, .01, event, True, 4., 4.))
        self.assertFalse(recovery_window(301, .01, None, True, 4., 4.))

    def test_unrecovered_push_and_legacy_initial_window_retained(self):
        self.assertTrue(recovery_window(350, .01, dict(end_step=310, recovered=False), True))
        self.assertTrue(recovery_window(149, .01, None, False))
        self.assertFalse(recovery_window(150, .01, None, False))

    def test_invalid_durations_rejected(self):
        for initial, tail in [(0., 4.), (4., -1.), (float('nan'), 1.), (1., float('inf'))]:
            with self.subTest(initial=initial, tail=tail), self.assertRaises(ValueError):
                recovery_window(0, .01, None, False, initial, tail)


if __name__ == '__main__':unittest.main()

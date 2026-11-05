"""Per-motor exploration accepts six values while preserving scalar configs."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from climber_legged_rl.mujoco_training import action_std_vector, value_loss_overrides


class MotorExplorationTests(unittest.TestCase):
    def test_value_loss_weight_is_explicit_and_validated(self):
        self.assertEqual(value_loss_overrides({}), {})
        self.assertEqual(value_loss_overrides({'vf_coef': .0001}), {'vf_coef': .0001})
        for value in (-1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'vf_coef'):
                value_loss_overrides({'vf_coef': value})

    def test_scalar_expands_to_all_six_motors(self):
        np.testing.assert_array_equal(action_std_vector(.03),np.full(6,.03))
        self.assertIsNone(action_std_vector(None))

    def test_wheel_exploration_can_differ_from_legs(self):
        expected=np.array([.03]*4+[.15]*2)
        np.testing.assert_array_equal(action_std_vector(expected.tolist()),expected)

    def test_rejects_wrong_shapes_negative_zero_and_nonfinite(self):
        for value in ([.1,.2],[[.1]*6],[-.1]*6,[.1]*5+[0.],float('nan'),[.1]*5+[float('inf')]):
            with self.subTest(value=value),self.assertRaises(ValueError):
                action_std_vector(value)

"""Verify category weights and reject provenance/order mismatches."""
import unittest
import numpy as np
from reweight_standing_rehearsal import reweighted_arrays


class ReweightTests(unittest.TestCase):
    def fixture(self):
        arrays = dict(observations=np.arange(4*180).reshape(4, 180),
                      actions=np.arange(24).reshape(4, 6), weights=np.array([.45,.45,.05,.05]))
        report = dict(successful_data_only=True, runtime_reference_assistance=False,
            category_mass=dict(stable=.9, recovery=.1), outcomes=[
                dict(role='stable', success=True, retained_samples=2),
                dict(role='recovery', success=True, retained_samples=2)])
        return arrays, report

    def test_changes_only_weights(self):
        arrays, report = self.fixture()
        result, _ = reweighted_arrays(arrays, report, .3)
        np.testing.assert_array_equal(result['observations'], arrays['observations'])
        np.testing.assert_array_equal(result['actions'], arrays['actions'])
        np.testing.assert_allclose(result['weights'], [.35,.35,.15,.15])
        np.testing.assert_allclose(arrays['weights'], [.45,.45,.05,.05])

    def test_rejects_failed_retained_data(self):
        arrays, report = self.fixture()
        report['outcomes'][1]['success'] = False
        with self.assertRaises(ValueError):
            reweighted_arrays(arrays, report, .3)

    def test_rejects_row_order_mismatch(self):
        arrays, report = self.fixture()
        arrays['weights'] = arrays['weights'][::-1]
        with self.assertRaises(ValueError):
            reweighted_arrays(arrays, report, .3)


if __name__ == '__main__':
    unittest.main()

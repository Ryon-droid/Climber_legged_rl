import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from round18_takeover_data import digest, load_successful_takeovers


class TakeoverDataTests(unittest.TestCase):
    def fixture(self, root):
        for name in ('config','model','policy'):
            (root/name).write_text(name)
        x=np.zeros((2,180),np.float32); x[:,-1]=1
        np.savez_compressed(root/'data.npz',observations=x,actions=np.zeros((2,6),np.float32))
        outcome=dict(dataset=str(root/'data.npz'),dataset_sha256=digest(root/'data.npz'),
            combined_success=True,takeover_seconds=1.,seconds=20.,reason='timeout',retained_samples=2,
            push_metrics=dict(push_count=2,recovered_push_count=2))
        report=dict(schema='validated_policy_takeovers_v1',full_episode_success_required=True,
            state_reset_at_takeover=False,history_reset_at_takeover=False,physics_assistance=False,
            runtime_controller='baseline_then_recovery_teacher',config_sha256=digest(root/'config'),
            model_sha256=digest(root/'model'),checkpoints=dict(teacher=dict(path=str(root/'policy'),sha256=digest(root/'policy'))),
            outcomes=[outcome])
        return report

    def check(self, change=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); report=self.fixture(root)
            if change: change(report,root)
            (root/'report.json').write_text(json.dumps(report))
            return load_successful_takeovers(root/'report.json',root/'config',root/'model')

    def test_accepts_explicit_switch_success(self):
        rows,_=self.check(); self.assertEqual(len(rows),1)

    def test_rejects_failed_episode(self):
        with self.assertRaises(ValueError):
            self.check(lambda r,p:r['outcomes'][0].update(combined_success=False))

    def test_rejects_unrecovered_push(self):
        with self.assertRaises(ValueError):
            self.check(lambda r,p:r['outcomes'][0]['push_metrics'].update(recovered_push_count=1))

    def test_rejects_reset_history(self):
        with self.assertRaises(ValueError):
            self.check(lambda r,p:r.update(history_reset_at_takeover=True))

    def test_rejects_modified_data(self):
        with self.assertRaises(ValueError):
            self.check(lambda r,p:(p/'data.npz').write_bytes(b'changed'))


if __name__=='__main__': unittest.main()

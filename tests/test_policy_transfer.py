"""Curriculum warm starts may change task/reward, never physics or controls."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ('stable_baselines3','yaml','mujoco'))

@unittest.skipUnless(AVAILABLE,'Training dependencies absent')
class PolicyTransferTests(unittest.TestCase):
    def setUp(self):
        import yaml
        self.temp=tempfile.TemporaryDirectory()
        self.folder=Path(self.temp.name)
        previous=self.folder/'previous'
        (previous/'best').mkdir(parents=True)
        self.checkpoint=previous/'best/best_model.zip'
        self.checkpoint.write_bytes(b'transfer validation fixture')
        self.model=self.folder/'robot_experiment.xml'
        self.model.write_text('<mujoco/>')
        (previous/'robot_experiment.xml').write_bytes(self.model.read_bytes())
        self.config=yaml.safe_load((ROOT/'configs/robot/cad_dynamics.yaml').read_text())
        (previous/'config.yaml').write_text(yaml.safe_dump(self.config))
        from climber_legged_rl.mujoco_training import validate_policy_transfer
        self.validate=validate_policy_transfer

    def tearDown(self):self.temp.cleanup()

    def test_resume_accepts_identical_task(self):
        self.assertEqual(self.validate(self.checkpoint,self.model,self.config),self.checkpoint)

    def test_resume_rejects_task_change(self):
        self.config['task']['episode_seconds']=20
        with self.assertRaisesRegex(ValueError,'task'):
            self.validate(self.checkpoint,self.model,self.config)

    def test_warm_start_accepts_new_task_reward_and_learning_rate(self):
        self.config['task']['episode_seconds']=20
        self.config['reward']['upright']=4
        self.config['training']['learning_rate']=1e-4
        self.assertEqual(self.validate(self.checkpoint,self.model,self.config,warm_start=True),self.checkpoint)

    def test_warm_start_rejects_physics_control_and_spring_changes(self):
        for section,key in [('physics','timestep'),('control','leg_kp'),('spring','force_N')]:
            config=copy.deepcopy(self.config)
            config[section][key]*=2
            with self.assertRaisesRegex(ValueError,section):
                self.validate(self.checkpoint,self.model,config,warm_start=True)

    def test_warm_start_rejects_changed_model_bytes(self):
        self.model.write_text('<mujoco model="different"/>')
        with self.assertRaisesRegex(ValueError,'identical'):
            self.validate(self.checkpoint,self.model,self.config,warm_start=True)

if __name__=='__main__':unittest.main()

"""Force pulses must end, preserve nominal resets, and require post-push settling."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))


class PushTests(unittest.TestCase):
    def setUp(self):
        import yaml
        self.config = yaml.safe_load((ROOT/'configs/robot/cad_dynamics_round6_push.yaml').read_text())
        self.config['push'].update(episode_probability=1.,force_N=[1.,1.],duration_seconds=[.02,.02],
                                   interval_seconds=[1.,1.],first_push_seconds=.01,
                                   recovery_seconds=.2,settle_seconds=.02)

    def test_pulse_duration_impulse_reset_and_seed(self):
        import numpy as np
        from climber_legged_rl.push_disturbance import PushDisturbance
        push=PushDisturbance(self.config['push'],.01)
        sequences=[]
        for _ in range(2):
            push.reset(np.random.default_rng(7))
            forces=np.array([push.force(i,100) for i in range(10)])
            sequences.append(forces)
            np.testing.assert_allclose(np.linalg.norm(forces,axis=1),[0,1,1,0,0,0,0,0,0,0])
            self.assertAlmostEqual(push.metrics()['push_impulse_Ns'],.02)
            self.assertEqual(push.metrics()['push_count'],1)
        np.testing.assert_array_equal(*sequences)

    def test_recovery_needs_continuous_settling_before_deadline(self):
        import numpy as np
        from climber_legged_rl.push_disturbance import PushDisturbance
        push=PushDisturbance(self.config['push'],.01);push.reset(np.random.default_rng(7))
        push.force(1,100)
        push.observe(4,.01,.01,.01,.01,.01,self.config['task'])
        self.assertFalse(push.active['recovered'])
        push.observe(5,.3,.01,.01,.01,.01,self.config['task'])
        push.observe(6,.01,.01,.01,.01,.01,self.config['task'])
        self.assertFalse(push.active['recovered'])
        push.observe(7,.01,.01,.01,.01,.01,self.config['task'])
        self.assertTrue(push.active['recovered'])
        self.assertAlmostEqual(push.active['recovery_seconds'],.04)
        self.assertTrue(push.success(True,False,.1,.1,.01,.01,.01,.01,.01,self.config['task']))
        self.assertFalse(push.success(True,False,.3,.1,.01,.01,.01,.01,.01,self.config['task']))
        self.assertFalse(push.success(True,False,.1,.1,.1,.01,.01,.01,.01,self.config['task']))

    def test_incomplete_late_pulse_is_not_started_and_no_push_is_not_evidence(self):
        import numpy as np
        from climber_legged_rl.push_disturbance import PushDisturbance
        push=PushDisturbance(self.config['push'],.01);push.reset(np.random.default_rng(7))
        np.testing.assert_array_equal(push.force(1,10),0.)
        self.assertEqual(push.metrics()['push_count'],0)
        self.assertFalse(push.success(True,False,0.,0.,0.,0.,0.,0.,0.,self.config['task']))
        push.force(1,100)
        self.assertEqual(push.metrics()['recovered_push_count'],0)

    def test_invalid_force_and_overlapping_schedule_rejected(self):
        from climber_legged_rl.push_disturbance import PushDisturbance
        for patch in [dict(force_N=[-1,2]),dict(interval_seconds=[.1,.2]),dict(episode_probability=2.)]:
            config=dict(self.config['push'],**patch)
            with self.assertRaises(ValueError):PushDisturbance(config,.01)

    def test_environment_applies_world_horizontal_force_then_clears_without_changing_reset(self):
        import yaml
        import numpy as np
        from climber_legged_rl.mujoco_env import ClimberMujocoEnv
        with tempfile.TemporaryDirectory() as temp:
            enabled=Path(temp)/'enabled.yaml';disabled=Path(temp)/'disabled.yaml'
            enabled.write_text(yaml.safe_dump(self.config))
            config=copy.deepcopy(self.config);config.pop('push')
            disabled.write_text(yaml.safe_dump(config))
            model=ROOT/'outputs/cad_dynamics_round5/robot_experiment.xml'
            env=ClimberMujocoEnv(model,enabled);old=ClimberMujocoEnv(model,disabled)
            try:
                a,_=env.reset(seed=9);b,_=old.reset(seed=9)
                np.testing.assert_array_equal(a,b)
                for e in (env,old):
                    e.data.qpos[2]+=1.
                    e.data.qvel[:]=0.
                for step in range(6):
                    _,_,_,_,info=env.step(np.zeros(6));old.step(np.zeros(6))
                    force=env.data.xfrc_applied[env.base_id]
                    self.assertAlmostEqual(np.linalg.norm(force[:2]),1. if step in (1,2) else 0.)
                    np.testing.assert_array_equal(force[2:],0.)
                    np.testing.assert_array_equal(env.data.qfrc_applied[env.spring_dofs],450.)
                self.assertGreater(np.linalg.norm(env.data.qvel[:2]-old.data.qvel[:2]),1e-5)
                env.reset(seed=9)
                np.testing.assert_array_equal(env.data.xfrc_applied,0.)
                self.assertEqual(env.push.metrics()['push_count'],0)
            finally:env.close();old.close()


if __name__=='__main__':unittest.main()

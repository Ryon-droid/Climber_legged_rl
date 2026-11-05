"""Expert labels must respect the student's PD map; fitting only updates the actor."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from climber_legged_rl.getup_imitation import expert_label,fit_actor


class GetupImitationTests(unittest.TestCase):
    def fixture(self,student_kp=80.,teacher_kp=20.):
        action=np.array([.1,-.2,.3,-.4,.5,-.6])
        student=SimpleNamespace(acquired=False,control={'leg_kp':student_kp,'leg_kd':2.},
            dynamic={'incremental_target_scale_rad':.6},q_indices=np.arange(4),v_indices=np.arange(4),
            data=SimpleNamespace(qpos=np.array([.1,.2,.3,.4]),qvel=np.array([.1,-.1,.2,-.2])),
            torque_limits=np.array([40.]*4),_set_motor_envelope=lambda:None)
        teacher=SimpleNamespace(control={'leg_kp':teacher_kp,'leg_kd':1.},torque_limits=np.array([10.]*4),
            _observation=lambda:np.zeros(36),assisted_action=lambda a:a,_set_available_torque=lambda:None,
            _leg_targets=lambda a:np.array([.2,.1,.4,.3]))
        policy=SimpleNamespace(predict=lambda obs,deterministic:(action,None))
        return student,teacher,policy,action

    def test_different_pd_gains_produce_equivalent_initial_torque(self):
        student,teacher,policy,_=self.fixture()
        label,_=expert_label(student,teacher,policy)
        actual=80.*(.6*label[:4])-2.*student.data.qvel
        expected=np.clip(20.*(teacher._leg_targets(None)-student.data.qpos)-student.data.qvel,-10.,10.)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        np.testing.assert_allclose(label[4:],[.5,-.6])

    def test_identical_pd_maps_keep_the_unclipped_joint_target(self):
        student,teacher,policy,_=self.fixture(80.,80.)
        teacher.control['leg_kd']=2.
        teacher.torque_limits[:]=40.
        label,_=expert_label(student,teacher,policy)
        np.testing.assert_allclose(student.data.qpos+.6*label[:4],teacher._leg_targets(None))

    def test_standing_keeps_parent_actor_action_definition(self):
        student,teacher,policy,action=self.fixture()
        student.acquired=True
        label,executed=expert_label(student,teacher,policy)
        np.testing.assert_array_equal(label,action)
        np.testing.assert_array_equal(executed,action)

    def test_supervised_actor_updates_leave_critic_weights_unchanged(self):
        import gymnasium as gym
        import torch
        from stable_baselines3 import PPO
        env=gym.make('Pendulum-v1')
        try:
            torch.manual_seed(1)
            model=PPO('MlpPolicy',env,n_steps=8,batch_size=8,device='cpu')
            before={k:v.clone() for k,v in model.policy.state_dict().items()}
            x=np.zeros((32,3),dtype=np.float32)
            y=np.full((32,1),.25,dtype=np.float32)
            fit_actor(model,x,y,np.ones(32),20,16,1e-3,1)
            after=model.policy.state_dict()
            for k in before:
                if 'value_net' in k:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in after.items() if 'action_net' in k))
        finally:
            env.close()


if __name__=='__main__':unittest.main()

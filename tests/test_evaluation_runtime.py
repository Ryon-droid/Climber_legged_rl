"""Evaluation acceleration must preserve episode order and deterministic actors."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import torch
import yaml
from stable_baselines3 import PPO

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from climber_legged_rl.evaluation_runtime import deterministic_actor_digest,merge_evaluations,ParallelEvaluation


class EvaluationRuntimeTests(unittest.TestCase):
    def test_actor_digest_ignores_critic_and_std_but_detects_action_changes(self):
        torch.set_num_threads(1)
        env=gym.make('Pendulum-v1')
        try:
            model=PPO('MlpPolicy',env,n_steps=8,batch_size=8)
            before=deterministic_actor_digest(model.policy)
            with torch.no_grad():
                model.policy.value_net.weight.add_(1.)
                model.policy.mlp_extractor.value_net[0].weight.add_(1.)
                model.policy.log_std.add_(1.)
            self.assertEqual(before,deterministic_actor_digest(model.policy))
            with torch.no_grad():model.policy.action_net.bias.add_(.001)
            self.assertNotEqual(before,deterministic_actor_digest(model.policy))
        finally:env.close()

    def test_merge_preserves_episode_order_and_handles_unequal_chunks(self):
        parts=[]
        for start,count in [(0,2),(2,1)]:
            records=[]
            for i in range(start,start+count):
                records.append(dict(reward=float(i),seconds=20.,reason='timeout',
                    stationary_success=i!=1,combined_success=i!=1,push_recovery_success=i!=1,
                    getup_metrics=dict(reset_mode='standing'),push_metrics=dict(push_count=int(i!=2))))
            parts.append(dict(seed=100+start,episodes=records,combined_success_fraction=0.,
                push_recovery_success_fraction=0.))
        result=merge_evaluations(parts)
        self.assertEqual([r['reward'] for r in result['episodes']],[0.,1.,2.])
        self.assertEqual(result['seed'],100)
        self.assertAlmostEqual(result['standing_success_fraction'],2/3)
        self.assertEqual(result['push_episode_count'],2)
        self.assertEqual(result['pushed_episode_success_fraction'],.5)
        self.assertIsNone(result['folded_success_fraction'])
        self.assertEqual(result['mean_reward'],1.)

    def test_cache_reuses_only_identical_actor_and_config(self):
        from climber_legged_rl.mujoco_training import CombinedEvalCallback
        torch.set_num_threads(1)
        env=gym.make('Pendulum-v1')
        try:
            model=PPO('MlpPolicy',env,n_steps=8,batch_size=8)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory);config=path/'config.yaml';physical=path/'model.xml'
                config.write_text(yaml.safe_dump(dict(experiment=dict(simulation_only=True,model_route='cad_dynamics'),
                    training=dict(cache_unchanged_policy_evaluation=True))))
                physical.write_text('<model/>')
                callback=CombinedEvalCallback(path,config,physical,frequency=1)
                def save(p):Path(p).write_bytes(b'policy')
                callback.model=SimpleNamespace(save=save,policy=model.policy,logger=SimpleNamespace(record=lambda *a:None))
                callback.locals={'infos':[]}
                result=dict(combined_success_fraction=1.,mean_reward=100.,
                    episodes=[dict(getup_metrics=dict(acquired=True))])
                with patch('climber_legged_rl.mujoco_training.evaluate',return_value=result) as evaluate:
                    callback.num_timesteps=1;callback._on_step();self.assertEqual(evaluate.call_count,2)
                    with torch.no_grad():model.policy.value_net.weight.add_(1.)
                    callback.num_timesteps=2;callback._on_step();self.assertEqual(evaluate.call_count,2)
                    self.assertTrue(callback.history[-1]['evaluation_cache_hit'])
                    self.assertEqual(callback.history[-1]['evaluation_source_timesteps'],1)
                    with torch.no_grad():model.policy.action_net.bias.add_(.001)
                    callback.num_timesteps=3;callback._on_step();self.assertEqual(evaluate.call_count,4)
                    config.write_text(config.read_text()+'# changed\n')
                    callback.num_timesteps=4;callback._on_step();self.assertEqual(evaluate.call_count,6)
                    self.assertFalse(callback.history[-1]['evaluation_cache_hit'])
        finally:env.close()

    def test_worker_count_is_bounded(self):
        for value in [0,9,1.5,True]:
            with self.assertRaises(ValueError):ParallelEvaluation(value)

if __name__=='__main__':unittest.main()

"""Combined checkpoint selection must represent both initial-state groups."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))

class CombinedSelectionTests(unittest.TestCase):
    def test_hard_push_selection_preserves_nominal_and_hard_standing(self):
        from climber_legged_rl.mujoco_training import CombinedEvalCallback
        import yaml
        for push_episodes in (0, 2):
            with self.subTest(push_episodes=push_episodes), tempfile.TemporaryDirectory() as directory:
                config = Path(directory)/'config.yaml'
                config.write_text(yaml.safe_dump(dict(experiment=dict(simulation_only=True,model_route='cad_dynamics'),
                    push=dict(enabled=True), training=dict(
                    evaluation_push_episodes=push_episodes, evaluation_hard_standing_seeds=[123],
                    evaluation_hard_push_seeds=[261106]))))
                callback = CombinedEvalCallback(directory, config, 'model.xml', frequency=1)
                def save(path): Path(path).write_bytes(b'policy')
                callback.model = SimpleNamespace(save=save, logger=SimpleNamespace(record=lambda *a: None))
                callback.locals = {'infos': []}
                rates = dict(ordinary=1., hard=1., hard_push=0., push=1., reward=100.)
                def assess(*args, seed, **kwargs):
                    rate = rates['hard_push' if seed == 261106 else 'hard' if seed == 123
                        else 'push' if kwargs.get('push_episode_probability') else 'ordinary']
                    return dict(combined_success_fraction=rate, mean_reward=rates['reward'],
                        episodes=[dict(getup_metrics=dict(acquired=True))])
                with patch('climber_legged_rl.mujoco_training.evaluate', side_effect=assess) as evaluator:
                    callback.num_timesteps = 1
                    callback._on_step()
                    initial = callback.best_score
                    calls = evaluator.call_args_list
                    hard_push = [c.kwargs for c in calls if c.kwargs['seed'] == 261106]
                    self.assertEqual(len(hard_push), 1)
                    self.assertEqual(hard_push[0]['reset_mode'], 'standing')
                    self.assertEqual(hard_push[0]['push_episode_probability'], 1.)
                    baseline = [c.kwargs for c in calls if c.kwargs['seed'] in (41000, 123)]
                    self.assertTrue(all(c['push_episode_probability'] == 0. for c in baseline))
                    rates.update(hard_push=1., reward=1.)
                    callback.num_timesteps = 2
                    callback._on_step()
                    self.assertGreater(callback.best_score, initial)
                    self.assertEqual(callback.history[-1]['hard_push_success_fraction'], 1.)
                    best = callback.best_score
                    for key in (('ordinary', 'hard', 'push') if push_episodes else ('ordinary', 'hard')):
                        rates[key] = 0.
                        callback.num_timesteps += 1
                        callback._on_step()
                        self.assertEqual(callback.best_score, best)
                        rates[key] = 1.

    def test_hard_push_evaluation_rejects_invalid_seeds_or_disabled_pushes(self):
        from climber_legged_rl.mujoco_training import CombinedEvalCallback
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/'config.yaml'
            for enabled, seeds in [(False, [1]), (True, [-1]), (True, [True]), (True, [1, 1]), (True, '1')]:
                config.write_text(yaml.safe_dump(dict(experiment=dict(simulation_only=True,model_route='cad_dynamics'),
                    push=dict(enabled=enabled),
                    training=dict(evaluation_hard_push_seeds=seeds))))
                with self.subTest(enabled=enabled, seeds=seeds), self.assertRaises(ValueError):
                    CombinedEvalCallback(directory, config, 'model.xml')

    def test_separate_push_groups_never_trade_away_unpushed_success(self):
        from climber_legged_rl.mujoco_training import CombinedEvalCallback
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'config.yaml'
            config.write_text(yaml.safe_dump(dict(experiment=dict(simulation_only=True,model_route='cad_dynamics'),push=dict(enabled=True),
                training=dict(evaluation_push_episodes=2))))
            callback=CombinedEvalCallback(directory,config,'model.xml',frequency=1,standing_episodes=3)
            saves=[]
            def save(path):
                Path(path).write_bytes(b'policy');saves.append(Path(path))
            callback.model=SimpleNamespace(save=save,logger=SimpleNamespace(record=lambda *args:None))
            callback.locals={'infos':[]}
            rates={'baseline':1.,'push':.5}
            def assess(*args,push_episode_probability,**kwargs):
                rate=rates['push' if push_episode_probability else 'baseline']
                return dict(combined_success_fraction=rate,mean_reward=100.,
                    episodes=[dict(getup_metrics=dict(acquired=True))])
            with patch('climber_legged_rl.mujoco_training.evaluate',side_effect=assess) as evaluation:
                callback.num_timesteps=1;callback._on_step()
                self.assertEqual([c.kwargs['push_episode_probability'] for c in evaluation.call_args_list],[0.,0.,1.,1.])
                self.assertEqual([c.kwargs['episodes'] for c in evaluation.call_args_list],[5,3,2,2])
                initial=callback.best_score
                rates.update(baseline=.9,push=1.)
                callback.num_timesteps=2;callback._on_step()
                self.assertEqual(callback.best_score,initial)
                rates.update(baseline=1.,push=1.)
                callback.num_timesteps=3;callback._on_step()
                self.assertGreater(callback.best_score,initial)
                self.assertEqual(sum(p.name=='best_model.zip' for p in saves),2)
                self.assertIn('push_groups',callback.history[-1])

    def test_separate_push_evaluation_requires_enabled_pushes(self):
        from climber_legged_rl.mujoco_training import CombinedEvalCallback
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'config.yaml'
            config.write_text(yaml.safe_dump(dict(experiment=dict(simulation_only=True,model_route='cad_dynamics'),training=dict(evaluation_push_episodes=2))))
            with self.assertRaisesRegex(ValueError,'push evaluation'):
                CombinedEvalCallback(directory,config,'model.xml')

    def test_all_pose_selection_never_hides_a_failed_pose_in_random_samples(self):
        from climber_legged_rl.mujoco_training import CombinedEvalCallback
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            callback=CombinedEvalCallback(directory,root/'configs/robot/cad_getup_stand_round10_ppo_refine.yaml',
                'model.xml',frequency=1,all_poses=True,standing_episodes=10)
            def save(path):Path(path).write_bytes(b'policy')
            callback.model=SimpleNamespace(save=save,logger=SimpleNamespace(record=lambda *args:None))
            callback.locals={'infos':[]}
            callback.num_timesteps=1
            def assess(*args,reset_mode,rearward_offset_m=None,**kwargs):
                rate=float(rearward_offset_m!=.37)
                episode=dict(combined_success=bool(rate),reward=100.,getup_metrics=dict(acquired=bool(rate)))
                return dict(combined_success_fraction=rate,mean_reward=100.,episodes=[episode])
            with patch('climber_legged_rl.mujoco_training.evaluate',side_effect=assess) as evaluation:
                callback._on_step()
            self.assertAlmostEqual(callback.history[0]['folded_success_fraction'],2/3)
            self.assertEqual(callback.history[0]['folded_pose_success']['0.37'],0.)
            self.assertEqual(len(evaluation.call_args_list),4)
            standing=[call for call in evaluation.call_args_list if call.kwargs['reset_mode']=='standing']
            self.assertEqual(standing[0].kwargs['episodes'],10)

    def test_parent_is_saved_before_training_and_not_replaced_by_worse_policy(self):
        from climber_legged_rl.mujoco_training import CombinedEvalCallback
        with tempfile.TemporaryDirectory() as directory:
            saves=[]
            def save(path):
                Path(path).write_bytes(b'parent')
                saves.append(Path(path))
            callback=CombinedEvalCallback(directory,'config.yaml','model.xml',frequency=10,evaluate_initial=True)
            callback.model=SimpleNamespace(save=save,logger=SimpleNamespace(record=lambda *args:None))
            callback.locals={'infos':[]}
            callback.num_timesteps=0
            rate=[.6]
            def assess(*args,**kwargs):
                return dict(combined_success_fraction=rate[0],mean_reward=100.,
                            episodes=[dict(getup_metrics=dict(acquired=True))])
            with patch('climber_legged_rl.mujoco_training.evaluate',side_effect=assess):
                callback._on_training_start()
                self.assertEqual(callback.history[0]['timesteps'],0)
                self.assertTrue((Path(directory)/'best/best_model.zip').is_file())
                rate[0]=0.
                callback.num_timesteps=10
                callback._on_step()
                self.assertEqual(sum(p.name=='best_model.zip' for p in saves),1)

    def test_high_standing_reward_cannot_replace_balanced_success(self):
        from climber_legged_rl.mujoco_training import CombinedEvalCallback
        with tempfile.TemporaryDirectory() as directory:
            saves=[]
            def save(path):
                path=Path(path)
                path.write_bytes(b'policy')
                saves.append(path)
            callback=CombinedEvalCallback(directory,'config.yaml','model.xml',frequency=1)
            callback.model=SimpleNamespace(save=save,logger=SimpleNamespace(record=lambda *args:None))
            callback.locals={'infos':[]}
            rates={'folded':.6,'standing':.8}
            def assess(*args,reset_mode,**kwargs):
                return dict(combined_success_fraction=rates[reset_mode],mean_reward=100.,
                    episodes=[dict(getup_metrics=dict(acquired=reset_mode=='standing'))])
            with patch('climber_legged_rl.mujoco_training.evaluate',side_effect=assess) as evaluation:
                callback.num_timesteps=1
                callback._on_step()
                initial_best=callback.best_score
                self.assertEqual({call.kwargs['reset_mode'] for call in evaluation.call_args_list},{'folded','standing'})
                rates.update(folded=0.,standing=1.)
                callback.num_timesteps=2
                callback._on_step()
                self.assertEqual(callback.best_score,initial_best)
                self.assertEqual(sum(path.name=='best_model.zip' for path in saves),1)
                self.assertFalse((Path(directory)/'evaluation_candidate.zip').exists())

    def test_hard_standing_success_outranks_reward_without_relaxing_getup(self):
        from climber_legged_rl.mujoco_training import CombinedEvalCallback
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/'config.yaml'
            settings = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/robot/cad_getup_stand_round21_hard_standing.yaml').read_text())
            settings['training'].update(evaluation_workers=1, cache_unchanged_policy_evaluation=False,
                evaluation_hard_standing_seeds=[123], evaluation_push_episodes=0)
            config.write_text(yaml.safe_dump(settings))
            callback = CombinedEvalCallback(directory, config, 'model.xml', frequency=1)
            def save(path): Path(path).write_bytes(b'policy')
            callback.model=SimpleNamespace(save=save, logger=SimpleNamespace(record=lambda *args:None))
            callback.locals={'infos':[]}
            hard_rate=[0.]; ordinary_rate=[1.]; reward=[100.]
            def assess(*args, seed, reset_mode, **kwargs):
                rate = hard_rate[0] if seed == 123 else ordinary_rate[0]
                return dict(combined_success_fraction=rate, mean_reward=reward[0],
                    episodes=[dict(getup_metrics=dict(acquired=True))])
            with patch('climber_legged_rl.mujoco_training.evaluate', side_effect=assess) as evaluator:
                callback.num_timesteps=1; callback._on_step(); first=callback.best_score
                hard_rate[0]=1.; reward[0]=1.
                callback.num_timesteps=2; callback._on_step()
                self.assertGreater(callback.best_score, first)
                self.assertEqual(callback.history[-1]['hard_standing_success_fraction'],1.)
                hard_calls=[c for c in evaluator.call_args_list if c.kwargs['seed']==123]
                self.assertTrue(all(c.kwargs['push_episode_probability']==0. for c in hard_calls))
                self.assertTrue(all(c.kwargs['reset_mode']=='standing' for c in hard_calls))
                best=callback.best_score; ordinary_rate[0]=0.
                callback.num_timesteps=3; callback._on_step()
                self.assertEqual(callback.best_score,best)

    def test_curriculum_callback_always_evaluates_all_three_groups(self):
        from climber_legged_rl.mujoco_training import CombinedEvalCallback
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'config.yaml'
            import yaml
            settings=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/robot/cad_getup_stand_round2.yaml').read_text())
            config.write_text(yaml.safe_dump(settings))
            callback=CombinedEvalCallback(directory,config,'model.xml',frequency=1)
            def save(path):Path(path).write_bytes(b'policy')
            callback.model=SimpleNamespace(save=save,logger=SimpleNamespace(record=lambda *args:None))
            callback.locals={'infos':[{'getup_metrics':dict(reset_mode='curriculum',combined_success=True,acquired=True)}]}
            callback.num_timesteps=1
            def assess(*args,reset_mode,**kwargs):
                rate=0. if reset_mode=='folded' else 1.
                return dict(combined_success_fraction=rate,mean_reward=100.,episodes=[dict(getup_metrics=dict(acquired=bool(rate)))])
            with patch('climber_legged_rl.mujoco_training.evaluate',side_effect=assess) as evaluation:
                callback._on_step()
                self.assertEqual({call.kwargs['reset_mode'] for call in evaluation.call_args_list},{'standing','folded','curriculum'})
                self.assertEqual(callback.history[0]['folded_success_fraction'],0.)
                self.assertEqual(callback.history[0]['curriculum_success_fraction'],1.)
                self.assertEqual(callback.best_score[0],0.)

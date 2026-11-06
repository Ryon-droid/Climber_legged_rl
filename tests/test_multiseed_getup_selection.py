"""Checkpoint selection must consider repeated resets and the weakest pose."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source'))
from climber_legged_rl.mujoco_training import CombinedEvalCallback


class MultiSeedGetupSelectionTests(unittest.TestCase):
    def test_repeated_pose_evaluation_prefers_improving_the_weak_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            config=folder/'config.yaml'
            config.write_text(yaml.safe_dump(dict(experiment=dict(simulation_only=True, model_route='cad_dynamics'),
                getup=dict(rearward_offsets_m=[.33, .34]),
                training=dict(evaluation_pose_episodes=8, evaluation_seed=431000,
                    evaluation_rank_worst_pose=True))))
            callback=CombinedEvalCallback(folder, config, folder/'model.xml', frequency=1, all_poses=True)
            callback.model=SimpleNamespace(save=lambda p: Path(p).write_bytes(b'policy'),
                logger=SimpleNamespace(record=lambda *args:None))
            callback.locals={'infos':[]}
            jobs=[]
            round_index=0

            def evaluate(**kwargs):
                jobs.append(kwargs)
                count=kwargs['episodes']
                offset=kwargs.get('rearward_offset_m')
                successes=count if offset is None else ({.33:3,.34:8} if round_index==0 else {.33:4,.34:6})[offset]
                records=[dict(combined_success=i<successes, reward=100., getup_metrics=dict(acquired=i<successes))
                    for i in range(count)]
                return dict(combined_success_fraction=successes/count, mean_reward=100., episodes=records)

            with patch('climber_legged_rl.mujoco_training.evaluate', side_effect=evaluate):
                callback.num_timesteps=1
                callback._on_step()
                self.assertEqual(callback.best_score[0], 3/8)
                round_index=1
                callback.num_timesteps=2
                callback._on_step()
            # Overall success decreases, but the weakest pose improves.
            self.assertEqual(callback.best_score[0], 4/8)
            pose_jobs=[j for j in jobs if j.get('rearward_offset_m') is not None]
            self.assertEqual(len(pose_jobs),4)
            self.assertTrue(all(j['episodes']==8 and j['seed']==431000 for j in pose_jobs))

    def test_invalid_sample_counts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            config=folder/'config.yaml'
            for count in (0, -1, True, 1.5, 65):
                config.write_text(yaml.safe_dump(dict(experiment=dict(simulation_only=True, model_route='cad_dynamics'),
                    training=dict(evaluation_pose_episodes=count))))
                with self.assertRaises(ValueError):
                    CombinedEvalCallback(folder, config, folder/'model.xml')


if __name__=='__main__':
    unittest.main()

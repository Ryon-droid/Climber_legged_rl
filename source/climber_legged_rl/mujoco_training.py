"""Bounded PPO experiment runner, evaluation and portable artifacts."""
from __future__ import annotations

from datetime import datetime
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecCheckNan

from .mujoco_env import make_mujoco_env as ClimberMujocoEnv
from .mujoco_model import ROOT, load_config


def _factory(model_path,config_path,seed,monitor_path):
    def make():
        env=ClimberMujocoEnv(model_path,config_path)
        if hasattr(env,'enable_training_curriculum'):
            env.enable_training_curriculum()
        training = load_config(config_path)['training']
        hard = training.get('hard_standing_resets', {})
        if monitor_path is not None and hard.get('enabled'):
            from .standing_reset_replay import StandingResetReplay
            env = StandingResetReplay(env, hard, seed)
        hard_push = training.get('hard_push_resets', {})
        if monitor_path is not None and hard_push.get('enabled'):
            from .push_reset_replay import PushResetReplay
            # Use a separate stream from the inner hard-standing sampler.
            env = PushResetReplay(env, hard_push, seed + 1000003)
        env.reset(seed=seed)
        return Monitor(env,filename=monitor_path)
    return make


def evaluate(checkpoint,config_path,model_path,episodes=10,seed=10000,random_actions=False,horizon_seconds=None,reset_mode=None,rearward_offset_m=None,push_episode_probability=None):
    if load_config(config_path).get('task',{}).get('velocity_tracking'):
        from .velocity_training import evaluate_velocity
        return evaluate_velocity(checkpoint,config_path,model_path,episodes,seed,horizon_seconds,random_actions)
    policy=PPO.load(str(checkpoint),device="cpu") if checkpoint else None
    env=ClimberMujocoEnv(model_path,config_path)
    if push_episode_probability is not None:
        if not env.push.enabled or not 0 <= push_episode_probability <= 1:
            env.close()
            raise ValueError('Push evaluation override requires enabled pushes and probability in [0,1]')
        env.push.config = dict(env.push.config, episode_probability=float(push_episode_probability))
    if rearward_offset_m is not None:
        env.pose_bank=[p for p in env.pose_bank if abs(p['rearward_offset_m']-rearward_offset_m)<1e-8]
        if not env.pose_bank:
            env.close()
            raise ValueError('Evaluation offset must match a prepared prone pose')
    if horizon_seconds is not None:
        if not 0 < horizon_seconds <= 120:
            raise ValueError("Evaluation horizon must be in (0,120] seconds")
        env.max_steps=int(np.ceil(horizon_seconds/env.dt))
    rng=np.random.default_rng(seed)
    records=[]
    for episode in range(episodes):
        obs,_=env.reset(seed=seed+episode, options={'reset_mode':reset_mode} if reset_mode is not None else None)
        heights=[]; uprights=[]; contacts=[]; speeds=[]; max_closure=0.0
        displacements=[]; headings=[]; planar_speeds=[]; yaw_rates=[]
        saturation = np.zeros(6)
        peak_torque = np.zeros(6)
        velocity_estimation_errors = []
        initial_position=env.data.xpos[env.base_id,:2].copy()
        initial_warning_counts=np.array(env.data.warning.number).copy()
        while True:
            if policy is not None:
                from .velocity_estimation import VelocityEstimatedPolicy
                if isinstance(policy.policy, VelocityEstimatedPolicy):
                    with torch.no_grad():
                        estimate = policy.policy.mlp_extractor.policy_net.estimate_velocity(
                            torch.as_tensor(obs, device=policy.device).reshape(1, -1)).cpu().numpy()[0]
                    velocity_estimation_errors.append(estimate - obs[-3:])
                action,_=policy.predict(obs,deterministic=True)
            else:
                action=rng.uniform(-1,1,6) if random_actions else np.zeros(6)
            obs,reward,done,truncated,info=env.step(action)
            heights.append(info["hip_height_m"]); uprights.append(info["upright"]); contacts.append(info["wheel_contacts"])
            speeds.append(info["forward_velocity_m_s"])
            displacements.append(info['planar_displacement_m'])
            headings.append(info['heading_error_rad'])
            planar_speeds.append(info['planar_speed_m_s'])
            yaw_rates.append(info['yaw_rate_rad_s'])
            saturation += np.asarray(info['motor_saturation_fraction'])
            peak_torque = np.maximum(peak_torque, np.abs(info['motor_torque_Nm']))
            max_closure=max(max_closure,info["closure_error_m"])
            if done or truncated:
                records.append({**info["episode_metrics"],"mean_hip_height_m":float(np.mean(heights)),
                                "mean_upright":float(np.mean(uprights)),"both_wheels_contact_fraction":float(np.mean(np.array(contacts)==2)),
                                "max_closure_error_m":max_closure,
                                "planar_displacement_m":float(np.linalg.norm(env.data.xpos[env.base_id,:2]-initial_position)),
                                "forward_velocity_rmse_m_s":float(np.sqrt(np.mean((np.array(speeds)-env.commands[0])**2))),
                                "simulator_warning_counts":(np.array(env.data.warning.number)-initial_warning_counts).tolist()})
                records[-1].update(motor_saturation_fraction=(saturation / len(heights)).tolist(),
                    peak_sampled_motor_torque_Nm=peak_torque.tolist())
                if velocity_estimation_errors:
                    errors = np.array(velocity_estimation_errors)
                    records[-1]['velocity_estimator_rmse_xyz_m_s'] = np.sqrt(np.mean(errors**2, axis=0)).tolist()
                records[-1].update(
                    max_planar_displacement_m=float(max(displacements)),
                    max_heading_error_deg=float(np.degrees(max(abs(np.unwrap(headings))))),
                    planar_speed_rmse_m_s=float(np.sqrt(np.mean(np.square(planar_speeds)))),
                    yaw_rate_rmse_rad_s=float(np.sqrt(np.mean(np.square(yaw_rates)))))
                records[-1]['stationary_success'] = env.stationary_success(
                    truncated, records[-1]['max_planar_displacement_m'],
                    float(max(abs(np.unwrap(headings)))), records[-1]['planar_speed_rmse_m_s'],
                    records[-1]['yaw_rate_rmse_rad_s'])
                if env.push.enabled:
                    records[-1]['push_metrics'] = info['push_metrics']
                    records[-1]['push_recovery_success'] = info['push_recovery_success']
                if 'getup_metrics' in info:
                    records[-1]['getup_metrics'] = info['getup_metrics']
                    records[-1]['combined_success'] = info['combined_success']
                break
    env.close()
    return {"simulation_only":True,"episodes":records,"seed":seed,
            "evaluation_horizon_seconds":env.max_steps*env.dt,
            "checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
            "mean_reward":float(np.mean([r["reward"] for r in records])),
            "mean_episode_seconds":float(np.mean([r["seconds"] for r in records])),
            "timeout_fraction":float(np.mean([r["reason"]=="timeout" for r in records])),
            "stationary_success_fraction":float(np.mean([r['stationary_success'] for r in records])),
            **({'combined_success_fraction':float(np.mean([r['combined_success'] for r in records])),
                'getup_episode_count':sum(r['getup_metrics']['reset_mode']=='folded' for r in records),
                'folded_success_fraction':float(np.mean([r['combined_success'] for r in records if r['getup_metrics']['reset_mode']=='folded']))
                    if any(r['getup_metrics']['reset_mode']=='folded' for r in records) else None,
                'curriculum_episode_count':sum(r['getup_metrics']['reset_mode']=='curriculum' for r in records),
                'curriculum_success_fraction':float(np.mean([r['combined_success'] for r in records if r['getup_metrics']['reset_mode']=='curriculum']))
                    if any(r['getup_metrics']['reset_mode']=='curriculum' for r in records) else None,
                'standing_success_fraction':float(np.mean([r['combined_success'] for r in records if r['getup_metrics']['reset_mode']=='standing']))
                    if any(r['getup_metrics']['reset_mode']=='standing' for r in records) else None} if records and 'getup_metrics' in records[0] else {}),
            **({"push_recovery_success_fraction":float(np.mean([r['push_recovery_success'] for r in records])),
                "push_episode_count":sum(r['push_metrics']['push_count']>0 for r in records),
                "pushed_episode_success_fraction":float(np.mean([r['push_recovery_success'] for r in records if r['push_metrics']['push_count']>0]))
                    if any(r['push_metrics']['push_count']>0 for r in records) else None} if env.push.enabled else {}),
            "model_route":env.config['experiment'].get('model_route','legacy_approximate'),
            "warning":"Simulation evaluation only; model assumptions are recorded in config.yaml. Not hardware validation."}


class CombinedEvalCallback(BaseCallback):
    """Fixed, separately forced get-up and standing episodes select the best actor."""
    def __init__(self, run_directory, config_path, model_path, frequency=50000, evaluate_initial=False,
                 all_poses=False, standing_episodes=5):
        super().__init__()
        self.folder=Path(run_directory)
        self.config_path=config_path
        self.model_path=model_path
        self.frequency=frequency
        self.evaluate_initial=evaluate_initial
        self.all_poses=all_poses
        self.standing_episodes=int(standing_episodes)
        if self.standing_episodes<1:
            raise ValueError('Standing evaluation episodes must be positive')
        self.pose_offsets=load_config(config_path)['getup']['rearward_offsets_m'] if all_poses else []
        self.last_evaluation=0
        self.best_score=None
        self.history=[]
        curriculum=load_config(config_path).get('getup',{}).get('reset_curriculum',{}).get('enabled') if Path(config_path).is_file() else False
        self.modes=('folded','standing','curriculum') if curriculum else ('folded','standing')
        self.counts={mode:[] for mode in self.modes}
        config = load_config(config_path) if Path(config_path).is_file() else {}
        evaluation_settings = config.get('training', {})
        self.pose_episodes = evaluation_settings.get('evaluation_pose_episodes', 1)
        self.evaluation_seed = evaluation_settings.get('evaluation_seed', 41000)
        if type(self.pose_episodes) is not int or not 1 <= self.pose_episodes <= 64:
            raise ValueError('Pose evaluation episodes must be an integer from 1 to 64')
        if type(self.evaluation_seed) is not int or self.evaluation_seed < 0:
            raise ValueError('Evaluation seed must be a nonnegative integer')
        self.rank_worst_pose = evaluation_settings.get('evaluation_rank_worst_pose', False)
        if type(self.rank_worst_pose) is not bool:
            raise ValueError('evaluation_rank_worst_pose must be a boolean')
        self.push_enabled = bool(config.get('push', {}).get('enabled'))
        self.push_episodes = int(config.get('training', {}).get('evaluation_push_episodes', 0))
        if self.push_episodes < 0 or (self.push_episodes and not config.get('push', {}).get('enabled')):
            raise ValueError('Separate push evaluation requires enabled pushes and positive episode count')
        self.hard_standing_seeds = config.get('training', {}).get('evaluation_hard_standing_seeds', [])
        if (not isinstance(self.hard_standing_seeds, list)
                or any(type(seed) is not int or seed < 0 for seed in self.hard_standing_seeds)
                or len(set(self.hard_standing_seeds)) != len(self.hard_standing_seeds)):
            raise ValueError('Hard standing evaluation requires distinct nonnegative integer seeds')
        self.hard_push_seeds = config.get('training', {}).get('evaluation_hard_push_seeds', [])
        if (not isinstance(self.hard_push_seeds, list)
                or any(type(seed) is not int or seed < 0 for seed in self.hard_push_seeds)
                or len(set(self.hard_push_seeds)) != len(self.hard_push_seeds)):
            raise ValueError('Hard push evaluation requires distinct nonnegative integer seeds')
        if self.hard_push_seeds and not self.push_enabled:
            raise ValueError('Hard push evaluation requires enabled pushes')
        from .evaluation_runtime import ParallelEvaluation
        self.evaluator = ParallelEvaluation(config.get('training', {}).get('evaluation_workers', 1))
        self.cache_unchanged = bool(config.get('training', {}).get('cache_unchanged_policy_evaluation', False))
        self._cached_key = self._cached_record = None

    def _on_training_start(self):
        if self.evaluate_initial:
            # Preserve the reviewed parent as best before the first PPO update.
            self.last_evaluation=self.num_timesteps-self.frequency
            self._on_step()

    def _on_step(self):
        for info in self.locals.get('infos',[]):
            metrics=info.get('getup_metrics')
            if metrics:
                self.counts[metrics['reset_mode']].append(metrics)
                self.counts[metrics['reset_mode']]=self.counts[metrics['reset_mode']][-100:]
        if self.num_timesteps-self.last_evaluation < self.frequency:
            return True
        self.last_evaluation=self.num_timesteps
        candidate=self.folder/'evaluation_candidate.zip'
        self.model.save(candidate)
        started = time.monotonic()
        cache_key = None
        if self.cache_unchanged:
            from .evaluation_runtime import deterministic_actor_digest
            cache_key = (deterministic_actor_digest(self.model.policy),
                hashlib.sha256(Path(self.config_path).read_bytes()).hexdigest(),
                hashlib.sha256(Path(self.model_path).read_bytes()).hexdigest())
            if cache_key == self._cached_key:
                record = copy.deepcopy(self._cached_record)
                record.update(timesteps=self.num_timesteps, evaluation_cache_hit=True,
                    evaluation_source_timesteps=self._cached_record['timesteps'],
                    evaluation_seconds=time.monotonic()-started)
                # Critic updates change the checkpoint archive even when its actor is identical.
                digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
                def refresh_digest(value):
                    if isinstance(value, dict):
                        for key, child in value.items():
                            if key == 'checkpoint_sha256': value[key] = digest
                            else: refresh_digest(child)
                    elif isinstance(value, list):
                        for child in value: refresh_digest(child)
                refresh_digest(record)
                return self._record_evaluation(record, candidate)
        baseline_options = {'push_episode_probability': 0.0} if (self.push_episodes or self.hard_push_seeds) else {}
        common = dict(checkpoint=candidate,config_path=self.config_path,model_path=self.model_path)
        jobs = {'baseline_'+mode: dict(common,episodes=self.standing_episodes if mode=='standing' else 5,
                    seed=self.evaluation_seed,reset_mode=mode,**baseline_options)
                for mode in self.modes if not (self.all_poses and mode=='folded')}
        for offset in self.pose_offsets:
            jobs['pose_'+str(offset)] = dict(common,episodes=self.pose_episodes,seed=self.evaluation_seed,reset_mode='folded',
                rearward_offset_m=offset,**baseline_options)
        for seed in self.hard_standing_seeds:
            jobs['hard_'+str(seed)] = dict(common, episodes=1, seed=seed, reset_mode='standing',
                **({'push_episode_probability': 0.0} if self.push_enabled else {}))
        for seed in self.hard_push_seeds:
            jobs['hard_push_'+str(seed)] = dict(common, episodes=1, seed=seed,
                reset_mode='standing', push_episode_probability=1.0)
        if self.push_episodes:
            for mode in ('folded', 'standing'):
                jobs['push_'+mode] = dict(common,episodes=self.push_episodes,seed=42000,
                    reset_mode=mode,push_episode_probability=1.0)
        # Serial mode keeps the established evaluator and its test seams unchanged.
        results = ({key:evaluate(**job) for key,job in jobs.items()} if self.evaluator.workers == 1
                   else self.evaluator.run(jobs))
        groups = {mode:results['baseline_'+mode] for mode in self.modes
                  if not (self.all_poses and mode=='folded')}
        pose_results={}
        if self.all_poses:
            pose_results={str(offset):results['pose_'+str(offset)] for offset in self.pose_offsets}
            records=[record for group in pose_results.values() for record in group['episodes']]
            groups['folded']=dict(episodes=records,
                combined_success_fraction=float(np.mean([r['combined_success'] for r in records])),
                mean_reward=float(np.mean([r['reward'] for r in records])))
        rates=[groups[mode]['combined_success_fraction'] for mode in self.modes]
        acquired=float(np.mean([r['getup_metrics']['acquired'] for r in groups['folded']['episodes']]))
        score=(min(rates),float(np.mean(rates)),acquired,float(np.mean([g['mean_reward'] for g in groups.values()])))
        if self.rank_worst_pose and pose_results:
            score=(min([*rates, *[g['combined_success_fraction'] for g in pose_results.values()]]), *score[1:])
        record=dict(timesteps=self.num_timesteps,folded_success_fraction=rates[0],standing_success_fraction=rates[1],
                    folded_acquired_fraction=acquired,score=score,groups=groups)
        if pose_results:
            record['folded_pose_success']={key:result['combined_success_fraction'] for key,result in pose_results.items()}
        if 'curriculum' in self.modes:
            record['curriculum_success_fraction']=rates[2]
        if self.push_episodes:
            pushed = {mode: results['push_'+mode] for mode in ('folded', 'standing')}
            # Count only episodes that actually received a pulse as recovery evidence.
            # Failure before a pulse still counts against combined task success.
            push_rates = [pushed[mode]['combined_success_fraction'] for mode in ('folded', 'standing')]
            score = (*score[:3], min(push_rates), float(np.mean(push_rates)), score[3])
            record.update(score=score, push_groups=pushed,
                pushed_folded_success_fraction=push_rates[0],pushed_standing_success_fraction=push_rates[1])
        if self.hard_standing_seeds:
            hard = {str(seed): results['hard_'+str(seed)] for seed in self.hard_standing_seeds}
            hard_rate = float(np.mean([g['combined_success_fraction'] for g in hard.values()]))
            # Preserve ordinary get-up/standing first, then rank the difficult initial states.
            record.update(score=(*score[:3], hard_rate, *score[3:]),
                hard_standing_success_fraction=hard_rate, hard_standing_groups=hard)
        if self.hard_push_seeds:
            hard_push = {str(seed): results['hard_push_'+str(seed)] for seed in self.hard_push_seeds}
            hard_push_rate = float(np.mean([g['combined_success_fraction'] for g in hard_push.values()]))
            # Keep all existing success groups ahead of this seen training case;
            # solving it must not replace broader recovery with overfitting.
            record.update(score=(*record['score'][:-1], hard_push_rate, record['score'][-1]),
                hard_push_success_fraction=hard_push_rate, hard_push_groups=hard_push)
        record.update(evaluation_cache_hit=False,evaluation_seconds=time.monotonic()-started)
        if self.cache_unchanged:
            record['deterministic_actor_sha256'] = cache_key[0]
            self._cached_key,self._cached_record = cache_key,copy.deepcopy(record)
        return self._record_evaluation(record,candidate)

    def _record_evaluation(self,record,candidate):
        score=tuple(record['score'])
        self.history.append(record)
        (self.folder/'combined_evaluations.json').write_text(json.dumps(self.history,indent=2))
        for key in ('folded_success_fraction','standing_success_fraction','folded_acquired_fraction'):
            self.logger.record('eval/'+key,record[key])
        if 'curriculum_success_fraction' in record:
            self.logger.record('eval/curriculum_success_fraction',record['curriculum_success_fraction'])
        if 'hard_standing_success_fraction' in record:
            self.logger.record('eval/hard_standing_success_fraction', record['hard_standing_success_fraction'])
        if 'hard_push_success_fraction' in record:
            self.logger.record('eval/hard_push_success_fraction', record['hard_push_success_fraction'])
        self.logger.record('eval/cache_hit',float(record['evaluation_cache_hit']))
        self.logger.record('eval/seconds',record['evaluation_seconds'])
        for mode in ('folded', 'standing'):
            if 'pushed_'+mode+'_success_fraction' in record:
                self.logger.record('eval/pushed_'+mode+'_success_fraction',record['pushed_'+mode+'_success_fraction'])
        if self.best_score is None or score>self.best_score:
            self.best_score=score
            (self.folder/'best').mkdir(exist_ok=True)
            self.model.save(self.folder/'best/best_model.zip')
        candidate.unlink()
        print('COMBINED_EVALUATION '+json.dumps({k:record[k] for k in record if k not in ('groups','push_groups','hard_standing_groups','hard_push_groups')}),flush=True)
        return True

    def _on_rollout_end(self):
        for mode,records in self.counts.items():
            if records:
                self.logger.record('getup/'+mode+'_episodes_last100',len(records))
                self.logger.record('getup/'+mode+'_success_last100',float(np.mean([r['combined_success'] for r in records])))
                if mode=='folded':
                    self.logger.record('getup/folded_acquired_last100',float(np.mean([r['acquired'] for r in records])))


class StatusCallback(BaseCallback):
    def __init__(self,run_directory,total_steps,start_steps=0):
        super().__init__()
        self.run_directory=Path(run_directory)
        self.total_steps=total_steps
        self.start_steps=start_steps
        self.started=time.monotonic()
        self.last_update=0

    def _on_training_start(self):
        self.started=time.monotonic()
        self.last_update=self.start_steps

    def _remaining_seconds(self):
        completed=self.num_timesteps-self.start_steps
        if completed<=0:
            return None
        remaining=max(self.total_steps-completed,0)
        return int(np.ceil((time.monotonic()-self.started)*remaining/completed))

    def _on_rollout_end(self):
        remaining=self._remaining_seconds()
        if remaining is not None:
            hours,remainder=divmod(remaining,3600)
            minutes,seconds=divmod(remainder,60)
            self.logger.record("time/remaining",f"{hours:02d}:{minutes:02d}:{seconds:02d}",exclude="tensorboard")

    def _on_step(self):
        if self.num_timesteps-self.last_update>=1024:
            self.last_update=self.num_timesteps
            status={"state":"training","simulation_only":True,"timesteps":self.num_timesteps,
                    "requested_additional_timesteps":self.total_steps,
                    "starting_timesteps":self.start_steps,
                    "additional_timesteps":self.num_timesteps-self.start_steps,
                    "target_total_timesteps":self.start_steps+self.total_steps,
                    "elapsed_seconds":time.monotonic()-self.started,
                    "remaining_seconds":self._remaining_seconds(),
                    "pid":os.getpid()}
            (self.run_directory/"status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        return True


def tightened_spring_limit_transfer(previous_model, model_path, previous_config, config):
    """Migrate an existing measured upper stop only with an explicit bounded recipe."""
    from .spring_geometry import configured_eye_limits
    previous_limits = configured_eye_limits(previous_config, required=True)
    limits = configured_eye_limits(config, required=True)
    recipe = config['training'].get('spring_limit_migration', {})
    if (not isinstance(recipe, dict) or recipe.get('mode') != 'tighten_upper_stop'
            or recipe.get('previous_eye_center_limits_m') != previous_limits):
        raise ValueError('Existing spring stops require an explicit tighten_upper_stop migration with matching previous limits')
    old_spring, new_spring = dict(previous_config['spring']), dict(config['spring'])
    old_spring.pop('eye_center_limits_m')
    new_spring.pop('eye_center_limits_m')
    if old_spring != new_spring:
        raise ValueError('Spring-stop tightening cannot change force or other mechanics')
    old, new = ET.parse(previous_model).getroot(), ET.parse(model_path).getroot()
    for side, bounds in limits.items():
        before = previous_limits[side]
        if bounds[0] is None or before[0] != bounds[0] or not bounds[0] < bounds[1] < before[1]:
            raise ValueError('Spring-stop tightening must preserve each lower stop and reduce each upper stop')
        name = side + '_gas_slide_joint'
        a, b = (root.find('.//joint[@name="'+name+'"]') for root in (old, new))
        body = new.find('.//body[@name="'+side+'_gas_rod"]')
        if (a is None or b is None or body is None or a.get('type') != 'slide'
                or b.get('type') != 'slide' or a.get('limited') != 'true' or b.get('limited') != 'true'):
            raise ValueError('Spring-stop tightening requires two existing limited sliders')
        zero = np.linalg.norm(np.fromstring(body.get('pos', ''), sep=' '))
        for element, expected in ((a, before), (b, bounds)):
            actual = np.fromstring(element.get('range', ''), sep=' ')
            if actual.shape != (2,) or not np.allclose(actual + zero, expected, atol=1e-12, rtol=0):
                raise ValueError('Gas-slider range does not match declared pin-center lengths')
        a.set('range', b.get('range'))
    canonical = lambda root: ET.canonicalize(ET.tostring(root, encoding='unicode'), strip_text=True)
    if canonical(old) != canonical(new):
        raise ValueError('Spring-stop tightening rejects XML changes beyond the two slider ranges')


def spring_limit_transfer(previous_model, model_path, previous_config, config, allow_getup_guards=False):
    """Permit only adding explicit spring travel stops; all other XML stays identical."""
    from .spring_geometry import configured_eye_limits
    limits = configured_eye_limits(config, required=True)
    old_spring = dict(previous_config['spring'])
    new_spring = dict(config['spring'])
    if old_spring.pop('eye_center_limits_m', None) is not None:
        if allow_getup_guards:
            raise ValueError('Cannot combine existing spring-stop tightening with CAD guard migration')
        return tightened_spring_limit_transfer(previous_model, model_path, previous_config, config)
    new_spring.pop('eye_center_limits_m', None)
    if old_spring != new_spring:
        raise ValueError('Spring-limit transfer cannot change spring force or other mechanics')
    old, new = ET.parse(previous_model).getroot(), ET.parse(model_path).getroot()
    if allow_getup_guards:
        expected = {name: [-2.5, 2.5] for name in (
            'left_link2_joint', 'left_link_rear3_joint', 'left_link_rear4_joint', 'left_link_reear2_joint',
            'right_link2_joint', 'right_rear_link3_joint', 'right_rear_link 4_joint', 'right_rear_link2_joint')}
        expected.update({name: [-3.2, 1.57] for name in ('left_link_rear_1_joint', 'right_rear_link1_joint')})
        if (not config.get('getup', {}).get('enabled')
                or previous_config['physics'].get('joint_range_overrides_rad')
                or config['physics'].get('joint_range_overrides_rad') != expected):
            raise ValueError('Get-up transfer only permits the explicit ten CAD branch guards')
        for name, bounds in expected.items():
            a = old.find('.//joint[@name="'+name+'"]')
            b = new.find('.//joint[@name="'+name+'"]')
            if (a is None or b is None or a.get('limited') != 'true' or b.get('limited') != 'true'
                    or a.get('range') != '-1.57 1.57'
                    or not np.array_equal(np.fromstring(b.get('range', ''), sep=' '), bounds)):
                raise ValueError('Unexpected get-up CAD joint guard: ' + name)
            a.set('range', b.get('range'))
    for side, bounds in limits.items():
        if bounds[0] is None:
            raise ValueError('Spring-limit transfer requires both measured end lengths')
        name = side + '_gas_slide_joint'
        a = old.find('.//joint[@name="'+name+'"]')
        b = new.find('.//joint[@name="'+name+'"]')
        body = new.find('.//body[@name="'+side+'_gas_rod"]')
        if a is None or b is None or body is None or b.get('limited') != 'true' or a.get('limited') != 'false':
            raise ValueError('Spring-limit transfer requires the original unlimited gas sliders')
        zero = np.linalg.norm([float(v) for v in body.get('pos').split()])
        actual = np.array([float(v) for v in b.get('range', '').split()])
        if actual.shape != (2,) or not np.allclose(actual + zero, bounds, atol=1e-12, rtol=0):
            raise ValueError('Gas-slider range does not match the confirmed pin-center lengths')
        a.set('limited', b.get('limited'))
        a.set('range', b.get('range'))
        for key, expected in [('solreflimit', '0.002 1'), ('solimplimit', '0.999 0.9999 0.0001 0.5 2')]:
            if b.get(key) != expected:
                raise ValueError('Unexpected numerical gas-spring stop settings: ' + key)
            a.set(key, expected)
    canonical = lambda root: ET.canonicalize(ET.tostring(root, encoding='unicode'), strip_text=True)
    if canonical(old) != canonical(new):
        raise ValueError('Spring-limit transfer rejects model changes beyond the two slider limits')


def supported_getup_transfer(previous_model, model_path, previous_config, config):
    """Allow the measured peak envelope while preserving the parent standing controller."""
    support = config['getup']['support_control']
    old_control, new_control = previous_config['control'], config['control']
    expected = {'leg_kp': 80., 'leg_kd': 2., 'leg_torque_limit_Nm': 40.}
    if any(new_control.get(k) != v for k, v in expected.items()):
        raise ValueError('Supported get-up requires the validated 80/2/40 rise envelope')
    restored = dict(new_control)
    restored.update({k: old_control[k] for k in expected})
    preserved = {'standing_leg_kp': 'leg_kp', 'standing_leg_kd': 'leg_kd',
                 'standing_torque_limit_Nm': 'leg_torque_limit_Nm',
                 'standing_leg_action_scale_rad': 'leg_action_scale_rad'}
    if restored != old_control or any(support.get(k) != old_control[v] for k, v in preserved.items()):
        raise ValueError('Supported get-up must preserve all parent standing controls')
    if (support.get('rated_torque_Nm') != 20. or support.get('peak_window_seconds') != .6
            or support.get('handoff_seconds') != 1.02 or support.get('learned_assistance_scale') != 0.):
        raise ValueError('Supported get-up requires the validated reference and handoff schedule')
    old, new = ET.parse(previous_model).getroot(), ET.parse(model_path).getroot()
    for name in old_control['leg_joints']:
        a = old.find('./actuator/motor[@joint="'+name+'"]')
        b = new.find('./actuator/motor[@joint="'+name+'"]')
        if a is None or b is None:
            raise ValueError('Supported get-up requires four matching leg motors')
        for element, limit in ((a, old_control['leg_torque_limit_Nm']), (b, 40.)):
            if (element.get('ctrllimited') != 'true' or not np.array_equal(
                    np.fromstring(element.get('ctrlrange', ''), sep=' '), [-limit, limit])):
                raise ValueError('Unexpected supported get-up motor limits')
        a.set('ctrlrange', b.get('ctrlrange'))
    canonical = lambda root: ET.canonicalize(ET.tostring(root, encoding='unicode'), strip_text=True)
    if canonical(old) != canonical(new):
        raise ValueError('Supported get-up rejects XML changes beyond four leg motor limits')


def full_policy_getup_transfer(previous_model, model_path, previous_config, config):
    """Migrate the old unassisted actor to the confirmed 20/40 Nm motor envelope."""
    envelope = config['getup']['policy_control']
    old_control, new_control = previous_config['control'], config['control']
    expected_old = {'leg_kp': 20., 'leg_kd': 1., 'leg_torque_limit_Nm': 10.}
    expected_new = {'leg_kp': 80., 'leg_kd': 2., 'leg_torque_limit_Nm': 40.}
    restored = dict(new_control)
    restored.update(expected_old)
    if (any(old_control.get(k) != v for k, v in expected_old.items())
            or any(new_control.get(k) != v for k, v in expected_new.items()) or restored != old_control):
        raise ValueError('Full policy migration only permits the validated leg motor envelope')
    expected_envelope = dict(enabled=True, rated_torque_Nm=20., peak_window_seconds=.6,
                             standing_leg_kp=20., standing_leg_kd=1.,
                             standing_torque_limit_Nm=10.)
    if envelope != expected_envelope or not previous_config.get('getup', {}).get('dynamic_control', {}).get('enabled'):
        raise ValueError('Full policy migration requires the original dynamic actor and validated envelope')
    old, new = ET.parse(previous_model).getroot(), ET.parse(model_path).getroot()
    for name in old_control['leg_joints']:
        a = old.find('./actuator/motor[@joint="'+name+'"]')
        b = new.find('./actuator/motor[@joint="'+name+'"]')
        if a is None or b is None:
            raise ValueError('Full policy migration requires four leg motors')
        for element, limit in ((a, 10.), (b, 40.)):
            if (element.get('ctrllimited') != 'true' or not np.array_equal(
                    np.fromstring(element.get('ctrlrange', ''), sep=' '), [-limit, limit])):
                raise ValueError('Unexpected full policy motor limits')
        a.set('ctrlrange', b.get('ctrlrange'))
    canonical = lambda root: ET.canonicalize(ET.tostring(root, encoding='unicode'), strip_text=True)
    if canonical(old) != canonical(new):
        raise ValueError('Full policy migration rejects XML changes beyond four leg motor limits')


def wheel_control_transfer(previous_model, model_path, previous_config):
    """Permit the validated wheel servo change plus chassis pose corrections."""
    old, new = ET.parse(previous_model).getroot(), ET.parse(model_path).getroot()
    for name in previous_config['control']['wheel_joints']:
        a = old.find('./actuator/motor[@joint="' + name + '"]')
        b = new.find('./actuator/motor[@joint="' + name + '"]')
        if a is None or b is None:
            raise ValueError('Wheel control migration requires both wheel motors')
        if (a.get('ctrllimited') != 'true' or b.get('ctrllimited') != 'true'
                or not np.array_equal(np.fromstring(a.get('ctrlrange', ''), sep=' '), [-2., 2.])
                or not np.array_equal(np.fromstring(b.get('ctrlrange', ''), sep=' '), [-4., 4.])):
            raise ValueError('Unexpected wheel motor limits')
        a.set('ctrlrange', b.get('ctrlrange'))
    # Copy wheel contact-layer edits: every geom's contype/conaffinity changed
    # so only the tire collides with ground in the level-chassis artefact.
    new_geoms = {g.get('name'): g for g in new.findall('.//body/geom')}
    for og in old.findall('.//body/geom'):
        ng = new_geoms.get(og.get('name'))
        if ng is not None:
            og.set('contype', ng.get('contype'))
            og.set('conaffinity', ng.get('conaffinity'))
    # Copy base pose and new keyframes.
    old_base = old.find('./worldbody/body')
    new_base = new.find('./worldbody/body')
    old_base.set('pos', new_base.get('pos'))
    old_base.set('quat', new_base.get('quat'))
    old_keys = {k.get('name'): k for k in old.findall('.//keyframe/key')}
    for nk in new.findall('.//keyframe/key'):
        oname = nk.get('name')
        if oname in old_keys:
            old_keys[oname].set('qpos', nk.get('qpos'))
    canonical = lambda root: ET.canonicalize(ET.tostring(root, encoding='unicode'), strip_text=True)
    if canonical(old) != canonical(new):
        raise ValueError('Wheel control migration rejects XML changes beyond wheel motor limits and base pose')


def validate_policy_transfer(checkpoint, model_path, config, warm_start=False):
    """Strict transfer with explicit, narrowly checked spring-stop / CAD-guard migrations."""
    checkpoint=Path(checkpoint).resolve()
    previous=checkpoint.parent.parent if checkpoint.parent.name in ('checkpoints','best') else checkpoint.parent
    previous_model=previous/'robot_experiment.xml'
    if not checkpoint.is_file() or not previous_model.is_file():
        raise ValueError('Policy transfer requires an existing checkpoint and the identical derived model.')
    previous_config=yaml.safe_load((previous/'config.yaml').read_text())
    if not warm_start:
        if (previous_config['experiment'].get('physics_backend', 'mujoco') !=
                config['experiment'].get('physics_backend', 'mujoco')):
            raise ValueError('Changing physics backend requires --warm-start, not --resume')
        if (bool(previous_config['training'].get('velocity_estimator', {}).get('enabled')) !=
                bool(config['training'].get('velocity_estimator', {}).get('enabled'))):
            raise ValueError('Changing velocity estimation requires explicit warm-start migration')
    old_dynamic=previous_config.get('getup',{}).get('dynamic_control',{}).get('enabled',False)
    new_dynamic=config.get('getup',{}).get('dynamic_control',{}).get('enabled',False)
    old_policy = previous_config.get('getup', {}).get('policy_control', {}).get('enabled', False)
    new_policy = config.get('getup', {}).get('policy_control', {}).get('enabled', False)
    velocity_standing = bool(warm_start and not config.get('getup', {}).get('enabled', False)
                             and config.get('task', {}).get('velocity_tracking', False)
                             and config['training'].get('allow_velocity_standing_warm_start', False)
                             and old_policy and not new_policy)
    if warm_start and old_dynamic!=new_dynamic:
        if not (new_dynamic and config['training'].get('allow_dynamic_getup_warm_start',False)) and not velocity_standing:
            raise ValueError('Dynamic get-up changes the rising action map; explicit warm-start permission required')
    model_changed = previous_model.read_bytes()!=Path(model_path).read_bytes()
    old_support = previous_config.get('getup', {}).get('support_control', {}).get('enabled', False)
    new_support = config.get('getup', {}).get('support_control', {}).get('enabled', False)
    support_migration = bool(warm_start and not old_support and new_support
                             and config['training'].get('allow_supported_getup_warm_start', False))
    if warm_start and old_support != new_support:
        if not support_migration:
            raise ValueError('Supported get-up action map requires explicit warm-start permission')
        supported_getup_transfer(previous_model, model_path, previous_config, config)
    policy_migration = bool(warm_start and not old_policy and new_policy and not old_support and not new_support
                            and config['training'].get('allow_policy_getup_warm_start', False))
    if warm_start and old_policy != new_policy:
        if not policy_migration and not velocity_standing:
            raise ValueError('Full policy get-up requires explicit warm-start permission')
        if policy_migration:
            full_policy_getup_transfer(previous_model, model_path, previous_config, config)
    getup_migration = bool(warm_start and model_changed and config['training'].get('allow_getup_guard_warm_start', False))
    spring_migration = bool(warm_start and model_changed and (config['training'].get('allow_spring_limit_warm_start', False) or getup_migration))
    wheel_control_migration = bool(
        warm_start and config['training'].get('allow_wheel_control_warm_start', False)
        and previous_config.get('control', {}).get('wheel_kv') == 0.2
        and config.get('control', {}).get('wheel_kv') == 0.5
        and previous_config.get('control', {}).get('wheel_torque_limit_Nm') == 2.
        and config.get('control', {}).get('wheel_torque_limit_Nm') == 4.)
    if wheel_control_migration:
        old_control = dict(previous_config['control'])
        new_control = dict(config['control'])
        old_control.pop('wheel_kv')
        new_control.pop('wheel_kv')
        old_control.pop('wheel_torque_limit_Nm')
        new_control.pop('wheel_torque_limit_Nm')
        if old_control != new_control:
            raise ValueError('Wheel control migration only permits wheel_kv and wheel torque limit')
    if model_changed and not (support_migration or policy_migration):
        if wheel_control_migration:
            wheel_control_transfer(previous_model, model_path, previous_config)
        elif spring_migration:
            spring_limit_transfer(previous_model, model_path, previous_config, config, allow_getup_guards=getup_migration)
        else:
            raise ValueError('Policy transfer requires an existing checkpoint and the identical derived model.')
    from .observation_history import history_frames
    if (warm_start and history_frames(previous_config) != history_frames(config)
            and not config['training'].get('history_standing_actor', False)):
        raise ValueError('Changed observation history requires explicit history actor migration')
    keys=(('mass','physics','control','spring') if warm_start else
          ('mass','physics','control','spring','task','reward','push','getup',
           'observations','robustness','balance_recovery'))
    for key in keys:
        if key == 'control' and (support_migration or policy_migration or wheel_control_migration):
            continue
        if key == 'spring' and spring_migration:
            continue
        if key == 'physics' and getup_migration:
            target_physics = dict(config['physics'])
            target_physics.pop('joint_range_overrides_rad')
            if target_physics != previous_config['physics']:
                raise ValueError('Policy transfer configuration changed: physics')
            continue
        if previous_config.get(key)!=config.get(key):
            raise ValueError('Policy transfer configuration changed: '+key)
    if previous_config['experiment'].get('model_route')!=config['experiment'].get('model_route'):
        raise ValueError('Policy transfer model route changed')
    return checkpoint


def value_loss_overrides(settings):
    """Only override a checkpoint's critic weight when explicitly configured."""
    if 'vf_coef' not in settings:
        return {}
    value = float(settings['vf_coef'])
    if not np.isfinite(value) or value < 0:
        raise ValueError('vf_coef must be finite and nonnegative')
    return {'vf_coef': value}


def load_warm_start(checkpoint, vector, settings, device, seed):
    """Preserve the old controller when appending stationary feedback inputs."""
    overrides = dict(learning_rate=settings['learning_rate'], ent_coef=settings['ent_coef'], seed=seed,
                     n_steps=settings['n_steps'], batch_size=settings['batch_size'], n_epochs=settings['n_epochs'],
                     gamma=settings['gamma'], gae_lambda=settings['gae_lambda'],
                     clip_range=settings['clip_range'], max_grad_norm=settings['max_grad_norm'])
    overrides.update(value_loss_overrides(settings))
    if 'target_kl' in settings:
        overrides['target_kl']=settings['target_kl']
    algorithm=PPO
    estimator_settings = settings.get('velocity_estimator', {})
    if estimator_settings.get('enabled'):
        from .velocity_estimation import VelocityEstimationPPO
        algorithm = VelocityEstimationPPO
        enabled = settings.get('separate_actor_critic_gradient_clipping', False)
        if type(enabled) is not bool:
            raise ValueError('Separate actor/critic gradient clipping must be a boolean')
        overrides['separate_actor_critic_gradient_clipping'] = enabled
    if settings.get('demonstration_replay',{}).get('enabled'):
        if estimator_settings.get('enabled'):
            raise ValueError('Velocity estimation cannot use single-frame demonstration replay')
        from .ppo_replay import DemonstrationReplayPPO
        algorithm=DemonstrationReplayPPO
    previous = algorithm.load(str(checkpoint), device=device, **overrides)
    parent_steps = previous.num_timesteps
    old_shape, new_shape = previous.observation_space.shape, vector.observation_space.shape
    if estimator_settings.get('enabled'):
        from .velocity_estimation import VelocityEstimatedPolicy, copy_estimation_parent
        if isinstance(previous.policy, VelocityEstimatedPolicy):
            if old_shape != new_shape:
                raise ValueError('Estimator warm start must retain the history length')
        else:
            kwargs = dict(previous.policy_kwargs, estimator_learning_rate=estimator_settings.get('learning_rate', 1e-3),
                freeze_rising_actor=settings.get('freeze_rising_actor', True))
            model = algorithm(VelocityEstimatedPolicy, vector, device=device, policy_kwargs=kwargs, **overrides)
            copy_estimation_parent(previous.policy, model.policy)
            return model, parent_steps, True
    if settings.get('history_standing_actor', False) and not estimator_settings.get('enabled'):
        from .history_policy import HistoryStandingPolicy, copy_history_parent
        if settings.get('phase_split_standing_actor', False):
            raise ValueError('Choose history_standing_actor or phase_split_standing_actor')
        if isinstance(previous.policy, HistoryStandingPolicy):
            if old_shape != new_shape:
                raise ValueError('An existing history policy must retain its history length')
        else:
            if old_shape != (36,) or new_shape[0] <= 36 or new_shape[0] % 36:
                raise ValueError('History warm start requires a 36-input parent and stacked frames')
            kwargs = dict(previous.policy_kwargs, freeze_rising_actor=settings.get('freeze_rising_actor', True))
            model = algorithm(HistoryStandingPolicy, vector, device=device,
                policy_kwargs=kwargs, **overrides)
            copy_history_parent(previous.policy, model.policy)
            return model, parent_steps, True
    if settings.get('phase_split_standing_actor', False):
        from .phase_split_policy import PhaseSplitStandingPolicy, copy_parent_policy
        if old_shape != (36,) or new_shape != (36,) or previous.action_space != vector.action_space:
            raise ValueError('Phase split warm start requires unchanged 36-input/six-action spaces')
        if not isinstance(previous.policy, PhaseSplitStandingPolicy):
            model = algorithm(PhaseSplitStandingPolicy, vector, device=device,
                policy_kwargs=previous.policy_kwargs, **overrides)
            copy_parent_policy(previous.policy, model.policy)
            return model, parent_steps, False
    if old_shape == new_shape:
        # Loading with env also resets rollout buffers for a changed worker count.
        previous = algorithm.load(str(checkpoint), env=vector, device=device, **overrides)
        from .history_policy import HistoryStandingPolicy
        if isinstance(previous.policy, HistoryStandingPolicy):
            frozen = settings.get('freeze_rising_actor', previous.policy.freeze_rising_actor)
            standing_frozen = settings.get('freeze_standing_actor', False)
            if type(frozen) is not bool or type(standing_frozen) is not bool:
                raise ValueError('freeze_rising_actor and freeze_standing_actor must be booleans')
            previous.policy.freeze_rising_actor = frozen
            previous.policy.mlp_extractor.policy_net.rising_net.requires_grad_(not frozen)
            previous.policy.mlp_extractor.policy_net.rising_head.requires_grad_(not frozen)
            previous.policy.mlp_extractor.policy_net.standing_net.requires_grad_(not standing_frozen)
            previous.policy.mlp_extractor.policy_net.standing_head.requires_grad_(not standing_frozen)
            previous.policy_kwargs['freeze_rising_actor'] = frozen
        fixed_std = settings.get('freeze_action_std', False)
        if type(fixed_std) is not bool:
            raise ValueError('freeze_action_std must be a boolean')
        previous.policy.log_std.requires_grad_(not fixed_std)
        if (settings.get('allow_spring_limit_warm_start', False) or settings.get('allow_getup_guard_warm_start', False)
                or settings.get('reset_optimizer_on_warm_start',False)):
            previous.policy.optimizer = previous.policy.optimizer_class(previous.policy.parameters(),
                lr=settings['learning_rate'], **previous.policy.optimizer_kwargs)
        return previous, parent_steps, False
    if (old_shape, new_shape) not in (((24,), (30,)), ((30,), (36,))) or previous.action_space != vector.action_space:
        raise ValueError('Only explicit 24-to-30 or 30-to-36 observation extensions can be warm-started')
    model = algorithm('MlpPolicy', vector, device=device, policy_kwargs=previous.policy_kwargs, **overrides)
    old_state, new_state = previous.policy.state_dict(), model.policy.state_dict()
    expanded = {'mlp_extractor.policy_net.0.weight', 'mlp_extractor.value_net.0.weight'}
    if old_state.keys() != new_state.keys():
        raise ValueError('Warm-start policy architectures differ')
    for key, value in old_state.items():
        if key in expanded:
            if value.ndim != 2 or value.shape[1] != old_shape[0] or new_state[key].shape != (value.shape[0], new_shape[0]):
                raise ValueError('Unexpected first layer in stationary warm start: ' + key)
            new_state[key].zero_()
            new_state[key][:, :old_shape[0]].copy_(value)
        elif value.shape == new_state[key].shape:
            new_state[key].copy_(value)
        else:
            raise ValueError('Unexpected policy shape change: ' + key)
    model.policy.load_state_dict(new_state)
    return model, parent_steps, True


def action_std_vector(value):
    """Accept one standard deviation or one for each of the six motors."""
    if value is None:
        return None
    result=np.asarray(value,dtype=float)
    if result.shape==():
        result=np.full(6,float(result))
    if result.shape!=(6,) or not np.isfinite(result).all() or np.any(result<=0):
        raise ValueError('action_std must be positive and finite: scalar or six motor values')
    return result


def train(config_path,total_steps=None,num_envs=None,device="cpu",run_name=None,resume=None,warm_start=None):
    config_path=Path(config_path).resolve()
    config=load_config(config_path)
    if resume and warm_start:
        raise ValueError('Choose either --resume or --warm-start')
    settings=config["training"]
    total_steps=int(settings["total_timesteps"] if total_steps is None else total_steps)
    num_envs=int(settings["num_envs"] if num_envs is None else num_envs)
    physics_backend = config['experiment'].get('physics_backend', 'mujoco')
    if physics_backend not in ('mujoco', 'mujoco_warp'):
        raise ValueError('Unknown physics_backend: ' + str(physics_backend))
    maximum_envs = 64 if physics_backend == 'mujoco_warp' else 14
    if total_steps<=0 or not 1 <= num_envs <= maximum_envs:
        raise ValueError(f'Require positive steps and 1..{maximum_envs} environments')
    if settings.get('velocity_estimator', {}).get('enabled'):
        if not settings.get('history_standing_actor') or not (warm_start or resume):
            raise ValueError('Velocity estimation requires a history policy and a pretrained/warm-start parent')
    action_std=settings.get('action_std')
    motor_std=action_std_vector(action_std)
    loss_overrides=value_loss_overrides(settings)
    model_dir=ROOT/config["experiment"]["output_directory"]
    model_path=model_dir/"robot_experiment.xml"
    if not model_path.is_file() or not (model_dir/"smoke_report.json").is_file():
        raise RuntimeError("Run prepare_mujoco_experiment.py and smoke_mujoco.py first")
    smoke=json.loads((model_dir/"smoke_report.json").read_text())
    if not smoke.get("passed"):
        raise RuntimeError("Model has not passed smoke checks")
    for key,path in (("config_sha256",config_path),("model_sha256",model_path),
                     ("env_sha256",ROOT/"source/climber_legged_rl/mujoco_env.py")):
        if smoke.get(key)!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError("Smoke report is stale: "+key+". Rerun smoke_mujoco.py.")
    if config['experiment'].get('model_route') == 'cad_dynamics':
        from .cad_dynamics import validate_artifacts
        validate_artifacts(config_path)
    if physics_backend == 'mujoco_warp':
        from .warp_backend import validate_gpu_report
        validate_gpu_report(config_path)
    # Validate before spawning workers or creating a run directory.
    if resume:
        resume=validate_policy_transfer(resume,model_path,config)
    if warm_start:
        warm_start=validate_policy_transfer(warm_start,model_path,config,warm_start=True)
    if settings.get('standing_rehearsal', {}).get('enabled'):
        if not settings.get('velocity_estimator', {}).get('enabled'):
            raise ValueError('Standing rehearsal requires the velocity-estimated history policy')
        from .standing_rehearsal import load_standing_rehearsal
        load_standing_rehearsal(settings['standing_rehearsal'], config, model_path)
    replay_settings=settings.get('demonstration_replay',{})
    if settings.get('history_standing_actor', False):
        from .observation_history import history_frames
        if (not (warm_start or resume) or history_frames(config) <= 1
                or not config.get('getup', {}).get('policy_control', {}).get('enabled')
                or config['getup']['policy_control'].get('phase_seconds') is None
                or settings.get('phase_split_standing_actor', False)):
            raise ValueError('History standing requires a full-policy parent, history frames and phase observation')
        if replay_settings.get('enabled'):
            raise ValueError('Single-frame replay cannot supply real history; disable it or collect new trajectories')
    if settings.get('phase_split_standing_actor', False):
        if (not warm_start or not config.get('getup', {}).get('policy_control', {}).get('enabled')
                or config['getup']['policy_control'].get('phase_seconds') is None):
            raise ValueError('Phase split requires a full-policy warm start with the established phase observation')
    replay_arrays=replay_provenance=None
    if replay_settings.get('enabled'):
        if not warm_start or resume or not config.get('getup',{}).get('policy_control',{}).get('enabled'):
            raise ValueError('Demonstration replay requires a full-policy warm start')
        from .ppo_replay import load_demonstration_replay
        replay_arrays,replay_provenance=load_demonstration_replay(replay_settings,config,model_path)
    run_name=run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if Path(run_name).name!=run_name:
        raise ValueError("Run name must be a single directory name")
    run_dir=ROOT/"outputs"/"mujoco_runs"/run_name
    run_dir.mkdir(parents=True,exist_ok=False)
    (run_dir/"checkpoints").mkdir()
    (run_dir/"config.yaml").write_text(yaml.safe_dump(config,allow_unicode=True),encoding="utf-8")
    (run_dir/"model_report.json").write_bytes((model_dir/"model_report.json").read_bytes())
    (run_dir/"robot_experiment.xml").write_bytes(model_path.read_bytes())
    if config.get('getup', {}).get('enabled'):
        (run_dir/'getup_poses.json').write_bytes((model_dir/'getup_poses.json').read_bytes())
        if config['getup'].get('reset_curriculum',{}).get('enabled'):
            (run_dir/'getup_curriculum_poses.json').write_bytes((model_dir/'getup_curriculum_poses.json').read_bytes())
    seed=settings["seed"]
    torch.set_num_threads(1)
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    try:
        git_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        git_sha="uncommitted_uploaded_working_tree"
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in (ROOT/"source",ROOT/"scripts") for p in folder.rglob("*.py")}
    metadata={"simulation_only":True,"backend":"MuJoCo_CPU_physics_SB3_PPO",
              'model_route':config['experiment'].get('model_route','legacy_approximate'),
              "device":device,"cuda_available":torch.cuda.is_available(),"torch":torch.__version__,
              "python":platform.python_version(),"pid":os.getpid(),"git_sha":git_sha,
              "source_sha256":sources,"model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "requested_timesteps":total_steps,"num_envs":num_envs}
    if physics_backend == 'mujoco_warp':
        metadata.update(backend='MuJoCo_Warp_CUDA_physics_SB3_PPO',
            physics_device=config.get('gpu_physics', {}).get('device', 'cuda:0'),
            host_task_bookkeeping=True, independent_evaluation_backend='MuJoCo_CPU')
    (run_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("RUN_DIRECTORY="+str(run_dir),flush=True)
    print('MODEL_ROUTE='+metadata['model_route']+'; SIMULATION ONLY',flush=True)
    factories=[_factory(model_path,config_path,seed+i,str(run_dir/f"env_{i}")) for i in range(num_envs)]
    if physics_backend == 'mujoco_warp':
        from .warp_backend import WarpVecEnv
        vector = WarpVecEnv(factories, config.get('gpu_physics', {}))
    else:
        vector=SubprocVecEnv(factories,start_method="spawn") if num_envs>1 else DummyVecEnv(factories)
    vector=VecCheckNan(vector,raise_exception=True)
    if resume:
        algorithm = PPO
        if settings.get('velocity_estimator', {}).get('enabled'):
            from .velocity_estimation import VelocityEstimationPPO
            algorithm = VelocityEstimationPPO
        model=algorithm.load(str(resume),env=vector,device=device)
        metadata["resume_checkpoint"]=str(resume)
        metadata["resume_timesteps"]=model.num_timesteps
        (run_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    elif warm_start:
        model,parent_steps,expanded=load_warm_start(warm_start,vector,settings,device,seed)
        if expanded:
            old_dimensions = PPO.load(str(warm_start), device='cpu').observation_space.shape[0]
            metadata['observation_extension']={'old_dimensions':old_dimensions,'new_dimensions':vector.observation_space.shape[0],
                                               'new_input_weights':'zero','optimizer_reset':True}
            if settings.get('history_standing_actor', False):
                metadata['observation_extension'].update(parent_input_frame='latest',
                    rising_reads_current_frame_only=True, rising_branch_frozen=model.policy.freeze_rising_actor)
            if settings.get('velocity_estimator', {}).get('enabled'):
                metadata['observation_extension'].update(kind='estimated_velocity_actor_privileged_critic',
                    actor_initialization_exact=False, note='Control weights copied; velocity estimator requires training')
        if action_std is not None:
            with torch.no_grad():
                model.policy.log_std.copy_(torch.as_tensor(np.log(motor_std),device=model.device,dtype=model.policy.log_std.dtype))
            # Old distribution moments do not describe the explicitly reset std.
            model.policy.optimizer.state.pop(model.policy.log_std,None)
            metadata['warm_start_action_std_reset']=action_std
        metadata['warm_start_checkpoint']=str(warm_start)
        metadata['parent_timesteps']=parent_steps
        metadata['warm_start_checkpoint_sha256']=hashlib.sha256(warm_start.read_bytes()).hexdigest()
        metadata['training_step_counter_resets']=True
        metadata['warm_start_optimizer_reset']=bool(expanded or settings.get('reset_optimizer_on_warm_start',False)
            or settings.get('allow_spring_limit_warm_start',False) or settings.get('allow_getup_guard_warm_start',False))
        if settings.get('phase_split_standing_actor', False):
            metadata['phase_split_actor'] = dict(rising_branch_frozen=True,
                standing_branch_trainable=True, phase_observation_index=35, threshold=.95,
                runtime_reference_assistance=False, checkpoint_contains_both_learned_branches=True)
        if config.get('getup',{}).get('policy_control',{}).get('enabled'):
            metadata['full_policy_getup_transfer'] = dict(
                rising_action_map='current_joint_position_plus_actor_increment',
                standing_action_map='historical_nominal_plus_actor_residual',
                policy_controls_all_motors=True,
                motor_envelope=config['getup']['policy_control'],
                note='One actor outputs all actions from the first step; no reference assistance')
        elif config.get('getup',{}).get('support_control',{}).get('enabled'):
            metadata['supported_getup_transfer'] = dict(
                rising_action_map='deterministic_reference_assistance',
                standing_action_map='historical_nominal_plus_residual',
                handoff_seconds=config['getup']['support_control']['handoff_seconds'],
                note='Train handoff and standing; assistance ignores actor actions before handoff')
        elif config.get('getup',{}).get('dynamic_control',{}).get('enabled'):
            metadata['dynamic_getup_transfer']=dict(
                rising_action_map='current_joint_position_plus_action_times_incremental_scale',
                standing_action_map='historical_nominal_plus_residual',
                note='Initialization only; old get-up behavior is not preserved by the new rising controller')
        if settings.get('allow_velocity_standing_warm_start', False):
            metadata['velocity_standing_transfer'] = dict(
                parent_control='policy_getup', standing_resets_only=True,
                velocity_tracking=bool(config.get('task', {}).get('velocity_tracking')),
                observation_layout=config.get('task', {}).get('velocity_observation_layout'),
                note='Get-up resets suspended; the 36-input standing layout is preserved')
        if settings.get('allow_wheel_control_warm_start', False):
            metadata['wheel_control_transfer'] = dict(
                parent_wheel_kv=0.2, wheel_kv=config['control']['wheel_kv'],
                parent_wheel_torque_limit_Nm=2., wheel_torque_limit_Nm=config['control']['wheel_torque_limit_Nm'],
                note='Stronger wheel servo and torque envelope; leg control unchanged')
        if settings.get('allow_spring_limit_warm_start', False) or settings.get('allow_getup_guard_warm_start', False):
            metadata['spring_limit_transfer'] = dict(
                kind='gas_slider_limits_and_explicit_getup_branch_guards' if settings.get('allow_getup_guard_warm_start') else 'only_add_two_gas_slider_travel_limits',
                joint_range_overrides_rad=config['physics'].get('joint_range_overrides_rad', {}),
                eye_center_limits_m=config['spring']['eye_center_limits_m'], optimizer_reset=True,
                parent_model_sha256=hashlib.sha256(((warm_start.parent.parent if warm_start.parent.name in
                    ('best','checkpoints') else warm_start.parent)/'robot_experiment.xml').read_bytes()).hexdigest())
        model.num_timesteps=0
        model._n_updates=0
        model.set_random_seed(seed)
        (run_dir/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    else:
        model=PPO("MlpPolicy",vector,device=device,seed=seed,n_steps=settings["n_steps"],
                  batch_size=min(settings["batch_size"],settings["n_steps"]*num_envs),
                  n_epochs=settings["n_epochs"],learning_rate=settings["learning_rate"],gamma=settings["gamma"],
                  gae_lambda=settings["gae_lambda"],clip_range=settings["clip_range"],ent_coef=settings["ent_coef"],
                  max_grad_norm=settings["max_grad_norm"],verbose=1,**loss_overrides,
                  policy_kwargs=dict(net_arch=dict(pi=settings["actor_hidden_dims"],vf=settings["critic_hidden_dims"]),activation_fn=torch.nn.ReLU,
                                     log_std_init=float(np.log(motor_std[0])) if motor_std is not None else 0.))
        if motor_std is not None:
            with torch.no_grad():
                model.policy.log_std.copy_(torch.as_tensor(np.log(motor_std),device=model.device,dtype=model.policy.log_std.dtype))
    metadata['effective_vf_coef']=model.vf_coef
    if settings.get('velocity_estimator', {}).get('enabled'):
        model.configure_estimator(settings['velocity_estimator'])
        model.configure_gradient_clipping(settings.get('separate_actor_critic_gradient_clipping', False))
        metadata['separate_actor_critic_gradient_clipping'] = model.separate_actor_critic_gradient_clipping
        rehearsal = settings.get('standing_rehearsal', {})
        provenance = model.configure_standing_rehearsal(rehearsal, config, model_path, seed,
                                                        restore_optimizer=bool(resume))
        if provenance is not None:
            metadata['standing_rehearsal'] = dict(settings=rehearsal, **provenance)
        metadata['velocity_estimator'] = dict(actor_uses_estimated_velocity=True,
            privileged_label_dimensions=3, target_unit='m/s', **settings['velocity_estimator'])
    from .observation_history import history_frames
    metadata['observation_history'] = dict(frames=history_frames(config), order='oldest_to_newest',
        reset_fill='repeat_first_frame', input_dimensions=vector.observation_space.shape[0])
    if settings.get('history_standing_actor', False):
        metadata['history_actor'] = dict(rising_branch_frozen=model.policy.freeze_rising_actor,
            standing_branch_trainable=not settings.get('freeze_standing_actor', False),
            current_frame_dimensions=36, phase_index=-1, threshold=.95)
    (run_dir/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    if replay_arrays is not None:
        model.configure_replay(replay_arrays,replay_settings,seed,config_path=config_path,model_path=model_path)
        metadata['demonstration_replay']={**replay_provenance,'settings':replay_settings}
        (run_dir/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    model.set_logger(configure(str(run_dir),["stdout","csv","tensorboard"]))
    eval_env=VecCheckNan(DummyVecEnv([_factory(model_path,config_path,30000,None)]),raise_exception=True)
    evaluation_callback = (CombinedEvalCallback(run_dir,config_path,model_path,
                              frequency=settings.get('evaluation_steps',50000),
                              evaluate_initial=settings.get('evaluate_initial_policy',False),
                              all_poses=settings.get('evaluation_all_poses',False),
                              standing_episodes=settings.get('evaluation_standing_episodes',5)) if config.get('getup',{}).get('enabled') else
        EvalCallback(eval_env,best_model_save_path=str(run_dir/"best"),log_path=str(run_dir/"eval"),
                     eval_freq=max(50000//num_envs,1),n_eval_episodes=5,deterministic=True))
    if config.get('task',{}).get('velocity_tracking'):
        from .velocity_training import VelocityEvalCallback
        evaluation_callback=VelocityEvalCallback(run_dir,config_path,model_path,
            frequency=settings.get('evaluation_steps',25000),
            evaluate_initial=settings.get('evaluate_initial_policy',True))
    callbacks=CallbackList([CheckpointCallback(save_freq=max(settings["checkpoint_steps"]//num_envs,1),
                    save_path=str(run_dir/"checkpoints"),name_prefix="climber"),StatusCallback(run_dir,total_steps,model.num_timesteps),
                    evaluation_callback])
    try:
        velocity_task=config.get('task',{}).get('velocity_tracking',False)
        baseline=evaluate(None,config_path,model_path,episodes=7 if velocity_task else 5)
        (run_dir/"baseline_zero.json").write_text(json.dumps(baseline,indent=2),encoding="utf-8")
        model.learn(total_timesteps=total_steps,callback=callbacks,log_interval=1,reset_num_timesteps=not bool(resume))
        model.save(str(run_dir/"final_policy"))
        (run_dir/'status.json').write_text(json.dumps(dict(state='evaluating',simulation_only=True,
            timesteps=model.num_timesteps,pid=os.getpid(),training_remaining_seconds=0),indent=2))
        final_groups = None
        if isinstance(evaluation_callback, CombinedEvalCallback) and evaluation_callback.evaluator.workers > 1:
            modes=('folded','standing','curriculum') if config['getup'].get('reset_curriculum',{}).get('enabled') else ('folded','standing')
            common=dict(checkpoint=run_dir/'final_policy.zip',config_path=config_path,model_path=model_path,episodes=10)
            jobs={'mixed':dict(common,seed=10000)}
            jobs.update({mode:dict(common,seed=51000,reset_mode=mode) for mode in modes})
            final_groups=evaluation_callback.evaluator.run(jobs)
        result=(final_groups['mixed'] if final_groups is not None else
                evaluate(run_dir/"final_policy.zip",config_path,model_path,episodes=21 if velocity_task else 10))
        (run_dir/"evaluation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        if config.get('getup',{}).get('enabled'):
            modes=('folded','standing','curriculum') if config['getup'].get('reset_curriculum',{}).get('enabled') else ('folded','standing')
            forced=({mode:final_groups[mode] for mode in modes} if final_groups is not None else
                    {mode:evaluate(run_dir/'final_policy.zip',config_path,model_path,episodes=10,seed=51000,reset_mode=mode)
                    for mode in modes})
            (run_dir/'evaluation_by_reset.json').write_text(json.dumps(forced,indent=2))
        status={"state":"completed","simulation_only":True,"timesteps":model.num_timesteps,
                "evaluation_mean_seconds":result["mean_episode_seconds"],"timeout_fraction":result["timeout_fraction"]}
        status['stationary_success_fraction']=result['stationary_success_fraction']
        if 'velocity_success_fraction' in result:
            status['velocity_success_fraction']=result['velocity_success_fraction']
            status['stop_success_fraction']=result['stop_success_fraction']
        if 'push_recovery_success_fraction' in result:
            status['push_recovery_success_fraction']=result['push_recovery_success_fraction']
        if 'combined_success_fraction' in result:
            status.update({key:result[key] for key in ('combined_success_fraction','getup_episode_count','folded_success_fraction','standing_success_fraction')})
            status.update({key:result[key] for key in ('curriculum_episode_count','curriculum_success_fraction')})
        (run_dir/"status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        print(json.dumps(status),flush=True)
    except BaseException as exc:
        model.save(str(run_dir/"interrupted_policy"))
        (run_dir/"status.json").write_text(json.dumps({"state":"failed","error":repr(exc),"timesteps":model.num_timesteps},indent=2),encoding="utf-8")
        raise
    finally:
        if isinstance(evaluation_callback, CombinedEvalCallback):
            evaluation_callback.evaluator.close()
        vector.close()
        eval_env.close()
    return run_dir

"""Standalone supervised warm-up with explicitly validated switched trajectories."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'scripts')]
import numpy as np
import torch
from climber_legged_rl.velocity_estimation import VelocityEstimationPPO
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.standing_rehearsal import load_standing_rehearsal
from climber_legged_rl.cad_dynamics import validate_artifacts
from round18_takeover_data import load_successful_takeovers,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--stamp',required=True)
    args=parser.parse_args(); torch.set_num_threads(1)
    cfg=ROOT/'configs/robot/cad_balance_estimator_round18a.yaml'
    config=load_config(cfg); validate_artifacts(cfg)
    xml=ROOT/config['experiment']['output_directory']/'robot_experiment.xml'
    source=ROOT/config['training']['standing_rehearsal']['dataset']
    (x,y,_),_=load_standing_rehearsal(config['training']['standing_rehearsal'],config,xml)
    source_report=json.loads(source.with_suffix('.json').read_text())
    stable=[]; recovery=[]; offset=0
    for item in source_report['outcomes']:
        n=item['retained_samples']
        (stable if item['role']=='stable' else recovery).extend(range(offset,offset+n)); offset+=n
    assert offset==len(x)
    accepted,protocol=load_successful_takeovers(ROOT/'outputs/round18_takeovers_20261105/report.json',
        ROOT/'configs/robot/cad_balance_estimator_round15.yaml',xml)
    # Keep one earliest and one latest successful switch per difficult push case.
    selected=[]
    for seed in (261106,261704):
        group=sorted([r for r in accepted if r[0]['seed']==seed],key=lambda r:r[0]['takeover_seconds'])
        if not group: raise ValueError('Missing successful push takeover data')
        selected.extend([group[0],group[-1]])
    new_x=np.concatenate([r[1] for r in selected]); new_y=np.concatenate([r[2] for r in selected])
    new_prob=np.concatenate([np.full(len(r[1]),1/len(selected)/len(r[1])) for r in selected])
    parent=ROOT/'outputs/mujoco_runs/cad_balance_estimator_round14_2080_20261105_151831/final_policy.zip'
    baseline=VelocityEstimationPPO.load(parent,device='cpu')
    env=make_mujoco_env(xml,cfg); env.unwrapped.push.config=dict(env.unwrapped.push.config,episode_probability=0.)
    anchor_x=[]; anchor_y=[]
    try:
        for seed in (275005,275006):
            obs,_=env.reset(seed=seed,options={'reset_mode':'standing'})
            while True:
                action=baseline.predict(obs,deterministic=True)[0]
                anchor_x.append(obs[:180].copy()); anchor_y.append(np.clip(action,-1,1))
                obs,_,done,truncated,info=env.step(action)
                if done or truncated: break
            if not info['combined_success']: raise RuntimeError('Baseline anchor failed')
    finally: env.close()
    anchor_x=np.asarray(anchor_x,np.float32); anchor_y=np.asarray(anchor_y,np.float32)
    out=ROOT/f'outputs/round18_warmup_{args.stamp}'; out.mkdir(exist_ok=False)
    np.savez_compressed(out/'baseline_stable_anchors.npz',observations=anchor_x,actions=anchor_y)
    shared_x=np.concatenate([x[stable],anchor_x,x[recovery]])
    shared_y=np.concatenate([y[stable],anchor_y,y[recovery]])
    for trial in ('a','b'):
        torch.manual_seed(147); rng=np.random.default_rng(147)
        model=VelocityEstimationPPO.load(parent,device='cpu')
        actor=model.policy.mlp_extractor.policy_net
        original={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
        trainable=[p for name,p in actor.named_parameters() if name.startswith(('standing_net.','standing_head.'))]
        for p in model.policy.parameters(): p.requires_grad_(False)
        for p in trainable: p.requires_grad_(True)
        optimizer=torch.optim.Adam(trainable,lr=1e-6)
        mass=.1 if trial=='a' else .05
        weights=np.r_[np.full(len(stable),.7/len(stable)),np.full(len(anchor_x),.2/len(anchor_x)),
                      np.full(len(recovery),mass/len(recovery))]
        data_x=shared_x; data_y=shared_y
        if trial=='b':
            data_x=np.concatenate([data_x,new_x]); data_y=np.concatenate([data_y,new_y])
            weights=np.r_[weights,.05*new_prob]
        assert np.isclose(weights.sum(),1.)
        tx=torch.from_numpy(data_x); ty=torch.from_numpy(data_y)
        scales=torch.tensor([.025]*4+[.25]*2)
        losses=[]
        for update in range(300):
            ids=torch.from_numpy(rng.choice(len(tx),256,p=weights))
            predicted=actor(tx[ids]).clamp(-1,1)
            loss=((predicted-ty[ids])/scales).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach()))
        changed=[]
        for name,value in model.policy.state_dict().items():
            if name.startswith(('mlp_extractor.policy_net.standing_net.','mlp_extractor.policy_net.standing_head.')):
                if not torch.equal(value,original[name]): changed.append(name)
            elif not torch.equal(value,original[name]):
                raise RuntimeError('Frozen weight changed: '+name)
        assert changed
        # Restore the original gradient flags needed by normal PPO loading.
        for name,p in model.policy.named_parameters():
            p.requires_grad_(not name.startswith(('mlp_extractor.policy_net.rising_',
                                                'mlp_extractor.policy_net.velocity_estimator.')))
        folder=ROOT/f'outputs/mujoco_runs/cad_balance_estimator_round18{trial}_warmup_{args.stamp}'
        folder.mkdir(exist_ok=False)
        trial_cfg=ROOT/f'configs/robot/cad_balance_estimator_round18{trial}.yaml'
        (folder/'config.yaml').write_bytes(trial_cfg.read_bytes())
        (folder/'robot_experiment.xml').write_bytes(xml.read_bytes())
        model.save(folder/'final_policy.zip')
        report=dict(simulation_only=True,training='offline_supervised_warmup',trial=trial,updates=300,
            learning_rate=1e-6,seed=147,stable_mass=.9,old_recovery_mass=mass,
            takeover_mass=0. if trial=='a' else .05,physics_and_reward_unchanged=True,
            original_rehearsal_loader_unchanged=True,nonstanding_weights_identical=True,
            parent=str(parent),parent_sha256=digest(parent),checkpoint_sha256=digest(folder/'final_policy.zip'),
            takeover_report_sha256=digest(ROOT/'outputs/round18_takeovers_20261105/report.json'),
            source_rehearsal_sha256=digest(source),anchor_seeds=[275005,275006],
            anchor_sha256=digest(out/'baseline_stable_anchors.npz'),
            selected_takeovers=[dict(seed=r[0]['seed'],takeover_seconds=r[0]['takeover_seconds'],
                dataset_sha256=r[0]['dataset_sha256']) for r in selected] if trial=='b' else [],
            first50_mean_loss=float(np.mean(losses[:50])),last50_mean_loss=float(np.mean(losses[-50:])))
        (folder/'warmup_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__': main()

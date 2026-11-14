"""Numerical contracts for the isolated GPU pilot, using exported native fixtures."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from supermimic_world_model import Actor, WorldModel, ref_reward, episode_windows


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);args=p.parse_args()
    folder=args.folder;g=json.loads((folder/'schema.json').read_text());torch.set_num_threads(1)
    torch.manual_seed(42)
    weights=dict(np.load(folder/'parent_actor.npz'));actor=Actor(weights)
    parity=np.load(folder/'actor_parity.npz')
    with torch.no_grad():error=float(abs(actor(torch.tensor(parity['observations'])).numpy()-parity['actions']).max())
    assert error<1e-6,error
    report=json.loads((folder/'dataset.json').read_text());rows=report['episodes']
    train=[r for r in rows if r['role']=='train'];val=[r for r in rows if r['role']=='validation']
    assert not set(r['seed'] for r in train)&set(r['seed'] for r in val)
    arrays=episode_windows(train[:2],8)
    assert arrays[0].shape[1]==9 and arrays[1].shape[1]==8
    for row in rows:
        a=np.load(row['path']);assert len(a['states'])==len(a['actions'])+1
        assert not a['dones'][:-1].any()
        assert np.isfinite(a['states']).all() and np.max(abs(a['actions']))<=1
    sample=torch.tensor(arrays[0][:4,0])
    mean=sample.mean(0).numpy();scale=np.ones(g['dimensions'],np.float32)
    model=WorldModel(g,mean,scale,np.full(len(g['output_indices']),.001,np.float32))
    with torch.no_grad():
        model.net[-1].weight.normal_(0,.001)
        model.net[-1].bias[-1]=-6.
    model.requires_grad_(False)
    actor.rising_net.requires_grad_(True);actor.rising_head.requires_grad_(True)
    action=actor(sample[:,g['history']:g['history']+180])
    nxt,reward,done=model(sample,action)
    torch.testing.assert_close(nxt[:,g['history']:g['latest']],sample[:,g['history']+36:g['latest']+36])
    torch.testing.assert_close(nxt[:,g['controller']],sample[:,g['controller']]+.01)
    torch.testing.assert_close(nxt[:,g['controller']+3:],sample[:,g['controller']+3:])
    torch.testing.assert_close(nxt[:,g['latest']+18:g['latest']+24],action)
    loss=nxt[:,:3].sum()+reward.sum()
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in actor.rising_head.parameters())>0
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in actor.standing_net.parameters())
    # Compare a model action Jacobian to centered finite differences.
    model=model.double();x=sample[:1].double();a=action[:1].detach().double().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda z:model(x,z)[1],(a,),eps=1e-6,atol=1e-5,rtol=1e-3)
    refs={k:torch.tensor(v) for k,v in dict(np.load(folder/'references.npz')).items()}
    state=torch.tensor(np.load(train[0]['path'])['states'][1:5])
    off=torch.full((len(state),),train[0]['offset_m'])
    actual=ref_reward(state,off,refs,g).numpy()
    l=g['latest'];m=g['metrics'];c=g['controller']
    features=np.concatenate([state[:,l+3:l+6],state[:,l+8:l+12],state[:,m:m+2],state[:,m+5:m+6],state[:,m+8:m+9]],axis=-1)
    idx=int(np.argmin(abs(refs['offsets'].numpy()-train[0]['offset_m'])))
    times=np.rint(state[:,c].numpy()/.01).astype(int)
    target=refs['features'].numpy()[idx,times]
    expected=3*np.exp(-np.mean(((features-target)/refs['scales'].numpy())**2,axis=-1))*(1-state[:,c+1].numpy())
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    result=dict(passed=True,native_actor_parity_max_error=error,episode_boundaries=True,
        train_validation_disjoint=True,history_and_controller_bookkeeping=True,
        frozen_model_transmits_policy_gradient=True,finite_difference_action_gradient=True,
        frozen_standing_and_estimator=True,reference_reward_numpy_torch_parity=True)
    (folder/'numerical_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

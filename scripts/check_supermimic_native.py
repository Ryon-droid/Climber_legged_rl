"""Check that reference rewards never alter native mechanics or actor inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from supermimic_pilot import make_env,ReferenceEnv,PARENT,sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1)
    a=make_env();b=ReferenceEnv(make_env(),args.folder/'references.npz')
    policy=PPO.load(PARENT,device='cpu')
    try:
        x,_=a.reset(seed=510099,options={'reset_mode':'folded'})
        y,_=b.reset(seed=510099,options={'reset_mode':'folded'})
        np.testing.assert_array_equal(x,y)
        for _ in range(120):
            action=policy.predict(x,deterministic=True)[0]
            x,r,d,t,info=a.step(action)
            y,rr,dd,tt,other=b.step(action)
            np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(a.data.qpos,b.unwrapped.data.qpos)
            np.testing.assert_array_equal(a.data.qvel,b.unwrapped.data.qvel)
            np.testing.assert_allclose(rr-r,other['reference_reward'],rtol=1e-7,atol=1e-7)
            assert (d,t,info['getup_acquired'])==(dd,tt,other['getup_acquired'])
    finally:a.close();b.close()
    provenance=json.loads((args.folder/'reference_provenance.json').read_text())
    assert all(r['role']=='train' and r['success'] and r['noise']==0 for r in provenance['sources'])
    result=dict(passed=True,reference_changes_only_training_reward=True,
        physics_and_observations_bit_identical_120_steps=True,
        reference_sources_unassisted_successful_training_only=True,
        parent_sha256=sha(PARENT))
    write_json(args.folder/'native_checks.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()

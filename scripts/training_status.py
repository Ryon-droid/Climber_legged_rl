"""Print the active CAD training process, checkpoints and latest evaluation."""
import json
from pathlib import Path
import os

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    active=json.loads((ROOT/'outputs/cad_dynamics/active_training.json').read_text())
    folder=Path(active['run_directory'])
    status_path=folder/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {'state':'initializing'}
    try:
        os.kill(active['pid'],0)
        alive=True
        proc=Path('/proc')/str(active['pid'])/'stat'
        if proc.exists() and proc.read_text().split()[2]=='Z':alive=False
    except ProcessLookupError:alive=False
    checkpoints=sorted((folder/'checkpoints').glob('*.zip'),key=lambda p:p.stat().st_mtime)
    result={'run_name':active['run_name'],'process_alive':alive,'pid':active['pid'],
            'status':status,'run_directory':str(folder),'log':active['log'],
            'checkpoint_count':len(checkpoints),'latest_checkpoint':str(checkpoints[-1]) if checkpoints else None}
    eval_path=folder/'eval/evaluations.npz'
    if eval_path.exists():
        import numpy as np
        try:
            with np.load(eval_path) as evaluation:
                result['latest_evaluation']={'timesteps':int(evaluation['timesteps'][-1]),
                    'mean_reward':float(evaluation['results'][-1].mean()),
                    'mean_episode_steps':float(evaluation['ep_lengths'][-1].mean())}
        except (OSError,ValueError,EOFError):result['latest_evaluation']='being written; retry'
    final=folder/'evaluation.json'
    if final.exists():
        evaluation=json.loads(final.read_text())
        result['final_evaluation']={k:evaluation[k] for k in ('mean_episode_seconds','timeout_fraction','mean_reward')}
    print(json.dumps(result,ensure_ascii=False,indent=2))

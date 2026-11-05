"""Parallelize independent seeded episodes without changing simulation or scoring."""
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import multiprocessing

import numpy as np
import torch


def deterministic_actor_digest(policy):
    """Include every policy tensor except the separate critic and stochastic std."""
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        if name == 'log_std' or 'value_net' in name or name.startswith('vf_features_extractor.'):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _initialize_worker():
    torch.set_num_threads(1)


def _evaluate_job(job):
    from .mujoco_training import evaluate
    return evaluate(**job)


def merge_evaluations(parts):
    """Reassemble ordered episodes and recompute the original summary fields."""
    result = copy.deepcopy(parts[0])
    records = [record for part in parts for record in part['episodes']]
    result['episodes'] = records
    for field, source in [('mean_reward', 'reward'), ('mean_episode_seconds', 'seconds'),
                          ('stationary_success_fraction', 'stationary_success')]:
        result[field] = float(np.mean([r[source] for r in records]))
    result['timeout_fraction'] = float(np.mean([r['reason'] == 'timeout' for r in records]))
    if 'combined_success_fraction' in result:
        result['combined_success_fraction'] = float(np.mean([r['combined_success'] for r in records]))
        for mode, count_key in [('folded', 'getup_episode_count'), ('curriculum', 'curriculum_episode_count')]:
            result[count_key] = sum(r['getup_metrics']['reset_mode'] == mode for r in records)
        for mode in ('folded', 'standing', 'curriculum'):
            group = [r for r in records if r['getup_metrics']['reset_mode'] == mode]
            result[mode+'_success_fraction'] = float(np.mean([r['combined_success'] for r in group])) if group else None
    if 'push_recovery_success_fraction' in result:
        result['push_recovery_success_fraction'] = float(np.mean([r['push_recovery_success'] for r in records]))
        pushed = [r for r in records if r['push_metrics']['push_count'] > 0]
        result['push_episode_count'] = len(pushed)
        result['pushed_episode_success_fraction'] = float(np.mean([r['push_recovery_success'] for r in pushed])) if pushed else None
    return result


class ParallelEvaluation:
    """Keep workers alive between evaluations; retain original episode seed order."""
    def __init__(self, workers):
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError('Evaluation workers must be an integer in [1,8]')
        self.workers = workers
        self.pool = None

    def run(self, jobs):
        if self.workers == 1:
            return {key: _evaluate_job(job) for key, job in jobs.items()}
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=self.workers,
                mp_context=multiprocessing.get_context('spawn'), initializer=_initialize_worker)
        pending = {}
        for key, job in jobs.items():
            count = job.get('episodes', 10)
            size = max(1, (count+self.workers-1)//self.workers)
            chunks = []
            for offset in range(0, count, size):
                chunk = dict(job, episodes=min(size, count-offset), seed=job.get('seed', 10000)+offset)
                chunks.append(self.pool.submit(_evaluate_job, chunk))
            pending[key] = chunks
        return {key: merge_evaluations([future.result() for future in chunks])
                for key, chunks in pending.items()}

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

"""Explicit loader for successful baseline-to-teacher simulation takeovers.

This format is separate from pure-policy rehearsal. It never relabels switched
episodes as a single-policy success or accepts unsuccessful teacher queries.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_successful_takeovers(report_path, config_path, model_path):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    if (report.get('schema') != 'validated_policy_takeovers_v1'
            or report.get('full_episode_success_required') is not True
            or report.get('state_reset_at_takeover') is not False
            or report.get('history_reset_at_takeover') is not False
            or report.get('physics_assistance') is not False
            or report.get('runtime_controller') != 'baseline_then_recovery_teacher'):
        raise ValueError('Invalid takeover protocol')
    if report['config_sha256'] != digest(config_path) or report['model_sha256'] != digest(model_path):
        raise ValueError('Takeover collection config/model mismatch')
    for entry in report['checkpoints'].values():
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Takeover policy provenance mismatch')
    accepted = []
    for outcome in report['outcomes']:
        if 'dataset' not in outcome:
            continue
        if (outcome['combined_success'] is not True or outcome['takeover_seconds'] <= 0
                or outcome['seconds'] < 19.99 or outcome['reason'] != 'timeout'
                or outcome['push_metrics']['push_count'] != outcome['push_metrics']['recovered_push_count']):
            raise ValueError('Unsuccessful trajectory has a retained dataset')
        path = Path(outcome['dataset']).resolve()
        if path.parent != report_path.parent or digest(path) != outcome['dataset_sha256']:
            raise ValueError('Takeover dataset location/hash mismatch')
        with np.load(path,allow_pickle=False) as data:
            x, y = data['observations'].copy(), data['actions'].copy()
        if (x.shape != (outcome['retained_samples'],180) or y.shape != (len(x),6)
                or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(np.abs(y)>1.000001) or np.any(x[:,-1]<=.95)
                or np.any(x.reshape(-1,5,36)[:,:,28:30]!=0)):
            raise ValueError('Invalid takeover arrays/information boundary')
        accepted.append((outcome,x,y))
    if not accepted:
        raise ValueError('No successful nonzero-time takeover trajectories')
    return accepted, report

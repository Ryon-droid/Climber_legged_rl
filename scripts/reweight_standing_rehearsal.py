"""Derive an audited rehearsal dataset by changing only category sampling weights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reweighted_arrays(arrays, report, recovery_mass):
    if not 0 < recovery_mass < 1:
        raise ValueError('Recovery mass must lie strictly between zero and one')
    if (not report.get('successful_data_only')
            or report.get('runtime_reference_assistance') is not False):
        raise ValueError('Only successful pure-policy source data is accepted')
    outcomes = [r for r in report['outcomes'] if r['retained_samples'] > 0]
    if any(not r['success'] or r['role'] not in ('stable', 'recovery') for r in outcomes):
        raise ValueError('Invalid retained trajectory provenance')
    counts = {role:sum(r['role'] == role for r in outcomes) for role in ('stable', 'recovery')}
    if not all(counts.values()):
        raise ValueError('Both successful trajectory categories are required')
    old, new = [], []
    masses = dict(stable=1-recovery_mass, recovery=recovery_mass)
    for row in outcomes:
        role, n = row['role'], row['retained_samples']
        old.extend([report['category_mass'][role]/counts[role]/n]*n)
        new.extend([masses[role]/counts[role]/n]*n)
    old = np.asarray(old, dtype=np.float64)
    if (arrays['observations'].shape[0] != len(old)
            or arrays['actions'].shape != (len(old), 6)
            or arrays['weights'].shape != old.shape
            or not np.allclose(arrays['weights'], old, rtol=1e-12, atol=1e-15)):
        raise ValueError('Source row order/count/weights do not match trajectory provenance')
    return dict(observations=arrays['observations'].copy(), actions=arrays['actions'].copy(),
                weights=np.asarray(new, dtype=np.float64)), masses


def create_dataset(source, output, recovery_mass):
    source, output = Path(source).resolve(), Path(output).resolve()
    report_path = source.with_suffix('.json')
    report = json.loads(report_path.read_text())
    if sha(source) != report['dataset_sha256']:
        raise ValueError('Source dataset hash differs from its report')
    with np.load(source, allow_pickle=False) as data:
        arrays, masses = reweighted_arrays(data, report, recovery_mass)
    output.mkdir(parents=True, exist_ok=False)
    dataset = output/'standing.npz'
    np.savez_compressed(dataset, **arrays)
    # This is the actual source collection config, not a fabricated new rollout.
    (output/'config.yaml').write_bytes((source.parent/'config.yaml').read_bytes())
    derived = copy.deepcopy(report)
    derived.update(dataset_sha256=sha(dataset), category_mass=masses)
    derived['reweight_provenance'] = dict(source_dataset=str(source), source_dataset_sha256=sha(source),
        source_report_sha256=sha(report_path), source_collection_config_sha256=sha(source.parent/'config.yaml'),
        observations_and_actions_unchanged=True, trajectory_order_unchanged=True,
        only_sampling_weights_changed=True, new_rollouts_collected=False)
    dataset.with_suffix('.json').write_text(json.dumps(derived, indent=2)+'\n')
    with np.load(source, allow_pickle=False) as original, np.load(dataset, allow_pickle=False) as final:
        for key in ('observations', 'actions'):
            np.testing.assert_array_equal(original[key], final[key])
        assert np.isclose(final['weights'].sum(), 1.)
    print(json.dumps(dict(dataset=str(dataset), dataset_sha256=sha(dataset), category_mass=masses)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recovery-mass', type=float, default=.3)
    args = parser.parse_args()
    create_dataset(args.source, args.output, args.recovery_mass)

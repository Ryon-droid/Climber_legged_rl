"""Validate the immutable source model after a documented path-only migration.

Use the checkpoint's original solved reset state. Re-solving an underdetermined
closed chain on another CPU/BLAS can select a different passive hinge state.
This does not relax model equality or change the physics of a warm start.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'source'), str(ROOT / 'scripts')]
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import mujoco
import numpy as np
from climber_legged_rl.cad_dynamics import input_hashes, validate_artifacts
from climber_legged_rl.cad_model import closure_vectors
from climber_legged_rl.getup import prepare_pose_bank
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.mujoco_training import validate_policy_transfer
from smoke_mujoco import run as smoke


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg = ROOT / 'configs/robot/cad_balance_estimator_round14.yaml'
    config = load_config(cfg)
    folder = ROOT / config['experiment']['output_directory']
    xml = folder / 'robot_experiment.xml'
    migration = ROOT / 'outputs/relocation_2080_20261104'
    manifest = json.loads((migration / 'manifest.json').read_text())
    assert manifest['new_root'] == str(ROOT)
    changes = {entry['path']: entry for entry in manifest['changes']}
    originals = migration / 'originals'
    original_xml = originals / xml.relative_to(ROOT)
    original_report = originals / (folder / 'model_report.json').relative_to(ROOT)
    report = json.loads(original_report.read_text())
    assert sha(original_xml) == report['generated_sha256']
    assert sha(cfg) == report['config_sha256']
    current_inputs = input_hashes(cfg)
    assert current_inputs.keys() == report['input_sha256'].keys()
    for name, before in report['input_sha256'].items():
        after = current_inputs[name]
        if before != after:
            entry = changes[name]
            assert before == entry['before_sha256'] and after == entry['after_sha256'], name
    imported = original_xml.read_bytes().replace(manifest['old_root'].encode(), str(ROOT).encode())
    parent = ROOT / 'outputs/mujoco_runs/cad_balance_estimator_round13_20261101_221330'
    assert imported == (parent / 'robot_experiment.xml').read_bytes()
    assert hashlib.sha256(imported).hexdigest() == manifest['parent_model_sha256']
    difference = None
    if xml.read_bytes() != imported:
        # Keep the first cross-host IK result as diagnostic evidence. Require
        # exact agreement of everything except the solved free-base/reset pose.
        candidate = xml.read_bytes()
        a, b = ET.fromstring(candidate), ET.fromstring(imported)
        for tree in (a, b):
            base = tree.find('./worldbody/body')
            for key in ('pos', 'quat'):
                base.attrib.pop(key, None)
            tree.find('./keyframe/key[@name="standing"]').attrib.pop('qpos')
        assert ET.tostring(a) == ET.tostring(b), 'Derived physical model differs beyond solved reset state'
        saved = migration / 'cross_host_ik_candidate'
        saved.mkdir(exist_ok=True)
        if not (saved / 'robot_experiment.xml').exists():
            (saved / 'robot_experiment.xml').write_bytes(candidate)
            for name in ('model_report.json', 'smoke_report.json', 'getup_poses.json'):
                if (folder / name).exists():
                    (saved / name).write_bytes((folder / name).read_bytes())
        difference = dict(candidate_sha256=hashlib.sha256(candidate).hexdigest(),
                          comparison='All non-reset physical XML attributes exactly match the imported model',
                          selected='Immutable checkpoint model, including its original solved reset state')
    xml.write_bytes(imported)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    closure = float(np.max(np.linalg.norm(closure_vectors(model, data), axis=1)))
    assert closure < 1e-4
    assert (model.nu, model.neq) == (6, 6)
    assert model.nq == report['nq'] and model.nv == report['nv']
    masses = {model.body(i).name: float(model.body_mass[i]) for i in range(1, model.nbody)}
    assert masses == report['body_masses_kg']
    np.testing.assert_allclose(float(model.body_mass.sum()), report['total_mass_kg'], rtol=0, atol=1e-12)
    report.update(generated_model=str(xml), generated_sha256=sha(xml),
                  source_sha256=sha(ROOT / config['experiment']['source_model']),
                  input_sha256=current_inputs, closure_error_m=closure)
    report['generation_provenance'] = dict(
        mode='Immutable cloud-generated physical model, resource paths relocated and compiled on target',
        timestamp=datetime.now(timezone.utc).isoformat(), original_report_sha256=sha(original_report),
        original_model_sha256=sha(original_xml),
        manifest='outputs/relocation_2080_20261104/manifest.json',
        physical_model_or_reset_state_changed=False,
        target_compile=dict(nq=model.nq, nv=model.nv, actuators=model.nu, closures=model.neq,
                            total_mass_kg=float(model.body_mass.sum()), closure_error_m=closure),
        cross_host_ik_diagnostic=difference)
    (folder / 'model_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    prepare_pose_bank(cfg)
    smoke(cfg, steps=1000)
    validate_artifacts(cfg)
    validate_policy_transfer(parent / 'best/best_model.zip', xml, config, warm_start=True)
    print('IMMUTABLE_MODEL_AND_WARM_START_VALIDATED=' + sha(xml), flush=True)


if __name__ == '__main__':
    main()

"""Finish independent acceptance after an interrupted paired training run."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import time

import train_spring208_push_ab as trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    trial.setup()
    folder = args.folder.resolve()
    lock = (trial.ROOT / 'outputs/spring208_phase1_launch/workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((folder / 'workflow.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    if protocol['parent_sha256'] != trial.PARENT_SHA:
        raise RuntimeError('Parent differs from the saved protocol')
    if state['steps'] != state['max_steps']:
        raise RuntimeError('Training is incomplete; refusing final acceptance')
    if (folder / 'assessment.json').exists():
        raise RuntimeError('Acceptance already exists')
    trial.write(folder / 'interrupted_workflow.json', state)
    start = time.monotonic()
    prior_seconds = state.get('elapsed_seconds', 0.)
    state.update(pid=os.getpid(), state='final_independent_test', resumed_evaluation=True)
    state.pop('error', None)
    trial.write(folder / 'workflow.json', state)
    assessor = trial.common.Assessor()
    try:
        before = assessor.evaluate(trial.PARENT, 18., 'test', True)
        trial.write(folder / 'parent_test.json', before)
        selected = folder / 'selected_policy.zip'
        identical = trial.sha(selected) == trial.sha(trial.PARENT)
        if identical:
            after = copy.deepcopy(before)
            after.update(checkpoint=str(selected), reused_identical_checkpoint_evaluation=True)
        else:
            after = assessor.evaluate(selected, 18., 'test', True)
        trial.write(folder / 'selected_test.json', after)
        physics_ok = not any(e['warnings'] or max(e['spring_upper_mm'] + e['spring_lower_mm']) > 1.
                             for e in after['episodes'])
        passed = (trial.common.pass_level(after, before)
                  and after['summary']['exact18']['success'] / 24 >= .9 and physics_ok)
        assessment = dict(target_force_N=18., target_passed=passed, target_150N_passed=False,
            steps=state['steps'], seconds=prior_seconds + time.monotonic() - start,
            parent_summary=before['summary'], selected_summary=after['summary'],
            selected_checkpoint=str(selected), selected_sha256=trial.sha(selected),
            parent_sha256=trial.PARENT_SHA, physical_checks_passed=physics_ok,
            selected_identical_to_parent=identical, simulation_only=True,
            resumed_evaluation=True, automatic_velocity_training=False)
        trial.write(folder / 'assessment.json', assessment)
        state.update(state='completed', elapsed_seconds=assessment['seconds'],
                     target_passed=passed, target_150N_passed=False)
        trial.write(folder / 'workflow.json', state)
        print(json.dumps(assessment), flush=True)
    except BaseException as error:
        state.update(state='failed', error=repr(error))
        trial.write(folder / 'workflow.json', state)
        raise
    finally:
        assessor.close()


if __name__ == '__main__':
    main()

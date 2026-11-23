"""Balance fresh failure replay with successful training-only retention cases."""
import copy
import random


def failure_weight(events):
    second = len(events) >= 2 and events[0]['recovered'] and not events[-1]['recovered']
    negative = bool(events and events[-1]['force_world_N'][0] < 0)
    delayed = bool(events and all(e['recovered'] for e in events))
    return 6 if second or negative else 4 if delayed else 1


def build_directional_replay(rows, seed, training_seed_range, retention_weights=None):
    """Half the entries target failures; half preserve successful force mixtures.

    Entire episodes are replayed, including the period after event recovery.
    Retention examples are reset seeds, not action labels or test-set examples.
    """
    low, high = training_seed_range
    if not rows or any(r['warnings'] for r in rows):
        raise ValueError('Discovery must be nonempty and free of numerical warnings')
    seeds = [r['spec']['seed'] for r in rows]
    if len(set(seeds)) != len(seeds) or any(not low <= s <= high for s in seeds):
        raise ValueError('Replay requires distinct seeds from the fresh training range')
    failures = []
    buckets = {k: [] for k in ('standing', 'fixed_negative_x', 'mixed_negative_x', 'mixed_other')}
    for row in rows:
        spec = row['spec']
        events = row['push_metrics']['push_events']
        case = dict(seed=spec['seed'], force_N=list(spec['force_N']), push=spec['push'],
                    expected_push_events=copy.deepcopy(events), source='fresh training discovery',
                    failure_reason=None if row['success'] else row['reason'],
                    delayed_failure_after_recovery=bool(not row['success'] and events
                        and all(e['recovered'] for e in events)),
                    failed_after_first_recovery=bool(not row['success'] and len(events) >= 2
                        and events[0]['recovered'] and not events[-1]['recovered']),
                    successful_retention=bool(row['success']))
        if not row['success']:
            failures.extend(copy.deepcopy(case) for _ in range(failure_weight(events)))
            continue
        negative = any(e['force_world_N'][0] < 0 for e in events)
        mixed = spec['force_N'][0] != spec['force_N'][1]
        group = ('standing' if not spec['push'] else 'mixed_negative_x' if mixed and negative
                 else 'fixed_negative_x' if negative else 'mixed_other' if mixed else None)
        if group is not None:
            case['retention_group'] = group
            buckets[group].append(case)
    if not failures:
        raise ValueError('No fresh training failures to target')
    # Prefer mixed-force anchors: the last round improved exact-force trials
    # while losing a mixed-force trial. Preserve standing as well.
    weights = dict(standing=1, fixed_negative_x=2, mixed_negative_x=3, mixed_other=2)
    if retention_weights is not None:
        if (set(retention_weights) != set(buckets)
                or any(type(v) is not int or v <= 0 for v in retention_weights.values())):
            raise ValueError('Retention weights require positive integers for every group')
        weights = dict(retention_weights)
        if not buckets['standing']:
            raise ValueError('Standing retention requires successful fresh no-push cases')
    schedule = [k for k, weight in weights.items() if buckets[k] for _ in range(weight)]
    if not schedule:
        raise ValueError('No successful training retention cases')
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    cursors = dict.fromkeys(buckets, 0)
    retention = []
    for index in range(len(failures)):
        group = schedule[index % len(schedule)]
        retention.append(copy.deepcopy(buckets[group][cursors[group] % len(buckets[group])]))
        cursors[group] += 1
    cases = failures + retention
    return cases, dict(failure_replay_entries=len(failures),
                       successful_retention_entries=len(retention),
                       available_retention_cases={k: len(v) for k, v in buckets.items()},
                       retention_entries_per_group=cursors,
                       retention_sampling_weights=weights,
                       fresh_training_seed_range=[low, high],
                       sampling_weights_combination='maximum applicable weight',
                       full_episode_replay=True, test_or_selection_seeds_used=False)

"""Read-only status collector, executed through the existing SSH connection."""
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/climber/Climber_legged_rl')
OUT = ROOT / 'outputs'


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def tail(path, limit=14000):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode('utf-8', 'replace')
    except OSError:
        return ''


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def optimizer_series(folder, protocol):
    """Read each branch separately; its CSV counter restarts at zero."""
    keys = {'steps': 'time/total_timesteps', 'reward': 'rollout/ep_rew_mean',
            'episode_steps': 'rollout/ep_len_mean', 'success': 'rollout/success_rate',
            'fps': 'time/fps', 'kl': 'train/approx_kl', 'value_loss': 'train/value_loss',
            'replay_loss': 'train/standing_rehearsal_loss'}
    series = []
    for path in sorted(folder.glob('logs*/progress.csv')):
        parts = path.parent.name.split('_')
        try:
            attempt = int(parts[-1])
            level = int(parts[-2])
            rows = list(csv.DictReader(io.StringIO(path.read_text())))
        except (OSError, ValueError):
            continue
        points = [{key: number(row.get(column)) for key, column in keys.items()} for row in rows]
        points = [p for p in points if p['steps'] is not None]
        if not points:
            continue
        arms = protocol.get('arms', [])
        arm = arms[attempt] if attempt < len(arms) else f'attempt_{attempt}'
        stride = max(1, math.ceil(len(points) / 500))
        sampled = points[::stride]
        if sampled[-1] is not points[-1]:
            sampled.append(points[-1])
        series.append(dict(arm=arm, attempt=attempt, level_N=level,
                           points=sampled, latest=points[-1], rows=len(points)))
    return series


def paired_steps(series, curves, arms, budget):
    total = 0
    for index, arm in enumerate(arms):
        counters = [r['latest']['steps'] for r in series if r['arm'] == arm]
        counters += [r.get('steps', 0) - index * budget for r in curves
                     if r.get('arm') == arm or (not r.get('arm') and r.get('attempt') == index)]
        total += max(0, min(budget, max(counters, default=0)))
    return int(total)


def collect(run_name=None):
    runs = []
    for path in OUT.glob('*/workflow.json'):
        state = read(path, {})
        if state.get('started_unix') and ('push' in path.parent.name or state.get('profile')):
            runs.append(dict(name=path.parent.name, **state))
    runs.sort(key=lambda row: row.get('started_unix', 0), reverse=True)
    if not runs:
        raise RuntimeError('没有找到训练记录')
    if run_name is not None and run_name not in {row['name'] for row in runs}:
        raise ValueError('未知训练记录')
    name = run_name or runs[0]['name']
    folder = OUT / name
    state = read(folder / 'workflow.json', {})
    protocol = read(folder / 'protocol.json', {})
    training_series = optimizer_series(folder, protocol)
    selected = read(folder / 'selected_selection.json', {})
    report = read(folder / 'assessment.json')
    level = state.get('level_N')
    parent = read(folder / f'parent_level{int(level or 0)}.json', {})
    if not parent:
        parent = read(folder / 'parent_selection.json', {})
    curves = []
    for path in folder.glob('curve*.json'):
        rows = read(path, [])
        if isinstance(rows, list):
            curves.extend(rows)
    curves.sort(key=lambda row: (row.get('steps', 0), row.get('attempt', 0)))
    live = False
    if state.get('pid') and state.get('state') not in ('completed', 'failed'):
        try:
            command = Path(f"/proc/{int(state['pid'])}/cmdline").read_bytes()
            live = b'train_spring208_push' in command and name.encode() in command
        except OSError:
            pass
    elapsed = (time.time() - state['started_unix']) if live else state.get('elapsed_seconds', 0)
    optimizer = {}
    progress_files = list(folder.glob('logs*/progress.csv'))
    if progress_files:
        latest = max(progress_files, key=lambda p: p.stat().st_mtime)
        try:
            rows = list(csv.DictReader(io.StringIO(latest.read_text())))
            if rows:
                optimizer = rows[-1]
        except (OSError, ValueError):
            pass
    # Complete episodes provide a lower bound, rather than claiming a live
    # optimizer counter when the trainer only checkpoints every 65k steps.
    sampled = None
    if live and state.get('state') == 'training' and level is not None:
        attempt = state.get('attempt', 0)
        current_curve = folder / f'curve_{int(level)}_{attempt}.json'
        previous = []
        for path in folder.glob('curve*.json'):
            if path != current_curve:
                previous.extend(read(path, []))
        base_steps = max((row.get('steps', 0) for row in previous), default=0)
        lengths = 0
        try:
            for path in folder.glob(f'monitor_{int(level)}_{attempt}_*.csv'):
                rows = csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith('#'))
                lengths += sum(int(row.get('l', 0)) for row in rows if row.get('l'))
            sampled = max(state.get('steps', 0), base_steps + lengths)
        except (OSError, ValueError):
            pass
    # Paired runs have an explicit per-arm budget and independent counters.
    # Use flushed rollout counters instead of combining unrelated monitor files.
    if protocol.get('steps_per_arm'):
        saved_steps = paired_steps(training_series, curves, protocol.get('arms', []), protocol['steps_per_arm'])
        sampled = max(state.get('steps', 0), int(saved_steps))
    hardware = {}
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        if gpu.returncode == 0:
            fields = [part.strip() for part in gpu.stdout.splitlines()[0].split(',')]
            hardware.update(gpu=fields[0], gpu_used_mb=int(fields[1]), gpu_total_mb=int(fields[2]), gpu_percent=int(fields[3]))
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        total = int(memory['MemTotal'].split()[0])
        available = int(memory['MemAvailable'].split()[0])
        hardware.update(memory_total_gb=round(total/1048576, 1), memory_used_gb=round((total-available)/1048576, 1))
        hardware['load'] = Path('/proc/loadavg').read_text().split()[0]
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    videos = []
    for pattern in ('*.mp4', '*.webm'):
        for path in OUT.rglob(pattern):
            if path.is_symlink():
                continue
            stat = path.stat()
            videos.append(dict(path=str(path.relative_to(ROOT)), name=path.name, modified=stat.st_mtime,
                               size_mb=round(stat.st_size/1048576, 2),
                               console_preview='training_console_previews' in path.parts))
    videos.sort(key=lambda row: row['modified'], reverse=True)
    return dict(connected=True, fetched_at=time.time(), run=name, runs=runs, state=state,
                live=live, elapsed_seconds=elapsed, sampled_steps_lower_bound=sampled,
                protocol=protocol, selected_summary=selected.get('summary', {}),
                selected_strict_summary=selected.get('strict_summary', {}),
                parent_strict_summary=parent.get('strict_summary', {}),
                selected_checkpoint=selected.get('checkpoint'), selected_sha256=selected.get('sha256'),
                selected_level_N=selected.get('level_N'), parent_summary=parent.get('summary', {}),
                assessment=report, levels=read(folder / 'levels.json', []), curves=curves,
                optimizer=optimizer, training_series=training_series, hardware=hardware,
                log=tail(folder / 'console.log'), videos=videos[:80])


if __name__ == '__main__':
    print(json.dumps(collect(sys.argv[1] if len(sys.argv) > 1 else None), ensure_ascii=False))

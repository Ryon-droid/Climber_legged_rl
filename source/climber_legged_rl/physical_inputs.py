"""Read confirmed body masses independently of pending COM/inertia inputs."""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEASUREMENTS = ROOT / 'configs/robot/measurements'


def load_body_masses(directory=DEFAULT_MEASUREMENTS):
    """Return only supplied, sourced masses; never infer missing body values."""
    path = Path(directory) / 'rigid_bodies.csv'
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    names = [row['body'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate body names in mass inputs')
    masses = {}
    for row in rows:
        if not row['mass_kg'].strip():
            continue
        mass = float(row['mass_kg'])
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError('Body mass must be positive and finite: ' + row['body'])
        if row['scope'] != 'complete_body_including_attached_parts' or not row['source'].strip():
            raise ValueError('Body mass requires complete-body scope and source: ' + row['body'])
        masses[row['body']] = mass
    return masses

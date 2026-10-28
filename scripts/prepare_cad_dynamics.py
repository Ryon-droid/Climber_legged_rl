"""Prepare floating-base six-motor dynamics using actual repaired CAD."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from climber_legged_rl.cad_dynamics import build_dynamics, DEFAULT_DYNAMICS_CONFIG

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_DYNAMICS_CONFIG)
    args=parser.parse_args()
    _, report=build_dynamics(args.config)
    print(json.dumps(report,ensure_ascii=False,indent=2))

"""Audit the actual CAD export without approximate dynamics or PPO."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))

if __name__=='__main__':
    from climber_legged_rl.cad_model import DEFAULT_CAD_CONFIG,audit_model
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CAD_CONFIG)
    parser.add_argument('--skip-motion',action='store_true')
    args=parser.parse_args()
    report=audit_model(args.config,run_motion=not args.skip_motion)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(0 if report['ready_for_training'] else 2)

"""Build and verify a separate gas-spring CAD kinematic repair candidate."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from climber_legged_rl.cad_repair import DEFAULT_REPAIR_CONFIG,build_repaired

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_REPAIR_CONFIG)
    args=parser.parse_args()
    path,r=build_repaired(args.config)
    print(json.dumps({k:r[k] for k in ('generated_model','motion_passed','stance','candidate_mass_kg','closure_count','ready_for_training')},ensure_ascii=False,indent=2))
    raise SystemExit(0 if r['motion_passed'] and r['stance']['passed'] else 2)

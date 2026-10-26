"""Build the separate approximate model and solve a closed-chain reset pose."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "source"))
from climber_legged_rl.mujoco_model import build_model, DEFAULT_CONFIG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    path, report = build_model(args.config)
    print(json.dumps(report,ensure_ascii=False,indent=2))

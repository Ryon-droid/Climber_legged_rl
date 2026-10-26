"""Check the real project and MJCF before spending GPU time on training.

Exit 0 means the checks passed, 2 means training is blocked. This command
does not train, change source assets, infer hardware limits, or change masses.
Use --dynamics in an environment containing MuJoCo and NumPy.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
LEG_JOINTS = (
    "left_link1_joint", "left_link_rear_1_joint",
    "right_link1_joint", "right_rear_link1_joint",
)
WHEEL_JOINTS = ("left_link 3_joint", "right_link 3_joint")


def placeholder_raises(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    label = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        func = node.exc.func
        name = func.id if isinstance(func, ast.Name) else ""
        if name == "NotImplementedError":
            results.append(f"{label}:{node.lineno}")
        elif name == "SystemExit" and node.exc.args:
            value = node.exc.args[0]
            if isinstance(value, ast.Constant) and "not configured" in str(value.value):
                results.append(f"{label}:{node.lineno}")
    return results


def inspect_model(root: ET.Element, model_path: Path) -> dict:
    world = root.find("worldbody")
    if world is None:
        raise ValueError("MJCF has no worldbody")
    base = world.find("body")
    if base is None:
        raise ValueError("MJCF has no robot root body")
    actuators = root.findall("./actuator/*")
    mapped = {a.get("joint") for a in actuators}
    mesh_paths = [model_path.parent / m.get("file", "") for m in root.findall("./asset/mesh")]
    mass = sum(float(i.get("mass", "0")) for i in root.findall(".//body/inertial"))
    return {
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "base_body": base.get("name"),
        "base_has_freejoint": (base.find("freejoint") is not None or any(j.get("type") == "free" for j in base.findall("joint"))),
        "joint_count": len(root.findall(".//body/joint")),
        "actuator_count": len(actuators),
        "actuator_joint_names": [a.get("joint") for a in actuators],
        "missing_motor_actuators": [j for j in LEG_JOINTS + WHEEL_JOINTS if j not in mapped],
        "unexpected_actuated_joints": sorted(j for j in mapped if j and j not in LEG_JOINTS + WHEEL_JOINTS),
        "connect_constraint_count": len(root.findall("./equality/connect")),
        "mesh_count": len(mesh_paths),
        "missing_meshes": [str(p) for p in mesh_paths if not p.is_file()],
        "exported_mass_kg": mass,
        "confirmed_total_mass_kg": 15.0,
        "mass_note": "15 kg is user-confirmed; no scaling or redistribution is applied.",
    }


def check_dynamics(root: ET.Element, model_path: Path) -> dict:
    import mujoco
    import numpy as np

    result = {"mujoco_version": mujoco.__version__, "source_modified": False}
    try:
        mujoco.MjModel.from_xml_path(str(model_path))
        result["original_compiles"] = True
    except Exception as exc:
        result.update(original_compiles=False, original_error=str(exc))
    # A diagnostic copy ONLY: STL surface shells are not closed volumes.
    # Explicit rigid-body inertials are retained; the original file is untouched.
    changed = []
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", str((model_path.parent / mesh.get("file", "")).resolve()))
        if "surface_shell" in mesh.get("name", ""):
            mesh.set("inertia", "shell")
            changed.append(mesh.get("name"))
    result["diagnostic_shell_mesh_count"] = len(changed)
    try:
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        closures = []
        for i in range(model.neq):
            if model.eq_type[i] != mujoco.mjtEq.mjEQ_CONNECT:
                continue
            closures.append({
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i),
                "site_distance_m": float(np.linalg.norm(data.site_xpos[model.eq_obj1id[i]] - data.site_xpos[model.eq_obj2id[i]])),
            })
        result.update(diagnostic_copy_compiles=True, nq=model.nq, nv=model.nv, nu=model.nu, closure_distances=closures)
        result["diagnostic_only"] = "Compilation and zero-pose closure are not locomotion or stability validation."
    except Exception as exc:
        result.update(diagnostic_copy_compiles=False, diagnostic_error=str(exc))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cad","cad-repaired","isaaclab","mujoco"], default="cad")
    parser.add_argument("--dynamics", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, help="Configuration for the CAD route")
    args = parser.parse_args()
    if args.backend == "cad-repaired":
        sys.path.insert(0, str(ROOT/"source"))
        from climber_legged_rl.cad_repair import build_repaired, DEFAULT_REPAIR_CONFIG
        _, report = build_repaired(args.config or DEFAULT_REPAIR_CONFIG)
        serialized=json.dumps(report,ensure_ascii=False,indent=2)
        print(serialized)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(serialized+"\n",encoding="utf-8")
        return 2
    if args.backend == "cad":
        modules = {n: importlib.util.find_spec(n) is not None for n in ("mujoco", "scipy", "yaml", "numpy")}
        if all(modules.values()):
            sys.path.insert(0, str(ROOT/"source"))
            from climber_legged_rl.cad_model import audit_model, DEFAULT_CAD_CONFIG
            report = audit_model(args.config or DEFAULT_CAD_CONFIG)
        else:
            report = {"model_policy": "source_cad_only", "ready_for_training": False,
                      "training_started": False, "modules": modules,
                      "blockers": ["Use the MuJoCo validation environment to audit actual CAD assets."]}
        serialized = json.dumps(report, ensure_ascii=False, indent=2)
        print(serialized)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized+"\n", encoding="utf-8")
        return 0 if report["ready_for_training"] else 2
    if args.backend == "mujoco":
        modules = {n: importlib.util.find_spec(n) is not None for n in ("mujoco","gymnasium","torch","stable_baselines3","scipy","yaml")}
        report = {"backend":"mujoco", "simulation_only":True, "training_started_by_this_check":False,
                  "modules":modules, "blockers":[], "hardware_ready":False}
        if not all(modules.values()):
            report["blockers"].append("Install requirements-mujoco.txt in an isolated environment.")
        else:
            sys.path.insert(0,str(ROOT/"source"))
            from climber_legged_rl.mujoco_model import load_config
            config_path=ROOT/"configs/robot/mujoco_experiment.yaml"
            config=load_config(config_path)
            folder=ROOT/config["experiment"]["output_directory"]
            model_path=folder/"robot_experiment.xml"
            smoke_path=folder/"smoke_report.json"
            if not model_path.is_file() or not smoke_path.is_file():
                report["blockers"].append("Run prepare_mujoco_experiment.py and smoke_mujoco.py.")
            else:
                smoke=json.loads(smoke_path.read_text())
                for key,path in (("config_sha256",config_path),("model_sha256",model_path),
                                 ("env_sha256",ROOT/"source/climber_legged_rl/mujoco_env.py")):
                    if smoke.get(key)!=hashlib.sha256(path.read_bytes()).hexdigest():
                        report["blockers"].append("Stale smoke check: "+key)
                if not smoke.get("passed"):
                    report["blockers"].append("Smoke check failed")
                report["smoke_report"]=str(smoke_path)
                report["approximations"]=config["experiment"]["approximation_notes"]
        report["ready_for_approximate_experiment"]=not report["blockers"]
        serialized=json.dumps(report,ensure_ascii=False,indent=2)
        print(serialized)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(serialized+"\n",encoding="utf-8")
        return 2 if report["blockers"] else 0
    model_path = ROOT / "sim2sim/climber_mjcf/robot.xml"
    root = ET.parse(model_path).getroot()
    info = inspect_model(root, model_path)
    files = ("scripts/train.py", "source/climber_legged_rl/tasks/climber/env.py",
             "source/climber_legged_rl/tasks/climber/env_cfg.py",
             "source/climber_legged_rl/tasks/climber/register.py")
    placeholders = [line for name in files for line in placeholder_raises(ROOT / name)]
    modules = {name: importlib.util.find_spec(name) is not None for name in ("torch", "isaacsim", "isaaclab", "rsl_rl", "mujoco")}
    blockers = []
    if placeholders:
        blockers.append("Training entry point/task/config/registration are not implemented.")
    if not all(modules[name] for name in ("torch", "isaacsim", "isaaclab", "rsl_rl")):
        blockers.append("The selected Python environment lacks required Isaac Lab training modules.")
    if not info["base_has_freejoint"]:
        blockers.append("MJCF robot base is fixed; floating-base locomotion is unavailable.")
    if info["missing_motor_actuators"] or info["unexpected_actuated_joints"]:
        blockers.append("Exported actuators do not match four leg motors plus two wheel motors.")
    if info["missing_meshes"]:
        blockers.append("Referenced mesh files are missing.")
    if abs(info["exported_mass_kg"] - info["confirmed_total_mass_kg"]) > 0.1:
        blockers.append("Exported mass disagrees with confirmed mass; reconcile or explicitly approve an approximate experiment.")
    report = {"training_started": False, "python": sys.executable, "modules": modules,
              "placeholder_locations": placeholders, "model": info, "blockers": blockers,
              "warnings": ["Original USD and latest MJCF are not verified as synchronized.",
                           "Motor output limits, current physical joint limits and gas-spring force model remain unverified."]}
    if args.dynamics:
        if not modules["mujoco"]:
            blockers.append("MuJoCo is required for --dynamics.")
        else:
            report["dynamics"] = check_dynamics(root, model_path)
            if not report["dynamics"]["original_compiles"]:
                blockers.append("Original MJCF does not compile in installed MuJoCo; see original_error.")
    report["ready"] = not blockers
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

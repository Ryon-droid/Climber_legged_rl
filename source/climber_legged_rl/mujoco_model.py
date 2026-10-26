"""Build an explicitly approximate model without editing CAD source exports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/robot/mujoco_experiment.yaml"


def load_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["experiment"]["simulation_only"] is not True:
        raise ValueError("This approximate model is simulation-only")
    if config["mass"]["total_kg"] <= 2 * config["mass"]["side_total_kg"]:
        raise ValueError("Mass breakdown leaves no positive central body mass")
    return config


def closure_error(model, data):
    pairs = [(model.eq_obj1id[i], model.eq_obj2id[i]) for i in range(model.neq)
             if model.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT]
    return np.array([data.site_xpos[a] - data.site_xpos[b] for a, b in pairs])


def _scale_inertial(element, new_mass):
    old_mass = float(element.get("mass"))
    factor = new_mass / old_mass
    element.set("mass", f"{new_mass:.12g}")
    for attribute in ("diaginertia", "fullinertia"):
        if attribute in element.attrib:
            element.set(attribute, " ".join(f"{float(x) * factor:.12g}" for x in element.get(attribute).split()))


def _mass_override(base, config):
    side_mass = config["mass"]["side_total_kg"]
    wheel_mass = config["mass"]["wheel_each_kg"]
    summary = {}
    for side in ("left", "right"):
        # Nonstandard helper-body names live inside the side's hip branches.
        bodies = []
        for branch in base.findall("body"):
            if branch.get("name", "").startswith(side):
                bodies.extend([branch, *branch.findall(".//body")])
        wheel = next(b for b in bodies if b.get("name") == f"{side}_link 3")
        nonwheel = [b.find("inertial") for b in bodies if b is not wheel and b.find("inertial") is not None]
        original = sum(float(i.get("mass")) for i in nonwheel)
        factor = (side_mass - wheel_mass) / original
        for inertial in nonwheel:
            _scale_inertial(inertial, float(inertial.get("mass")) * factor)
        _scale_inertial(wheel.find("inertial"), wheel_mass)
        summary[side] = {"mass_kg": side_mass, "nonwheel_scale": factor, "wheel_kg": wheel_mass}
    central = config["mass"]["total_kg"] - 2 * side_mass
    _scale_inertial(base.find("inertial"), central)
    summary["base_kg"] = central
    return summary


def _solve_pose(model, config):
    data = mujoco.MjData(model)
    wheel_names = config["control"]["wheel_joints"]
    wheel_indices = [int(model.joint(n).qposadr[0]) for n in wheel_names]
    indices = [i for i in range(model.nq) if i not in wheel_indices]
    # Fixed-base intermediate model has one scalar per joint.
    lows = np.array([max(model.jnt_range[i, 0], -1.5) for i in indices])
    highs = np.array([min(model.jnt_range[i, 1], 1.5) for i in indices])
    target_height = config["physics"]["rest_height_hip_above_wheel_m"]
    wheel_bodies = [int(model.body(f"{side}_link 3").id) for side in ("left", "right")]
    hip_bodies = [int(model.body(f"{side}_link1").id) for side in ("left", "right")]
    base_com_x = model.body_ipos[model.body("base").id, 0]

    def residual(values):
        data.qpos[:] = 0.0
        data.qpos[indices] = values
        mujoco.mj_forward(model, data)
        heights = np.array([data.xpos[h, 2] - data.xpos[w, 2] for h, w in zip(hip_bodies, wheel_bodies)])
        wheel_x = data.xpos[wheel_bodies, 0]
        # Primary closure, secondary supplied height, tertiary numerical reset.
        return np.concatenate((closure_error(model, data).ravel() * 1000,
                               (heights - target_height) * 20,
                               (wheel_x - base_com_x) * 2, values * 0.00001))

    rng = np.random.default_rng(config["physics"]["ik_seed"])
    attempts = []
    best = None
    for k in range(8):
        initial = np.zeros(len(indices)) if k == 0 else rng.uniform(-0.8, 0.8, len(indices))
        result = least_squares(residual, initial, bounds=(lows, highs), max_nfev=1200,
                               ftol=1e-10, xtol=1e-10, gtol=1e-9)
        residual(result.x)
        max_error = float(np.linalg.norm(closure_error(model, data), axis=1).max())
        heights = [float(data.xpos[h, 2] - data.xpos[w, 2]) for h, w in zip(hip_bodies, wheel_bodies)]
        height_error = max(abs(h - target_height) for h in heights)
        record = {"attempt": k, "closure_error_m": max_error, "hip_heights_m": heights, "cost": float(result.cost)}
        attempts.append(record)
        if max_error < config["physics"]["ik_max_closure_error_m"] and height_error < config["physics"]["ik_max_height_error_m"]:
            best = (data.qpos.copy(), record, data.xpos[wheel_bodies].copy())
            break
    if best is None:
        raise RuntimeError("No feasible closed-chain standing reset found: " + json.dumps(attempts))
    qpos, chosen, wheel_positions = best
    return qpos, wheel_positions, {"chosen": chosen, "attempts": attempts,
        "joint_angles_rad": {model.joint(i).name: float(qpos[i]) for i in range(model.njnt)},
        "note": "Numerically feasible simulation reset; NOT measured encoder zeros or hardware stance."}


def build_model(config_path=DEFAULT_CONFIG, output_directory=None):
    config = load_config(config_path)
    source = ROOT / config["experiment"]["source_model"]
    output = Path(output_directory) if output_directory else ROOT / config["experiment"]["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    root = ET.parse(source).getroot()
    root.set("model", "climber_APPROXIMATE_SIMULATION_ONLY")
    base = root.find("./worldbody/body")
    mass_summary = _mass_override(base, config)
    # The source CAD coordinate labels are not an IMU calibration. A pi yaw
    # makes the named physical left appear at canonical +y. Forward is provisional.
    base.set("quat", "0 0 0 1")
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", str((source.parent / mesh.get("file")).resolve()))
        if "surface_shell" in mesh.get("name", ""):
            mesh.set("inertia", "shell")
    for geom in root.findall(".//body/geom"):
        geom.set("contype", "0"); geom.set("conaffinity", "0")
        geom.set("group", "1")
    for joint in root.findall(".//body/joint"):
        joint.attrib.pop("actuatorfrcrange", None)
        joint.attrib.pop("actuatorfrclimited", None)
        joint.set("damping", "0.05"); joint.set("armature", "0.002")
    for element in root.findall("actuator") + root.findall("keyframe"):
        root.remove(element)
    ET.SubElement(root, "option", timestep=str(config["physics"]["timestep"]),
                           iterations=str(config["physics"]["solver_iterations"]),
                           integrator="implicitfast", solver="Newton", jacobian="dense", gravity="0 0 -9.81")
    # Solve using the fixed-base intermediate model; no contact forces or training.
    intermediate = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    qpos, wheel_positions, pose_report = _solve_pose(intermediate, config)
    radius = config["physics"]["wheel_radius_m"]
    z = radius - float(wheel_positions[:, 2].mean()) + 0.001
    base.set("pos", f"0 0 {z:.12g}")
    base.insert(0, ET.Element("freejoint", name="floating_base"))
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="ground", type="plane", size="10 10 0.1",
                  rgba="0.65 0.7 0.75 1", friction="0.8 0.005 0.0001", condim="3")
    ET.SubElement(world, "light", name="sun", pos="0 0 3", dir="0 0 -1", directional="true")
    center = base.find("inertial").get("pos")
    ET.SubElement(base, "geom", name="base_collision", type="box", pos=center,
                  size=" ".join(map(str,config["physics"]["base_collision_half_extents_m"])),
                  rgba="0.2 0.35 0.6 0.2", group="2", condim="3")
    for side in ("left", "right"):
        wheel = next(b for b in base.findall(".//body") if b.get("name") == f"{side}_link 3")
        ET.SubElement(wheel, "geom", name=f"{side}_wheel_collision", type="sphere",
                      size=str(radius), rgba="0.15 0.15 0.15 0.4", group="2", condim="3",
                      friction=f"{config['physics']['wheel_friction']} 0.005 0.0001")
    actuator = ET.SubElement(root, "actuator")
    control = config["control"]
    joints = control["leg_joints"] + control["wheel_joints"]
    for i, joint in enumerate(joints):
        limit = control["leg_torque_limit_Nm"] if i < 4 else control["wheel_torque_limit_Nm"]
        ET.SubElement(actuator, "motor", name=f"motor_{i}", joint=joint, gear="1",
                      ctrllimited="true", ctrlrange=f"{-limit} {limit}")
    full_qpos = np.concatenate(([0, 0, z, 0, 0, 0, 1], qpos))
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="standing", qpos=" ".join(f"{v:.14g}" for v in full_qpos))
    model_path = output / "robot_experiment.xml"
    ET.indent(root)
    ET.ElementTree(root).write(model_path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model,data,0); mujoco.mj_forward(model,data)
    if model.nu != 6 or abs(model.body_mass.sum() - config["mass"]["total_kg"]) > 1e-8:
        raise RuntimeError("Generated model motor/mass check failed")
    if np.linalg.norm(closure_error(model,data),axis=1).max() > config["physics"]["ik_max_closure_error_m"]:
        raise RuntimeError("Generated free-base model violates reset closure tolerance")
    report = {"simulation_only": True, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "generated_model": str(model_path), "mass_override": mass_summary,
              "total_mass_kg": float(model.body_mass.sum()), "nq": model.nq, "nv": model.nv,
              "nu": model.nu, "standing_base_z_m": z, "pose_solver": pose_report,
              "frame_note": "Root yaw pi: named left at +world-y. Body -source-x is provisional forward. Not IMU calibration.",
              "approximations": config["experiment"]["approximation_notes"]}
    (output / "model_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return model_path, report

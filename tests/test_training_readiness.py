"""Standard-library tests for the deployment readiness gate."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("readiness", ROOT / "scripts/check_training_readiness.py")
readiness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readiness)


class ReadinessTests(unittest.TestCase):
    def test_existing_model_assets_are_intact(self):
        path = ROOT / "sim2sim/climber_mjcf/robot.xml"
        info = readiness.inspect_model(ET.parse(path).getroot(), path)
        self.assertEqual(info["missing_meshes"], [])
        self.assertGreater(info["exported_mass_kg"], 0)
        self.assertGreater(info["mesh_count"], 0)

    def test_missing_wheel_actuators_and_fixed_base_are_detected(self):
        xml = '<mujoco><worldbody><body name="base"><inertial mass="2"/></body></worldbody><actuator><motor joint="left_link1_joint"/></actuator></mujoco>'
        info = readiness.inspect_model(ET.fromstring(xml), ROOT / "sim2sim/climber_mjcf/robot.xml")
        self.assertFalse(info["base_has_freejoint"])
        self.assertTrue(set(readiness.WHEEL_JOINTS).issubset(info["missing_motor_actuators"]))

    def test_floating_base_and_six_motor_mapping_pass(self):
        root = ET.fromstring('<mujoco><worldbody><body name="base"><freejoint/></body></worldbody><actuator/></mujoco>')
        for joint in readiness.LEG_JOINTS + readiness.WHEEL_JOINTS:
            ET.SubElement(root.find("actuator"), "motor", joint=joint)
        info = readiness.inspect_model(root, ROOT / "sim2sim/climber_mjcf/robot.xml")
        self.assertTrue(info["base_has_freejoint"])
        self.assertEqual(info["missing_motor_actuators"], [])
        self.assertEqual(info["unexpected_actuated_joints"], [])

    def test_placeholder_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "placeholder.py"
            path.write_text('raise NotImplementedError("todo")\nraise SystemExit("Training is not configured yet")\n', encoding="utf-8")
            self.assertEqual(len(readiness.placeholder_raises(path)), 2)

    def test_implemented_code_is_not_treated_as_placeholder(self):
        self.assertEqual(readiness.placeholder_raises(ROOT / "scripts/check_training_readiness.py"), [])


if __name__ == "__main__":
    unittest.main()

"""Reject accidental physics changes while reusing a policy across a shorter stop."""
import copy
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
from climber_legged_rl.mujoco_training import validate_policy_transfer


class SpringStopTighteningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        previous = root/'parent'
        previous.mkdir()
        self.checkpoint = previous/'final_policy.zip'
        self.checkpoint.write_bytes(b'validation-only fixture')
        self.model = root/'new.xml'
        self.config = yaml.safe_load((ROOT/'configs/robot/cad_dynamics.yaml').read_text())
        self.config['spring']['eye_center_limits_m'] = {s: [.155, .22] for s in ('left', 'right')}
        (previous/'config.yaml').write_text(yaml.safe_dump(self.config))
        old = ET.fromstring('<mujoco><worldbody><body name="base"/></worldbody></mujoco>')
        for side in ('left', 'right'):
            body = ET.SubElement(old.find('.//body'), 'body', name=side+'_gas_rod', pos='.154 0 0')
            ET.SubElement(body, 'joint', name=side+'_gas_slide_joint', type='slide', limited='true',
                          range='.001 .066', solreflimit='.002 1', damping='.05')
        (previous/'robot_experiment.xml').write_text(ET.tostring(old, encoding='unicode'))
        self.new = copy.deepcopy(old)
        for joint in self.new.findall('.//joint'):
            joint.set('range', '.001 .054')
        self.model.write_text(ET.tostring(self.new, encoding='unicode'))
        self.config['training']['allow_spring_limit_warm_start'] = True
        self.config['training']['spring_limit_migration'] = dict(mode='tighten_upper_stop',
            previous_eye_center_limits_m=copy.deepcopy(self.config['spring']['eye_center_limits_m']))
        self.config['spring']['eye_center_limits_m'] = {s: [.155, .208] for s in ('left', 'right')}

    def validate(self, config=None, warm_start=True):
        return validate_policy_transfer(self.checkpoint, self.model, config or self.config, warm_start=warm_start)

    def test_explicit_warm_start_accepts_shorter_stop_but_resume_does_not(self):
        self.assertEqual(self.validate(), self.checkpoint)
        with self.assertRaises(ValueError):
            self.validate(warm_start=False)

    def test_missing_or_mismatched_recipe_is_rejected(self):
        for recipe in (None, {}, {'mode': 'tighten_upper_stop', 'previous_eye_center_limits_m': {'left': [.155, .23]}}):
            config = copy.deepcopy(self.config)
            config['training']['spring_limit_migration'] = recipe
            with self.assertRaises(ValueError):
                self.validate(config)
        config = copy.deepcopy(self.config)
        config['training'].pop('allow_spring_limit_warm_start')
        with self.assertRaises(ValueError):
            self.validate(config)

    def test_lengthening_lower_stop_and_force_changes_are_rejected(self):
        for side in ('left', 'right'):
            for bounds in ([.154, .208], [.155, .23], [None, .208]):
                config = copy.deepcopy(self.config)
                config['spring']['eye_center_limits_m'][side] = bounds
                with self.assertRaises(ValueError):
                    self.validate(config)
        config = copy.deepcopy(self.config)
        config['spring']['force_N'] += 1
        with self.assertRaises(ValueError):
            self.validate(config)

    def test_unrelated_xml_and_declared_length_mismatch_are_rejected(self):
        for attribute, value in [('damping', '3'), ('range', '.001 .055'), ('solreflimit', '.01 1')]:
            changed = copy.deepcopy(self.new)
            changed.find('.//joint').set(attribute, value)
            self.model.write_text(ET.tostring(changed, encoding='unicode'))
            with self.assertRaises(ValueError):
                self.validate()
        changed = copy.deepcopy(self.new)
        changed.find('.//body').set('pos', '1 0 0')
        self.model.write_text(ET.tostring(changed, encoding='unicode'))
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == '__main__':
    unittest.main()

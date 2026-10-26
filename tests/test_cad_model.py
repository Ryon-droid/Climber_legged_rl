"""Real CAD route must preserve physical input and refuse unsupported training."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ('mujoco','scipy','yaml','numpy'))

class DefaultRouteTests(unittest.TestCase):
    def test_train_defaults_to_cad(self):
        tree=ast.parse((ROOT/'scripts/train.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='--backend':
                default=next(k.value.value for k in node.keywords if k.arg=='default')
                self.assertEqual(default,'cad')
                break
        else:
            self.fail('No explicit backend selection')

@unittest.skipUnless(AVAILABLE,'MuJoCo validation dependencies absent')
class CadPreservationTests(unittest.TestCase):
    def test_adapter_preserves_geometry_inertia_and_closures(self):
        from climber_legged_rl.cad_model import diagnostic_tree
        original=ET.parse(ROOT/'sim2sim/climber_mjcf/robot.xml').getroot()
        adapted=diagnostic_tree(ROOT/'sim2sim/climber_mjcf/robot.xml')
        self.assertEqual([b.attrib for b in original.findall('.//body')], [b.attrib for b in adapted.findall('.//body')])
        for tag in ('inertial','site'):
            self.assertEqual([e.attrib for e in original.findall('.//'+tag)], [e.attrib for e in adapted.findall('.//'+tag)])
        self.assertEqual(ET.tostring(original.find('equality')),ET.tostring(adapted.find('equality')))
        for a,b in zip(original.findall('.//body/joint'),adapted.findall('.//body/joint')):
            for key in ('name','type','pos','axis','range','limited'):
                self.assertEqual(a.get(key),b.get(key))
        self.assertEqual(len(original.findall('.//geom')),len(adapted.findall('.//geom')))
        self.assertTrue(all(g.get('type')=='mesh' for g in adapted.findall('.//body/geom')))
        self.assertIsNone(adapted.find('actuator'))
        self.assertEqual(adapted.find('compiler').get('balanceinertia'),'false')

    def test_unscaled_source_compiles_and_zero_closes(self):
        import mujoco
        import numpy as np
        from climber_legged_rl.cad_model import diagnostic_tree,closure_vectors
        root=diagnostic_tree(ROOT/'sim2sim/climber_mjcf/robot.xml')
        m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        self.assertAlmostEqual(float(m.body_mass.sum()),7.7327391466,places=8)
        self.assertEqual(m.nu,0)
        self.assertEqual(m.neq,6)
        self.assertLess(float(np.linalg.norm(closure_vectors(m,d),axis=1).max()),1e-7)

    def test_independent_sweep_detects_left_branch_problem(self):
        import mujoco
        import yaml
        from climber_legged_rl.cad_model import diagnostic_tree,motion_probe
        cfg=yaml.safe_load((ROOT/'configs/robot/cad_model.yaml').read_text())
        root=diagnostic_tree(ROOT/'sim2sim/climber_mjcf/robot.xml')
        m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        report=motion_probe(m,cfg['audit'],cfg['expected_leg_joints'],cfg['expected_wheel_joints'])
        self.assertFalse(report['passed'])
        self.assertEqual(len(report['samples']),16)
        self.assertTrue(all(r['passed'] for r in report['samples'] if r['joint'].startswith('right')))
        self.assertTrue(any(not r['passed'] for r in report['samples'] if r['joint'].startswith('left')))

if __name__=='__main__':
    unittest.main()

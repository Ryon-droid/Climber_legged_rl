"""Regression checks for source geometry repair and non-release of dynamics."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
AVAILABLE=all(importlib.util.find_spec(n) is not None for n in ('mujoco','numpy','scipy','yaml'))

@unittest.skipUnless(AVAILABLE,'MuJoCo validation dependencies absent')
class CadRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from climber_legged_rl.cad_repair import build_repaired,DEFAULT_REPAIR_CONFIG
        cls.temp=tempfile.TemporaryDirectory()
        cfg=yaml.safe_load(DEFAULT_REPAIR_CONFIG.read_text());cfg['output_directory']=cls.temp.name
        config=Path(cls.temp.name)/'config.yaml';config.write_text(yaml.safe_dump(cfg))
        cls.path,cls.report=build_repaired(config)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_mesh_inertia_matches_analytic_solid_cube(self):
        import numpy as np
        from climber_legged_rl.cad_repair import mesh_inertial
        # Outward-oriented triangles of a unit cube centered at origin.
        v=np.array([[-.5,-.5,-.5],[.5,-.5,-.5],[.5,.5,-.5],[-.5,.5,-.5],[-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]])
        f=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
        inertia=mesh_inertial(v[f],6.,np.zeros(3))
        self.assertAlmostEqual(float(inertia.get('mass')),6.)
        np.testing.assert_allclose(np.fromstring(inertia.get('pos'),sep=' '),0,atol=1e-12)
        np.testing.assert_allclose(np.fromstring(inertia.get('fullinertia'),sep=' '),[1,1,1,0,0,0],atol=1e-12)

    def test_preserves_meshes_restores_right_rod_and_six_closures(self):
        r=self.report
        self.assertEqual(r['closure_count'],6)
        self.assertEqual(r['restored_part_ids'],['part_433'])
        self.assertEqual(len(r['part_mapping']),8)
        self.assertTrue(all(p['cad_zero_mesh_bbox_error_m']<2e-7 for p in r['part_mapping']))
        self.assertEqual(r['remaining_unassigned_part_count'],72)
        root=ET.parse(self.path).getroot()
        original=ET.parse(ROOT/'sim2sim/climber_mjcf/robot.xml').getroot()
        self.assertEqual(len(root.findall('.//body/geom')),len(original.findall('.//body/geom'))+1)
        for old in original.findall('.//body'):
            if old.get('name') in ('base_splink','base_spring link 2','right_spring_link1','right_spring_link2'):continue
            new=root.find('.//body[@name="'+old.get('name')+'"]')
            self.assertEqual(old.attrib,new.attrib)
            self.assertEqual(old.find('inertial').attrib,new.find('inertial').attrib)

    def test_sliders_are_passive_and_explicitly_uncalibrated(self):
        root=ET.parse(self.path).getroot()
        self.assertEqual(self.report['nu'],0)
        for side in ('left','right'):
            joint=root.find('.//joint[@name="'+side+'_gas_slide_joint"]')
            self.assertEqual(joint.get('type'),'slide')
            self.assertEqual(joint.get('limited'),'false')
        self.assertFalse(self.report['ready_for_training'])
        self.assertLess(self.report['candidate_mass_kg'],8.)

    def test_independent_motion_and_135mm_stance(self):
        self.assertEqual(len(self.report['motion_samples']),24)
        self.assertTrue(self.report['motion_passed'])
        self.assertTrue(self.report['stance']['passed'])
        self.assertLess(self.report['stance']['max_closure_error_m'],1e-7)
        for h in self.report['stance']['hip_wheel_vertical_m']:
            self.assertAlmostEqual(h,.135,places=7)

    def test_left_closure_is_on_existing_pin_axis(self):
        root=ET.parse(self.path).getroot()
        p=root.find('.//site[@name="closure_003_b_site"]')
        import numpy as np
        local=np.fromstring(p.get('pos'),sep=' ')
        np.testing.assert_allclose(local[:2],0,atol=1e-8)
        self.assertEqual(self.report['closure_anchor_repair']['constraint'],'closure_003')

    def test_continuous_symmetric_motion_has_no_branch_jumps(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('validate_repair',ROOT/'scripts/validate_cad_repair.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        report=module.validate(self.path,frames=101)
        self.assertTrue(report['passed'])
        self.assertLess(report['max_left_right_wheel_xz_difference_m'],1e-4)
        self.assertLess(report['max_step_after_first_frame'],.05)

    def test_source_hash_unchanged(self):
        from climber_legged_rl.cad_model import sha256
        self.assertEqual(self.report['source_sha256'],sha256(ROOT/'sim2sim/climber_mjcf/robot.xml'))

if __name__=='__main__':unittest.main()

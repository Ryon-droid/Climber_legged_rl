"""Incomplete or inconsistent physical inputs must never be treated as released."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
AVAILABLE=importlib.util.find_spec('numpy') is not None

@unittest.skipUnless(AVAILABLE,'NumPy absent')
class PhysicalInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('physical_inputs',ROOT/'scripts/check_physical_measurements.py')
        cls.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.module)

    def test_missing_files_are_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            report=self.module.check(folder)
        self.assertFalse(report['ready_for_dynamics_identification'])
        self.assertTrue(report['blockers'])

    def test_current_blank_measurements_do_not_release(self):
        model=ROOT/'outputs/cad_repaired/robot_cad_repaired.xml'
        if not model.exists():self.skipTest('Prepare CAD candidate first')
        report=self.module.check(ROOT/'configs/robot/measurements')
        self.assertFalse(report['ready_for_dynamics_identification'])
        self.assertFalse(report['hardware_ready'])
        self.assertEqual(report['completed_body_count'],0)
        self.assertEqual(report['expected_body_count'],23)

    def test_invalid_inertia_is_not_accepted(self):
        if not (ROOT/'outputs/cad_repaired/robot_cad_repaired.xml').exists():self.skipTest('Prepare CAD candidate first')
        source=ROOT/'configs/robot/measurements/rigid_bodies.csv'
        rows=list(csv.DictReader(source.open()));fields=list(rows[0])
        first=rows[0]
        for key in ['com_x_m','com_y_m','com_z_m','ixy_kg_m2','ixz_kg_m2','iyz_kg_m2']:first[key]='0'
        first.update(mass_kg='1',ixx_kg_m2='-1',iyy_kg_m2='1',izz_kg_m2='1',source='test invalid tensor')
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder)/'rigid_bodies.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            report=self.module.check(folder)
        self.assertFalse(report['ready_for_dynamics_identification'])
        self.assertEqual(report['completed_body_count'],0)
        self.assertTrue(any('positive definite' in s for s in report['blockers']))

if __name__=='__main__':unittest.main()

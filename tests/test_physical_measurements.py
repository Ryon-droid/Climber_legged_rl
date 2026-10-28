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

    def test_partial_measurements_accept_masses_without_releasing_dynamics(self):
        model=ROOT/'outputs/cad_repaired/robot_cad_repaired.xml'
        if not model.exists():self.skipTest('Prepare CAD candidate first')
        report=self.module.check(ROOT/'configs/robot/measurements')
        self.assertFalse(report['ready_for_dynamics_identification'])
        self.assertFalse(report['hardware_ready'])
        self.assertEqual(report['completed_body_count'],0)
        self.assertEqual(report['expected_body_count'],23)
        self.assertEqual(report['provided_mass_count'],5)
        self.assertAlmostEqual(report['provided_mass_subtotal_kg'],15.56)
        self.assertEqual(report['provided_body_masses_kg'],{
            'base':12.7,'left_link 3':.54,'right_link 3':.54,
            'left_link2':.89,'right_link2':.89})
        self.assertFalse(report['all_body_masses_provided'])
        for name in report['provided_body_masses_kg']:
            messages=[s for s in report['blockers'] if s.startswith('Missing body fields: '+name+': ')]
            self.assertEqual(len(messages),1)
            self.assertNotIn('mass_kg',messages[0])

    def test_complete_body_tensors_do_not_require_old_total_or_side_sums(self):
        model=ROOT/'outputs/cad_repaired/robot_cad_repaired.xml'
        if not model.exists():self.skipTest('Prepare CAD candidate first')
        with (ROOT/'configs/robot/measurements/rigid_bodies.csv').open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        for row in rows:
            row['mass_kg']=row['mass_kg'] or '0.1'
            for key in ['com_x_m','com_y_m','com_z_m','ixy_kg_m2','ixz_kg_m2','iyz_kg_m2']:
                row[key]='0'
            for key in ['ixx_kg_m2','iyy_kg_m2','izz_kg_m2']:
                row[key]='0.001'
            row['source']='synthetic test fixture, not robot measurements'
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder)/'rigid_bodies.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
            report=self.module.check(folder)
        self.assertEqual(report['completed_body_count'],23)
        self.assertAlmostEqual(report['completed_mass_sum_kg'],17.36)
        self.assertTrue(report['all_body_masses_provided'])
        self.assertFalse(any('15 kg' in s or '5.14' in s for s in report['blockers']))
        self.assertFalse(report['ready_for_dynamics_identification'])

    def test_invalid_inertia_is_not_accepted(self):
        if not (ROOT/'outputs/cad_repaired/robot_cad_repaired.xml').exists():self.skipTest('Prepare CAD candidate first')
        source=ROOT/'configs/robot/measurements/rigid_bodies.csv'
        with source.open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        fields=list(rows[0])
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

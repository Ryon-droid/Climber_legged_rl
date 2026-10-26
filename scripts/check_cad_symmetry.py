"""Compare user-confirmed bilateral mechanisms without changing CAD parameters."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ('left_link1', 'right_link1'),
    ('left_link2', 'right_link2'),
    ('left_link_rear_1', 'right_rear_link1'),
    ('left_link_reear2', 'right_rear_link2'),
    ('left_link_rear3', 'right_rear_link3'),
    ('left_link_rear4', 'right_rear_link 4'),
    ('base_linkother', 'right_other 1_link'),
    ('base_linkother2', 'right_other 2_link'),
    ('base_splink', 'right_spring_link2'),
    ('base_spring link 2', 'right_spring_link1'),
    ('left_link 3', 'right_link 3'),
]


def check():
    source = ROOT/'sim2sim/climber_mjcf/user_model.json'
    xml_source = ROOT/'sim2sim/climber_mjcf/robot.xml'
    model = json.loads(source.read_text())
    links = {link['name']: link for link in model['links']}
    joints = {joint['child']: joint for joint in model['joints']}
    counterpart = {'base': 'base', **dict(PAIRS)}
    rows = []
    for left, right in PAIRS:
        lj, rj = joints[left], joints[right]
        rows.append({
            'left': left, 'right': right,
            'pair_status': 'mechanism-role candidate; not certified inertial or part correspondence',
            'left_parent': lj['parent'], 'right_parent': rj['parent'],
            'expected_right_parent_from_left': counterpart[lj['parent']],
            'parent_mapping_matches': counterpart[lj['parent']] == rj['parent'],
            'left_export_mass_kg': links[left]['mass'],
            'right_export_mass_kg': links[right]['mass'],
            'left_part_count': len(links[left]['part_ids']),
            'right_part_count': len(links[right]['part_ids']),
        })
    xml = ET.parse(xml_source).getroot()
    direct_mass = sum(float(e.get('mass')) for e in xml.findall('.//body/inertial'))
    base = xml.find('./worldbody/body')
    exported_base_mass = float(base.find('inertial').get('mass'))
    exported_right_mass = sum(float(body.find('inertial').get('mass'))
                              for branch in base.findall('body') if branch.get('name', '').startswith('right')
                              for body in [branch, *branch.findall('.//body')] if body.find('inertial') is not None)
    closures = [{key: c.get(key) for key in ('name', 'link_a', 'link_b', 'anchor_mode')}
                for c in model['loop_constraints'] if c['name'] in ('closure_007', 'closure_008')]
    report = {
        'symmetry_basis': 'User confirms the physical chassis is bilaterally symmetric.',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'xml_sha256': hashlib.sha256(xml_source.read_bytes()).hexdigest(),
        'body_pairs': rows,
        'parent_mapping_mismatches': [r for r in rows if not r['parent_mapping_matches']],
        'spring_closures': closures,
        'original_export_mass_kg': direct_mass,
        'total_if_both_side_totals_used_larger_export_without_redistribution_kg':
            exported_base_mass + 2 * exported_right_mass,
        'limitations': [
            'Role pairs are audit candidates. Mesh/physical correspondence must be checked before mirroring.',
            'Neither side is a certified reference: the right spring closure stays within one hip branch.',
            'Symmetry alone does not identify missing masses, COM, inertia or spring force/travel.',
            'Current local motion checks are diagnostic, not physical-model certification.',
            'Encoder zero, direction, feedback ratio and output limits remain per-channel bench measurements.',
        ],
        'source_modified': False,
        'training_started': False,
    }
    output = ROOT/'outputs/cad_model/symmetry_report.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    result = check()
    print(json.dumps({k: result[k] for k in ('parent_mapping_mismatches', 'spring_closures',
          'original_export_mass_kg', 'total_if_both_side_totals_used_larger_export_without_redistribution_kg',
          'source_modified', 'training_started')}, ensure_ascii=False, indent=2))

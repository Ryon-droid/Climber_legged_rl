"""Physical pin-to-pin gas-spring lengths and explicit slider travel limits."""
from __future__ import annotations
import numpy as np


def eye_center_distance(model, data, side):
    """Cylinder body origin and rod pin site are the reconstructed mounting eyes."""
    return float(np.linalg.norm(data.site(side + '_gas_rod_pin').xpos -
                                data.body(side + '_gas_cylinder').xpos))


def configured_eye_limits(config, required=False):
    limits = config['spring'].get('eye_center_limits_m')
    if limits is None:
        if required:
            raise ValueError('Gas-spring compressed/extended pin-center lengths are required; '
                             'set spring.eye_center_limits_m for left and right before preparing a prone pose')
        return None
    if set(limits) != {'left', 'right'}:
        raise ValueError('Gas-spring eye_center_limits_m requires left and right ranges')
    for side, bounds in limits.items():
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or bounds[1] is None or not np.isfinite(bounds[1]) or bounds[1] <= 0
                or (bounds[0] is not None and
                    (not np.isfinite(bounds[0]) or not 0 < bounds[0] < bounds[1]))):
            raise ValueError('Invalid gas-spring pin-center length range: ' + side)
    return limits


def apply_eye_limits(root, model, qpos, config):
    """L = CAD zero pin distance + slider q; never interpret q as total length."""
    limits = configured_eye_limits(config, required=config.get('getup', {}).get('enabled', False))
    if limits is None:
        return {}
    report = {}
    for side, bounds in limits.items():
        zero = float(np.linalg.norm(model.body(side + '_gas_rod').pos))
        # Unknown compressed length stays unknown. The zero-length lower bound
        # only prevents inversion through the cylinder pin; it is not a hardware stop.
        minimum = 0. if bounds[0] is None else bounds[0]
        slide_range = np.asarray([minimum, bounds[1]]) - zero
        joint_id = model.joint(side + '_gas_slide_joint').id
        value = float(qpos[int(model.jnt_qposadr[joint_id])])
        if not slide_range[0] <= value <= slide_range[1]:
            raise ValueError('Standing pose gas-spring length is outside supplied limits: ' + side)
        joint = root.find('.//joint[@name="' + side + '_gas_slide_joint"]')
        joint.set('limited', 'true')
        joint.set('range', ' '.join(format(v, '.15g') for v in slide_range))
        # Default MuJoCo soft limits stretch by centimeters under the nominal 450 N.
        # Stiffer numerical end stops retain a small, explicitly measured compliance.
        joint.set('solreflimit', '0.002 1')
        joint.set('solimplimit', '0.999 0.9999 0.0001 0.5 2')
        report[side] = dict(eye_center_limits_m=list(bounds), cad_zero_eye_distance_m=zero,
                            slider_range_m=slide_range.tolist(), standing_eye_distance_m=zero+value,
                            stop_solver_settings='simulation end stop, not measured stiffness',
                            compressed_length_verified=bounds[0] is not None)
    return report

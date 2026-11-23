"""Explicit per-run motor limits without rewriting historical model assets."""
import math


def apply_standing_motor_limit(env, standing_leg_torque_limit_Nm=None, standing_wheel_kv=None):
    """Opt-in standing ceiling and wheel gain; retain the rising envelope."""
    base = env.unwrapped
    envelope = base.envelope
    previous = float(envelope['standing_torque_limit_Nm'])
    requested = previous if standing_leg_torque_limit_Nm is None else float(standing_leg_torque_limit_Nm)
    rated = float(envelope['rated_torque_Nm'])
    peak = float(base.config['control']['leg_torque_limit_Nm'])
    if not math.isfinite(requested) or not 0 < requested <= rated <= peak:
        raise ValueError('Standing torque must be positive and no greater than rated/peak torque')
    if standing_wheel_kv is not None:
        standing_wheel_kv = float(standing_wheel_kv)
        if not math.isfinite(standing_wheel_kv) or standing_wheel_kv <= 0:
            raise ValueError('Standing wheel speed gain must be finite and positive')
    for low, high in base.model.actuator_ctrlrange[:4]:
        if float(low) > -requested or float(high) < requested:
            raise ValueError('Standing torque exceeds model actuator limits')
    if standing_leg_torque_limit_Nm is not None or standing_wheel_kv is not None:
        # Do not mutate config/getup: those identify the original validated assets.
        base.envelope = dict(envelope, standing_torque_limit_Nm=requested)
        if standing_wheel_kv is not None:
            base.envelope['standing_wheel_kv'] = standing_wheel_kv
        base._set_motor_envelope()
    return dict(standing_leg_torque_limit_Nm=requested,
        configured_standing_leg_torque_limit_Nm=float(base.config['getup']['policy_control']['standing_torque_limit_Nm']),
        runtime_override=standing_leg_torque_limit_Nm is not None or standing_wheel_kv is not None,
        rising_rated_leg_torque_Nm=rated, rising_peak_leg_torque_Nm=peak,
        peak_window_seconds=float(envelope['peak_window_seconds']),
        wheel_torque_limits_Nm=[float(x) for x in base.torque_limits[4:]],
        torque_reference='joint_output', pd_gains_unchanged=standing_wheel_kv is None,
        standing_wheel_kv=float(base.envelope.get('standing_wheel_kv', base.rise_wheel_kv)),
        rising_wheel_kv=float(base.rise_wheel_kv))

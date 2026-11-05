"""Optional broad recovery penalties alongside existing precise hold rewards."""
import numpy as np


class BalanceRecoveryReward:
    def __init__(self, settings):
        self.enabled = settings.get('enabled', False)
        self.settings = settings
        self.attitude_terms = [name for name in ('tilt', 'tilt_rate')
                               if name + '_scale' in settings or name + '_weight' in settings]
        if self.enabled:
            for name in ['position', 'velocity', 'heading', 'yaw_rate', *self.attitude_terms]:
                for suffix in ('scale', 'weight'):
                    value = settings.get(name + '_' + suffix)
                    if value is None or not np.isfinite(value) or value <= 0:
                        raise ValueError('Invalid balance recovery parameter: ' + name + '_' + suffix)

    def terms(self, displacement, velocity, heading, yaw_rate, *, tilt=None, tilt_rate=None):
        if not self.enabled:
            return {}
        errors = {'position': np.linalg.norm(displacement), 'velocity': np.linalg.norm(velocity),
                  'heading': abs(heading), 'yaw_rate': abs(yaw_rate)}
        for name, value in (('tilt', tilt), ('tilt_rate', tilt_rate)):
            if name in self.attitude_terms:
                if value is None or not np.isfinite(value) or value < 0:
                    raise ValueError('Recovery attitude measurement must be finite and nonnegative: ' + name)
                errors[name] = value
        terms = {}
        for name, error in errors.items():
            scaled = error / self.settings[name + '_scale']
            # Huber: quadratic near zero, linear outside. No exponential collapse
            # at larger errors and no unbounded quadratic spike after a collision.
            huber = .5 * scaled**2 if scaled <= 1. else scaled - .5
            terms['recovery_' + name] = -float(huber * self.settings[name + '_weight'])
        return terms

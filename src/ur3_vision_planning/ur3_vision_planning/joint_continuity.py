"""Deterministic winding selection and fail-closed trajectory checks (no IO)."""

import itertools
import math

from .environment_manager import EnvironmentError, finite_number


def checked_arm(values, names, model):
    result = {}
    for name in names:
        value = values.get(name)
        if not finite_number(value):
            raise EnvironmentError(f'INVALID_ARM_STATE: missing/nonfinite {name}')
        if not model[name]['lower'] <= value <= model[name]['upper']:
            raise EnvironmentError(f'JOINT_LIMIT_VIOLATION: {name}={value}')
        result[name] = value
    return result


def branch_candidates(raw, previous, names, model, margin=0., maximum=64):
    """Sort equivalent windings by actual command distance, not modulo distance.

    Never change a prismatic joint. For continuous joints preserve the seed's
    winding; wrapping every response to [-pi,pi] would introduce controller jumps.
    """
    checked_arm(previous, names, model)
    choices = []
    for name in names:
        q = raw.get(name)
        if not finite_number(q):
            raise EnvironmentError(f'INVALID_IK_RESPONSE: {name}')
        spec = model[name]
        if spec.get('type', 'revolute') not in ('revolute', 'continuous'):
            candidates = [q]  # No +/-2pi transformation of translations.
        else:
            nearest = round((previous[name]-q)/(2*math.pi))
            low, high = spec['lower'], spec['upper']
            if math.isfinite(low) and math.isfinite(high):
                first = math.ceil((low+margin-q)/(2*math.pi))
                last = math.floor((high-margin-q)/(2*math.pi))
                candidates = [q+2*math.pi*k for k in range(first, last+1)]
            else:
                candidates = [q+2*math.pi*nearest]
        candidates = [v for v in candidates if spec['lower']+margin <= v <= spec['upper']-margin]
        if not candidates:
            raise EnvironmentError(f'NO_IN_LIMIT_EQUIVALENT_BRANCH: {name}={q}')
        choices.append(sorted(candidates, key=lambda v: (abs(v-previous[name]), v)))
    # UR arm has only a few legal windings per joint. Reject unreasonable models
    # instead of generating an unbounded Cartesian product.
    if math.prod(len(v) for v in choices) > 4096:
        raise EnvironmentError('TOO_MANY_EQUIVALENT_BRANCHES')
    candidates = [dict(zip(names, values)) for values in itertools.product(*choices)]
    return sorted(candidates, key=lambda v: (sum((v[n]-previous[n])**2 for n in names),
                                            tuple(v[n] for n in names)))[:maximum]


def deltas(first, second, names):
    physical = {n: second[n]-first[n] for n in names}
    angular = {n: math.atan2(math.sin(v), math.cos(v)) for n, v in physical.items()}
    return physical, angular


def check_trajectory(trajectory, start, names, model, start_tolerance=.03,
                     point_delta_limit=.5, goal=None, goal_tolerance=.01):
    """Do NOT repair/rewrite a planned trajectory: a winding jump is a rejection."""
    joints = list(trajectory.joint_names)
    if len(joints) != len(names) or set(joints) != set(names) or len(set(joints)) != len(joints):
        raise EnvironmentError('TRAJECTORY_WRONG_JOINT_NAMES')
    if len(trajectory.points) < 2:
        raise EnvironmentError('TRAJECTORY_NO_CONNECTING_PATH')
    previous, previous_time = checked_arm(start, names, model), -1.
    maximum = 0.
    for index, point in enumerate(trajectory.points):
        if len(point.positions) != len(joints):
            raise EnvironmentError(f'TRAJECTORY_MISSING_POSITIONS: {index}')
        values = checked_arm(dict(zip(joints, point.positions)), names, model)
        for field in ('velocities', 'accelerations', 'effort'):
            vector = list(getattr(point, field))
            if vector and (len(vector) != len(joints) or not all(finite_number(v) for v in vector)):
                raise EnvironmentError(f'TRAJECTORY_INVALID_{field.upper()}: {index}')
        stamp = point.time_from_start
        t = stamp.sec+stamp.nanosec*1e-9
        if stamp.sec < 0 or not 0 <= stamp.nanosec < 1000000000 or t <= previous_time:
            raise EnvironmentError(f'TRAJECTORY_NONINCREASING_TIME: {index}')
        step = max(abs(values[n]-previous[n]) for n in names)
        if index == 0 and step > start_tolerance:
            raise EnvironmentError(f'TRAJECTORY_START_MISMATCH: max_error={step:.9f}')
        if index and step > point_delta_limit:
            raise EnvironmentError(f'TRAJECTORY_JOINT_JUMP: point={index}, delta={step:.9f}')
        maximum = max(maximum, step)
        previous, previous_time = values, t
    if goal is not None and max(abs(previous[n]-goal[n]) for n in names) > goal_tolerance:
        raise EnvironmentError('TRAJECTORY_ENDPOINT_MISMATCH')
    return previous, maximum

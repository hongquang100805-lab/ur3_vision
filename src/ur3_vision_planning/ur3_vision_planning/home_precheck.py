"""HOME joint/model validation; no ROS clients or actuator calls."""

import copy
import math
import xml.etree.ElementTree as ET

from moveit_msgs.msg import RobotState

from .environment_manager import EnvironmentError, finite_number


def home_joint_model(urdf_xml, srdf_xml, group_name, joint_names, overrides=None):
    """Use the running move_group model, not a guessed UR3/UR3e limits file."""
    try:
        urdf, srdf = ET.fromstring(urdf_xml), ET.fromstring(srdf_xml)
        joints = {item.attrib['name']: item for item in urdf.findall('joint')}
        parents = {item.find('child').attrib['link']: item for item in joints.values()}
        groups = {item.attrib['name']: item for item in srdf.findall('group')}
        def members(name, visited):
            if name in visited or name not in groups:
                raise ValueError(f'invalid/cyclic SRDF group {name}')
            result = set()
            for item in groups[name]:
                if item.tag == 'joint':
                    if joints[item.attrib['name']].attrib['type'] != 'fixed':
                        result.add(item.attrib['name'])
                elif item.tag == 'group':
                    result.update(members(item.attrib['name'], visited | {name}))
                elif item.tag == 'chain':
                    link, base = item.attrib['tip_link'], item.attrib['base_link']
                    seen = set()
                    while link != base:
                        if link in seen:
                            raise ValueError('cyclic URDF chain')
                        seen.add(link)
                        joint = parents[link]
                        if joint.attrib['type'] != 'fixed':
                            result.add(joint.attrib['name'])
                        link = joint.find('parent').attrib['link']
                else:
                    raise ValueError(f'unsupported SRDF group element {item.tag}')
            return result
        if len(joint_names) != 6 or len(set(joint_names)) != 6 or members(group_name, set()) != set(joint_names):
            raise ValueError('planning group must contain exactly the six named arm joints, no gripper')
        model = {}
        for name in joint_names:
            joint = joints[name]
            continuous = joint.attrib['type'] == 'continuous'
            if joint.attrib['type'] not in ('revolute', 'continuous'):
                raise ValueError(f'{name}: expected revolute/continuous joint')
            lower, upper = -math.inf, math.inf
            if not continuous:
                limit = joint.find('limit')
                lower, upper = float(limit.attrib['lower']), float(limit.attrib['upper'])
                if not finite_number(lower) or not finite_number(upper):
                    raise ValueError(f'{name}: invalid URDF bounds')
                safety = joint.find('safety_controller')
                if safety is not None:
                    soft_lower = float(safety.attrib.get('soft_lower_limit', lower))
                    soft_upper = float(safety.attrib.get('soft_upper_limit', upper))
                    if not finite_number(soft_lower) or not finite_number(soft_upper):
                        raise ValueError(f'{name}: invalid URDF safety bounds')
                    lower, upper = max(lower, soft_lower), min(upper, soft_upper)
            override = (overrides or {}).get(name, {})
            if override.get('has_position_limits') is True:
                low, high = override.get('min_position'), override.get('max_position')
                if not finite_number(low) or not finite_number(high):
                    raise ValueError(f'{name}: incomplete MoveIt position-limit override')
                lower, upper = max(lower, low), min(upper, high)
            else:
                # Jazzy RobotModelLoader also applies individual min/max values
                # without a has_position_limits flag; unset values retain URDF.
                for field, side in (('min_position', 'lower'), ('max_position', 'upper')):
                    if field in override:
                        value = override[field]
                        if not finite_number(value) or continuous:
                            raise ValueError(f'{name}: invalid {field} override for joint type')
                        if side == 'lower':
                            lower = max(lower, value)
                        else:
                            upper = min(upper, value)
            if lower > upper:
                raise ValueError(f'{name}: contradictory URDF/MoveIt limits')
            model[name] = {'type': joint.attrib['type'], 'continuous': continuous, 'lower': lower, 'upper': upper}
        return model
    except (ET.ParseError, KeyError, ValueError, AttributeError) as error:
        raise EnvironmentError(f'HOME_MODEL_INVALID: {error}') from error


def joint_values(state):
    names, values = list(state.joint_state.name), list(state.joint_state.position)
    if len(names) != len(values) or len(set(names)) != len(names) or not all(finite_number(v) for v in values):
        raise EnvironmentError('HOME_JOINT_STATE_INVALID: duplicate names, missing or nonfinite values')
    return dict(zip(names, values))


def normalize_arm(values, names, model, label):
    result = {}
    for name in names:
        if name not in values or not finite_number(values[name]):
            raise EnvironmentError(f'HOME_JOINT_STATE_INVALID: {label}: missing/nonfinite {name}')
        value = values[name]
        if model[name]['continuous']:
            value = math.atan2(math.sin(value), math.cos(value))
        # Never modulo a bounded revolute joint: its winding is a real limit.
        if not model[name]['lower'] <= value <= model[name]['upper']:
            raise EnvironmentError(f'HOME_JOINT_LIMIT_VIOLATION: {label}: {name}={value}, '
                                   f'limits=[{model[name]["lower"]},{model[name]["upper"]}]')
        result[name] = value
    return result


def joint_errors(actual, desired, names, model):
    errors = {}
    for name in names:
        delta = actual[name] - desired[name]
        errors[name] = abs(math.atan2(math.sin(delta), math.cos(delta))) if model[name]['continuous'] else abs(delta)
    return errors


def explicit_state(seed, arm_values, names):
    """Full robot seed for collision geometry; arm constraints remain six only.

    Keep auxiliary joints (e.g. open fingers) from the SIMULATED seed, never
    replace them with actual current state or put them in the arm goal/group.
    """
    values = joint_values(seed)
    values.update(arm_values)
    order = list(names) + [name for name in values if name not in names]
    state = RobotState(is_diff=False)
    state.joint_state = copy.deepcopy(seed.joint_state)
    state.joint_state.name = order
    state.joint_state.position = [values[name] for name in order]
    state.joint_state.velocity, state.joint_state.effort = [], []
    state.multi_dof_joint_state = copy.deepcopy(seed.multi_dof_joint_state)
    return state

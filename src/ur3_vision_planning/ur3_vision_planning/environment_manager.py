"""Camera environment snapshots and deterministic dry-run goal resolution.

No robot action clients, simulator queries, pose commands or LLM calls.
"""

import copy
import json
import math
import os
import time

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import yaml

from .task_validator import TaskValidator


class EnvironmentError(ValueError):
    pass


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def parse_json(payload):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EnvironmentError(f'INVALID_JSON: duplicate key {key!r}')
            result[key] = value
        return result

    def invalid_constant(value):
        raise EnvironmentError(f'INVALID_JSON: non-finite constant {value}')

    try:
        return json.loads(payload, object_pairs_hook=unique_keys,
                          parse_constant=invalid_constant) if isinstance(payload, str) else copy.deepcopy(payload)
    except (json.JSONDecodeError, TypeError) as error:
        raise EnvironmentError(f'INVALID_JSON: {error}') from error


class DryRunPlanValidator:
    ALLOWED_SKILLS = TaskValidator.ALLOWED_SKILLS
    ALLOWED_ZONES = TaskValidator.ALLOWED_ZONES
    ALLOWED_OBJECTS = frozenset(('red_cube', 'yellow_cube', 'blue_cube',
                                 'green_cube', 'purple_cube'))
    REQUIRED_FIELDS = {
        'home': {'skill'}, 'pick': {'skill', 'object'},
        'place': {'skill', 'object', 'destination'},
    }

    def validate(self, payload, trusted_temporary_positions, environment_state, goal=None):
        try:
            data = parse_json(payload)
            if not isinstance(data, dict) or set(data) != {'plan', 'temporary_positions'}:
                raise EnvironmentError('INVALID_PLAN: root requires only plan and temporary_positions')
            positions = data['temporary_positions']
            if not isinstance(positions, dict) or positions != trusted_temporary_positions:
                raise EnvironmentError('UNTRUSTED_COORDINATES: temporary positions must exactly match manager selection')
            for name, point in positions.items():
                if (not isinstance(name, str) or name in self.ALLOWED_ZONES
                        or not isinstance(point, dict) or set(point) != {'frame_id', 'x', 'y', 'z'}
                        or point['frame_id'] != 'base_link'
                        or not all(finite_number(point[axis]) for axis in ('x', 'y', 'z'))):
                    raise EnvironmentError('UNTRUSTED_COORDINATES: invalid manager destination')
            steps = data['plan']
            if not isinstance(steps, list) or not steps:
                raise EnvironmentError('INVALID_PLAN: plan must be a non-empty list')
            occupancy = {name: zone['object'] for name, zone in environment_state['zones'].items()
                         if zone['occupied']}
            held = None
            home_seen = False
            used_slots = set()
            for index, step in enumerate(steps):
                if not isinstance(step, dict) or not isinstance(step.get('skill'), str):
                    raise EnvironmentError(f'INVALID_PLAN: step {index+1}')
                skill = step['skill']
                if skill not in self.ALLOWED_SKILLS or set(step) != self.REQUIRED_FIELDS[skill]:
                    raise EnvironmentError(f'INVALID_SKILL_OR_FIELDS: step {index+1}')
                if home_seen:
                    raise EnvironmentError('INVALID_ORDER: motion after home')
                if skill == 'home':
                    if held is not None or index != len(steps)-1:
                        raise EnvironmentError('INVALID_ORDER: home must be last with no held object')
                    home_seen = True
                    continue
                obj = step['object']
                if not isinstance(obj, str) or obj not in self.ALLOWED_OBJECTS:
                    raise EnvironmentError(f'INVALID_OBJECT: {obj!r}')
                if skill == 'pick':
                    if held is not None:
                        raise EnvironmentError(f'INVALID_ORDER: already holding {held}')
                    if not environment_state['objects'][obj]['detected']:
                        raise EnvironmentError(f'OBJECT_NOT_DETECTED: {obj}')
                    held = obj
                    occupancy = {dest: name for dest, name in occupancy.items() if name != obj}
                else:
                    destination = step['destination']
                    if (not isinstance(destination, str)
                            or destination not in self.ALLOWED_ZONES | frozenset(positions)):
                        raise EnvironmentError(f'INVALID_DESTINATION: {destination!r}')
                    if held != obj:
                        raise EnvironmentError(f'INVALID_ORDER: cannot place {obj}; held={held}')
                    if occupancy.get(destination) not in (None, obj):
                        raise EnvironmentError(f'DESTINATION_OCCUPIED: {destination}')
                    occupancy[destination] = obj
                    if destination in positions:
                        used_slots.add(destination)
                    held = None
            if held is not None or not home_seen:
                raise EnvironmentError('INVALID_ORDER: plan must release object and end with home')
            if used_slots != set(positions):
                raise EnvironmentError('INVALID_PLAN: unused or missing temporary destination')
            if goal is not None:
                goal = validate_structured_goal(goal)
                obj, zone = goal['object'], goal['target_zone']
                occupant = environment_state['zones'][zone]['object']
                expected = []
                already_satisfied = environment_state['objects'][obj]['zone'] == zone
                if not already_satisfied:
                    if occupant is not None and occupant != obj:
                        if len(positions) != 1:
                            raise EnvironmentError('INVALID_PLAN: one trusted temporary slot required')
                        slot = next(iter(positions))
                        expected.extend([{'skill': 'pick', 'object': occupant},
                                         {'skill': 'place', 'object': occupant, 'destination': slot}])
                    elif positions:
                        raise EnvironmentError('INVALID_PLAN: unnecessary temporary slot')
                    expected.extend([{'skill': 'pick', 'object': obj},
                                     {'skill': 'place', 'object': obj, 'destination': zone}])
                expected.append({'skill': 'home'})
                if steps != expected or occupancy.get(zone) != obj:
                    raise EnvironmentError('GOAL_MISMATCH: plan must clear occupant first and satisfy the exact goal')
            return True, 'DRY-RUN PLAN VALID'
        except (EnvironmentError, KeyError, TypeError) as error:
            return False, str(error)


def validate_structured_goal(payload):
    """Strictly separate untrusted LLM intent from manager-owned coordinates."""
    goal = parse_json(payload)
    if not isinstance(goal, dict) or set(goal) != {'object', 'target_zone'}:
        raise EnvironmentError('INVALID_GOAL: only object and target_zone are permitted; no coordinates or skills')
    if not isinstance(goal['object'], str) or goal['object'] not in DryRunPlanValidator.ALLOWED_OBJECTS:
        raise EnvironmentError('INVALID_GOAL_OBJECT: expected one of five allowed cubes')
    if not isinstance(goal['target_zone'], str) or goal['target_zone'] not in DryRunPlanValidator.ALLOWED_ZONES:
        raise EnvironmentError('INVALID_GOAL_ZONE: expected zone_a, zone_b or zone_c')
    return goal


class EnvironmentPlanner:
    """Pure goal resolver; object locations always come from camera state."""

    def __init__(self, scene, config, camera):
        self.scene = scene
        self.config = config
        self.camera = camera
        self.validator = DryRunPlanValidator()
        self.last_rejections = {}
        self.selected_temporary_positions = {}

    def check_environment(self, state):
        if state is None:
            raise EnvironmentError('ENVIRONMENT_NOT_RECEIVED')
        if not isinstance(state, dict):
            raise EnvironmentError('INVALID_ENVIRONMENT: root must be an object')
        if state.get('status') != 'ok' or state.get('stale') is not False:
            raise EnvironmentError(f'ENVIRONMENT_NOT_READY: status={state.get("status")!r}, stale={state.get("stale")!r}')
        if state.get('frame_id') != self.config['environment']['target_frame']:
            raise EnvironmentError('INVALID_FRAME: expected base_link')
        if not finite_number(state.get('stamp')) or state['stamp'] < 0:
            raise EnvironmentError('INVALID_ENVIRONMENT: invalid measurement stamp')
        if not finite_number(state.get('published_at')):
            raise EnvironmentError('INVALID_ENVIRONMENT: invalid published_at')
        if not 0 <= state['published_at']-state['stamp'] <= self.config['environment']['measurement_timeout_s']:
            raise EnvironmentError('ENVIRONMENT_STALE: measurement stamp is old')
        objects = state.get('objects')
        zones = state.get('zones')
        if not isinstance(objects, dict) or set(objects) != self.validator.ALLOWED_OBJECTS:
            raise EnvironmentError('INVALID_ENVIRONMENT: five known camera objects required')
        if not isinstance(zones, dict) or set(zones) != self.validator.ALLOWED_ZONES:
            raise EnvironmentError('INVALID_ENVIRONMENT: three known zones required')
        for name, obj in objects.items():
            if not isinstance(obj, dict) or obj.get('detected') is not True:
                raise EnvironmentError(f'OBJECT_NOT_DETECTED: {name}')
            if obj.get('stale') is not False:
                raise EnvironmentError(f'OBJECT_STALE: {name}')
            if (not finite_number(obj.get('age_s')) or obj['age_s'] < 0
                    or obj['age_s'] > self.config['environment']['object_timeout_s']):
                raise EnvironmentError(f'OBJECT_STALE: {name} age_s invalid or expired')
            if (not finite_number(obj.get('stamp')) or obj['stamp'] > state['stamp']
                    or state['stamp']-obj['stamp'] > self.config['environment']['object_timeout_s']):
                raise EnvironmentError(f'OBJECT_STALE: {name} stamp invalid or expired')
            position = obj.get('position')
            if (not isinstance(position, dict) or set(position) != {'x', 'y', 'z'}
                    or not all(finite_number(position[axis]) for axis in position)):
                raise EnvironmentError(f'INVALID_POSITION: {name}')
            if obj.get('zone') is not None and (not isinstance(obj['zone'], str)
                                               or obj['zone'] not in self.validator.ALLOWED_ZONES):
                raise EnvironmentError(f'INVALID_OBJECT_ZONE: {name}')
        for name, zone in zones.items():
            if not isinstance(zone, dict) or not isinstance(zone.get('objects'), list):
                raise EnvironmentError(f'INVALID_ZONE_STATE: {name}')
            occupants = zone['objects']
            if len(occupants) > 1:
                raise EnvironmentError(f'MULTIPLE_ZONE_OCCUPANTS: {name}: {occupants}')
            if zone.get('stale') is not False or type(zone.get('occupied')) is not bool:
                raise EnvironmentError(f'ZONE_STALE_OR_UNKNOWN: {name}')
            expected = [obj for obj, item in objects.items() if item['zone'] == name]
            if (occupants != expected or zone['occupied'] != bool(expected)
                    or zone.get('object') != (expected[0] if expected else None)
                    or zone.get('status') != ('occupied' if expected else 'empty')):
                raise EnvironmentError(f'INCONSISTENT_ZONE_OCCUPANCY: {name}')
        return state

    @staticmethod
    def rpy_rotation(rpy):
        r, p, y = rpy
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                         [-sp, cp*sr, cp*cr]])

    def candidate_rejection(self, point, state):
        if (not isinstance(point, dict) or set(point) != {'frame_id', 'x', 'y', 'z'}
                or point['frame_id'] != self.config['environment']['target_frame']
                or not all(finite_number(point[axis]) for axis in ('x', 'y', 'z'))):
            return 'invalid coordinate/frame'
        safety = self.config['safety']
        size = self.scene['cube_size']
        table = self.scene['table']
        base_height = self.scene.get('robot_base_height', 0.0)
        support_z = table['center'][2]+table['size'][2]/2-base_height+size[2]/2
        if abs(point['z']-support_z) > safety['support_z_tolerance_m']:
            return 'not on tabletop at cube-centre height'
        for i, axis in enumerate(('x', 'y')):
            if abs(point[axis]-table['center'][i])+size[i]/2+safety['table_edge_margin_m'] > table['size'][i]/2:
                return 'cube footprint outside safe table boundary'
        radius = math.hypot(point['x'], point['y'])
        if not safety['min_base_distance_m'] <= radius <= safety['max_base_distance_m']:
            return 'outside configured robot work radius'
        # Expanded zone rectangles account for both full marker and cube footprint.
        for name, zone in self.scene['zones'].items():
            if all(abs(point[axis]-zone['position'][i]) <= self.scene['zone_marker_size'][i]/2
                   + size[i]/2+safety['zone_edge_clearance_m'] for i, axis in enumerate(('x', 'y'))):
                return f'too close to zone {name}'
        for name, obj in state['objects'].items():
            position = obj['position']
            if math.hypot(point['x']-position['x'], point['y']-position['y']) < safety['object_distance_m']:
                return f'too close to camera object {name}'
        camera = self.camera
        rotation = self.rpy_rotation(camera['rpy']) @ self.rpy_rotation(camera['optical_rpy'])
        tangent_x = math.tan(camera['horizontal_fov']/2)
        tangent_y = tangent_x*camera['height']/camera['width']
        for dx in (-size[0]/2, size[0]/2):
            for dy in (-size[1]/2, size[1]/2):
                world_point = np.array([point['x']+dx, point['y']+dy, point['z']+size[2]/2+base_height])
                optical = rotation.T @ (world_point-np.array(camera['position']))
                if (not camera['near_clip'] <= optical[2] <= camera['far_clip']
                        or abs(optical[0]/optical[2]) > tangent_x
                        or abs(optical[1]/optical[2]) > tangent_y):
                    return 'cube outside configured camera frustum'
        return None

    def free_temporary_candidates(self, environment_state):
        """Tier 1 only: camera geometry, never a claim of IK reachability."""
        state = self.check_environment(environment_state)
        self.last_rejections = {}
        candidates = {}
        for name, point in self.config['temporary_positions'].items():
            reason = self.candidate_rejection(point, state)
            if reason:
                self.last_rejections[name] = reason
                continue
            candidates[name] = copy.deepcopy(point)
        return candidates

    def find_free_temporary_position(self, environment_state):
        candidates = self.free_temporary_candidates(environment_state)
        if candidates:
            return next(iter(candidates.items()))
        raise EnvironmentError(f'NO_FREE_TEMPORARY_POSITION: {self.last_rejections}')

    def adopt_prechecked_plan(self, plan, goal, state, candidates):
        """Only manager-provided slots may replace the provisional destination."""
        for name, point in plan['temporary_positions'].items():
            if candidates.get(name) != point or self.config['temporary_positions'].get(name) != point:
                raise EnvironmentError('UNTRUSTED_COORDINATES: precheck selected an unknown slot')
        previous = self.selected_temporary_positions
        self.selected_temporary_positions = copy.deepcopy(plan['temporary_positions'])
        valid, reason = self.validate_plan(plan, goal, state)
        if not valid:
            self.selected_temporary_positions = previous
            raise EnvironmentError(f'PRECHECK_PLAN_INVALID: {reason}')

    def resolve_goal(self, object_name, target_zone, environment_state):
        self.selected_temporary_positions = {}
        state = self.check_environment(environment_state)
        if not isinstance(object_name, str) or object_name not in self.validator.ALLOWED_OBJECTS:
            raise EnvironmentError(f'INVALID_OBJECT: {object_name!r}')
        if not isinstance(target_zone, str) or target_zone not in self.validator.ALLOWED_ZONES:
            raise EnvironmentError(f'INVALID_TARGET_ZONE: {target_zone!r}')
        target = state['objects'][object_name]
        occupant = state['zones'][target_zone]['object']
        steps = []
        if target['zone'] != target_zone:
            if occupant is not None and occupant != object_name:
                slot, point = self.find_free_temporary_position(state)
                self.selected_temporary_positions = {slot: point}
                steps.extend([{'skill': 'pick', 'object': occupant},
                              {'skill': 'place', 'object': occupant, 'destination': slot}])
            steps.extend([{'skill': 'pick', 'object': object_name},
                          {'skill': 'place', 'object': object_name, 'destination': target_zone}])
        steps.append({'skill': 'home'})
        plan = {'plan': steps, 'temporary_positions': copy.deepcopy(self.selected_temporary_positions)}
        valid, message = self.validate_plan(
            plan, {'object': object_name, 'target_zone': target_zone}, state)
        if not valid:
            raise EnvironmentError(message)
        return plan

    def validate_plan(self, plan, goal, environment_state):
        """Recheck selected geometry using the same fresh camera snapshot."""
        try:
            state = self.check_environment(environment_state)
            goal = validate_structured_goal(goal)
            for name, point in self.selected_temporary_positions.items():
                if self.config['temporary_positions'].get(name) != point:
                    raise EnvironmentError('UNTRUSTED_COORDINATES: slot is not from manager YAML')
                reason = self.candidate_rejection(point, state)
                if reason:
                    raise EnvironmentError(f'UNSAFE_TEMPORARY_POSITION: {name}: {reason}')
            return self.validator.validate(plan, self.selected_temporary_positions, state, goal=goal)
        except EnvironmentError as error:
            return False, str(error)


class EnvironmentManager(Node):
    def __init__(self, node_name='environment_manager_test'):
        super().__init__(node_name)
        share = get_package_share_directory('ur3_vision_planning')
        self.declare_parameter('temporary_config', os.path.join(share, 'config', 'temporary_positions.yaml'))
        self.declare_parameter('scene_file', os.path.join(share, 'config', 'scene.yaml'))
        self.declare_parameter('camera_file', os.path.join(share, 'config', 'camera.yaml'))
        self.declare_parameter('object_name', 'red_cube')
        self.declare_parameter('target_zone', 'zone_b')
        def load(parameter):
            with open(self.get_parameter(parameter).value, encoding='utf-8') as f:
                return yaml.safe_load(f)
        self.config = load('temporary_config')
        self.planner = EnvironmentPlanner(load('scene_file'), self.config, load('camera_file')['camera'])
        self.latest_state = None
        self.latest_error = 'ENVIRONMENT_NOT_RECEIVED'
        self.last_received = None
        self.last_progress = None
        self.previous_stamp = None
        self.subscription = self.create_subscription(
            String, self.config['environment']['topic'], self.on_environment, 10)

    def on_environment(self, message):
        self.last_received = time.monotonic()
        self.latest_state = None
        try:
            state = parse_json(message.data)
            self.latest_state = state
            self.planner.check_environment(state)
            if self.previous_stamp is None or state['stamp'] != self.previous_stamp:
                self.last_progress = self.last_received
                self.previous_stamp = state['stamp']
            self.latest_error = None
        except (EnvironmentError, TypeError, KeyError) as error:
            self.latest_error = str(error)

    def snapshot(self):
        if self.latest_error is not None:
            raise EnvironmentError(self.latest_error)
        now = time.monotonic()
        environment = self.config['environment']
        if self.last_received is None or now-self.last_received > environment['receive_timeout_s']:
            raise EnvironmentError('ENVIRONMENT_STALE: no recent environment_state message')
        if self.last_progress is None or now-self.last_progress > environment['measurement_timeout_s']:
            raise EnvironmentError('ENVIRONMENT_STALE: camera measurement stamp stopped advancing')
        state = copy.deepcopy(self.latest_state)
        elapsed = now-self.last_received
        for obj in state['objects'].values():
            obj['age_s'] += elapsed
        measurement_age = self.get_clock().now().nanoseconds*1e-9-state['stamp']
        if not 0 <= measurement_age <= environment['measurement_timeout_s']:
            raise EnvironmentError('ENVIRONMENT_STALE: measurement age invalid; use_sim_time must match camera')
        return self.planner.check_environment(state)

    def find_free_temporary_position(self, environment_state):
        return self.planner.find_free_temporary_position(environment_state)

    def resolve_goal(self, object_name, target_zone):
        return self.planner.resolve_goal(object_name, target_zone, self.snapshot())


def main(args=None):
    rclpy.init(args=args)
    node = EnvironmentManager()
    try:
        deadline = time.monotonic()+node.config['environment']['wait_timeout_s']
        while rclpy.ok() and node.last_received is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        # DDS can deliver environment_state before this new node's first /clock.
        # Wait for the clock instead of weakening freshness checks or using wall time.
        while (rclpy.ok() and node.latest_error is None
               and node.get_clock().now().nanoseconds*1e-9 < node.latest_state['stamp']
               and time.monotonic() < deadline):
            rclpy.spin_once(node, timeout_sec=0.1)
        obj = node.get_parameter('object_name').value
        zone = node.get_parameter('target_zone').value
        plan = node.resolve_goal(obj, zone)
        state = node.latest_state
        print(f'CAMERA OCCUPANT: {zone} = {state["zones"][zone]["object"]}', flush=True)
        if len(plan['plan']) == 1:
            print('TASK ALREADY SATISFIED', flush=True)
        print('DRY-RUN PLAN VALID — no robot motion', flush=True)
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    except EnvironmentError as error:
        print(f'DRY-RUN REFUSED: {error}', flush=True)
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0

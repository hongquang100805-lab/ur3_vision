"""Explicitly authorized execution with camera confirmation between steps."""

import copy
import json
import math
import time

import rclpy

from .environment_manager import EnvironmentError, finite_number


class MotionExecutionError(EnvironmentError):
    pass


def execution_enabled(dry_run, execute_robot):
    if type(dry_run) is not bool or type(execute_robot) is not bool:
        raise EnvironmentError('INVALID_EXECUTION_FLAGS: flags must be bool')
    if dry_run:
        return False
    if not execute_robot:
        raise EnvironmentError('EXECUTION_NOT_CONFIRMED: require dry_run=false AND execute_robot=true')
    return True


def validate_execution_config(config):
    if not isinstance(config, dict):
        raise EnvironmentError('INVALID_EXECUTION_CONFIG')
    continuity = config.get('continuity')
    if not isinstance(continuity, dict):
        raise EnvironmentError('INVALID_EXECUTION_CONFIG: continuity')
    for key, maximum in (('maximum_ik_delta_rad', math.pi), ('maximum_trajectory_delta_rad', 1.),
                         ('execution_start_tolerance_rad', .05), ('joint_limit_margin_rad', .2),
                         ('repeat_branch_tolerance_rad', .02)):
        if not finite_number(continuity.get(key)) or not 0 < continuity[key] <= maximum:
            raise EnvironmentError(f'INVALID_EXECUTION_CONFIG: continuity.{key}')
    for key, maximum in (('maximum_equivalent_candidates', 64), ('maximum_home_candidates', 8)):
        if type(continuity.get(key)) is not int or not 1 <= continuity[key] <= maximum:
            raise EnvironmentError(f'INVALID_EXECUTION_CONFIG: continuity.{key}')
    for key in ('server_timeout_s', 'service_timeout_s', 'precheck_scene_drift_m',
                'place_release_clearance_m', 'place_approach_clearance_m', 'zone_offset_margin_m'):
        if not finite_number(config.get(key)) or config[key] <= 0:
            raise EnvironmentError(f'INVALID_EXECUTION_CONFIG: {key} must be positive')
    if type(config.get('ik_timeout_s')) is not int or config['ik_timeout_s'] <= 0:
        raise EnvironmentError('INVALID_EXECUTION_CONFIG: ik_timeout_s')
    for key in ('cartesian_step_m', 'collision_diagnostic_step_m'):
        if not finite_number(config.get(key)) or not .002 <= config[key] <= .005:
            raise EnvironmentError(f'INVALID_EXECUTION_CONFIG: {key} must be 0.002..0.005 m')
    home = config.get('home_precheck')
    if (not isinstance(home, dict) or not finite_number(home.get('planning_time_s'))
            or not 10 <= home['planning_time_s'] <= 20
            or type(home.get('planning_attempts')) is not int or not 1 <= home['planning_attempts'] <= 3):
        raise EnvironmentError('INVALID_EXECUTION_CONFIG: home_precheck time/attempts')
    for key, maximum in (('joint_tolerance_rad', .01), ('start_tolerance_rad', .001)):
        if not finite_number(home.get(key)) or not 0 < home[key] <= maximum:
            raise EnvironmentError(f'INVALID_EXECUTION_CONFIG: home_precheck.{key}')
    gripper = config.get('precheck_gripper')
    if not isinstance(gripper, dict):
        raise EnvironmentError('INVALID_EXECUTION_CONFIG: precheck_gripper')
    for mode in ('open', 'holding'):
        positions = gripper.get(mode, {})
        if (not isinstance(positions, dict) or set(positions) != {
                'simple_gripper_left_finger_joint', 'simple_gripper_right_finger_joint'}
                or not all(finite_number(v) and 0 <= v <= .043 for v in positions.values())):
            raise EnvironmentError(f'INVALID_EXECUTION_CONFIG: precheck_gripper.{mode}')
    point = config.get('camera_retreat', {})
    if (not isinstance(point, dict) or point.get('frame_id') != 'base_link'
            or not all(finite_number(point.get(a)) for a in ('x', 'y', 'z'))):
        raise EnvironmentError('INVALID_EXECUTION_CONFIG: camera_retreat')
    verification = config.get('verification', {})
    if (not isinstance(verification, dict) or type(verification.get('minimum_frames')) is not int
            or verification['minimum_frames'] < 3):
        raise EnvironmentError('INVALID_EXECUTION_CONFIG: at least 3 camera frames required')
    for key in ('timeout_s', 'position_tolerance_m', 'z_tolerance_m'):
        if not finite_number(verification.get(key)) or verification[key] <= 0:
            raise EnvironmentError(f'INVALID_EXECUTION_CONFIG: verification.{key}')


class CameraPlacementConfirmation:
    """Count distinct post-retreat camera measurements, not repeated JSON publishes."""

    def __init__(self, name, destination, point, vacated_zones, config, after_stamp):
        self.name, self.destination, self.point = name, destination, point
        self.vacated_zones, self.config = vacated_zones, config
        self.after_stamp = after_stamp
        self.last_stamp = None
        self.count = 0
        self.reason = 'waiting for post-retreat camera frames'

    def observe(self, state):
        obj = state['objects'][self.name]
        if (state.get('status') != 'ok' or state.get('stale') is not False
                or obj.get('detected') is not True or obj.get('stale') is not False
                or not finite_number(obj.get('stamp')) or not isinstance(obj.get('position'), dict)):
            self.count = 0
            self.reason = 'camera/object not fresh and detected'
            return False
        stamp = obj['stamp']
        if stamp <= self.after_stamp or (self.last_stamp is not None and stamp <= self.last_stamp):
            return False
        self.last_stamp = stamp
        p = obj['position']
        near = (math.hypot(p['x']-self.point['x'], p['y']-self.point['y'])
                <= self.config['position_tolerance_m']
                and abs(p['z']-self.point['z']) <= self.config['z_tolerance_m'])
        if self.destination in state['zones']:
            zone = state['zones'][self.destination]
            destination_ok = (obj['zone'] == self.destination and zone['object'] == self.name
                              and zone['objects'] == [self.name] and zone['occupied'] is True
                              and zone['stale'] is False and zone['status'] == 'occupied')
        else:
            destination_ok = obj['zone'] is None
        empty = all(state['zones'][z]['occupied'] is False
                    and state['zones'][z]['object'] is None
                    and state['zones'][z]['objects'] == []
                    and state['zones'][z]['stale'] is False
                    and state['zones'][z]['status'] == 'empty' for z in self.vacated_zones)
        if (state['status'] == 'ok' and state['stale'] is False
                and obj['detected'] is True and obj['stale'] is False
                and near and destination_ok and empty):
            self.count += 1
            self.reason = f'{self.count} distinct matching camera measurements'
        else:
            self.count = 0
            self.reason = f'near={near}, destination_matches={destination_ok}, vacated_zones_empty={empty}'
        return self.count >= max(3, self.config['minimum_frames'])


def wait_camera_confirmation(manager, skills, name, destination, point, vacated, config):
    check = CameraPlacementConfirmation(name, destination, point, vacated, config,
                                        after_stamp=skills.get_clock().now().nanoseconds*1e-9)
    started = time.monotonic()
    deadline = started + config['timeout_s']
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(manager, timeout_sec=0.1)
        if manager.last_received is None or manager.last_received < started:
            continue
        try:
            state = manager.snapshot()
            if check.observe(state):
                print(f'CAMERA VERIFICATION: {name} -> {destination} PASS '
                      f'({check.count} distinct frames; vacated zones={vacated})', flush=True)
                return state
        except (EnvironmentError, KeyError, TypeError) as error:
            check.count = 0
            check.reason = str(error)
    raise MotionExecutionError(f'CAMERA_VERIFICATION_FAILED: {name} -> {destination}: {check.reason}')


def _default_factory(manager, config):
    # No RobotSkills import/construction, MoveIt clients or gripper publisher in dry-run.
    from .vision_robot_skills import VisionRobotSkills
    return VisionRobotSkills(manager.get_parameter('scene_file').value, config)


def print_final_execution_plan(plan):
    print('\nFINAL EXECUTION PLAN:', flush=True)
    for index, step in enumerate(plan['plan'], 1):
        args = ', '.join(str(step[k]) for k in ('object', 'destination') if k in step)
        print(f'{index}. {step["skill"]}({args})', flush=True)
    print('temporary_positions=' + json.dumps(plan['temporary_positions'], ensure_ascii=False), flush=True)
    print('VALIDATION AFTER TEMPORARY SELECTION: PLAN VALID', flush=True)


def execute_validated_plan(manager, goal, state, plan, config, skills_factory=None):
    if not execution_enabled(manager.get_parameter('dry_run').value,
                             manager.get_parameter('execute_robot').value):
        return False
    precheck_only = manager.get_parameter('precheck_only').value
    if type(precheck_only) is not bool:
        raise EnvironmentError('INVALID_EXECUTION_FLAGS: precheck_only must be bool')
    validate_execution_config(config)
    valid, reason = manager.planner.validate_plan(plan, goal, state)
    if not valid:
        raise MotionExecutionError(f'PLAN_INVALID: {reason}')
    skills = None
    motion_started = False
    precheck_success = False
    previous_selection = copy.deepcopy(manager.planner.selected_temporary_positions)
    try:
        candidates = manager.planner.free_temporary_candidates(state)
        if plan['temporary_positions']:
            print('\nTEMPORARY CANDIDATE SEARCH (camera geometry):', flush=True)
            for name, point in manager.planner.config['temporary_positions'].items():
                reason = manager.planner.last_rejections.get(name)
                print(f'{name} {point} ... {"REJECTED: " + reason if reason else "GEOMETRY PASS"}', flush=True)
        # Never mutate the caller's validated plan on a failed transaction.
        plan = copy.deepcopy(plan)
        skills = (skills_factory or _default_factory)(manager, config)
        selection_printed = False
        def selected(working):
            nonlocal selection_printed
            manager.planner.adopt_prechecked_plan(working, goal, state, candidates)
            print_final_execution_plan(working)
            selection_printed = True
        # This method waits for servers/TF and virtually simulates attachments
        # in MoveIt, restoring the scene before returning. No physical motion.
        precheck_error = None
        latest_after_precheck = None
        try:
            resolved = skills.feasibility_precheck(plan, state, temporary_candidates=candidates,
                                                 on_temporary_selected=selected)
        except BaseException as error:
            precheck_error = error
            raise
        finally:
            if getattr(skills, 'precheck_scene_restored', False):
                try:
                    latest_after_precheck = manager.fresh_snapshot()
                    skills.sync_camera_scene(latest_after_precheck)
                    if hasattr(skills, 'precheck_report'):
                        skills.precheck_report['scene_resynced'] = True
                    print('PLANNING SCENE RESYNCED FROM CAMERA AFTER PRECHECK', flush=True)
                except Exception as sync_error:
                    print(f'POST_PRECHECK_CAMERA_SYNC_FAILED: {sync_error}', flush=True)
                    if precheck_error is None:
                        raise MotionExecutionError(f'POST_PRECHECK_CAMERA_SYNC_FAILED: {sync_error}') from sync_error
        if not selection_printed:
            selected(plan)
        precheck_success = True
        if precheck_only:
            print('PRECHECK ONLY — ROBOT MOTION NOT STARTED', flush=True)
            return False
        latest = latest_after_precheck or manager.fresh_snapshot()
        valid, reason = manager.planner.validate_plan(plan, goal, latest)
        if not valid:
            raise MotionExecutionError(f'PLAN_CHANGED_BEFORE_MOTION: {reason}')
        if any(math.dist(tuple(obj['position'][a] for a in ('x', 'y', 'z')),
                         tuple(latest['objects'][name]['position'][a] for a in ('x', 'y', 'z')))
               > config['precheck_scene_drift_m'] for name, obj in state['objects'].items()):
            raise MotionExecutionError('PRECHECK_SCENE_CHANGED: camera positions changed; re-plan required')
        skills.sync_camera_scene(latest)
        print('PLANNING SCENE SYNCED FROM FRESH CAMERA BEFORE EXECUTION', flush=True)
        print('\nEXECUTION:', flush=True)
        picked_from = {}
        for index, step in enumerate(plan['plan'], 1):
            skill, name = step['skill'], step.get('object')
            destination = step.get('destination')
            label = f"{skill}({', '.join([v for v in (name, destination) if v])})"
            result = 'EXECUTION_FAILED'
            try:
                if skill == 'pick':
                    if skills.attached_object is not None:
                        raise MotionExecutionError(f'OBJECT_STILL_ATTACHED: {skills.attached_object}')
                    observed = manager.fresh_snapshot()  # refreshed immediately before every pick
                    picked_from[name] = observed['objects'][name]['zone']
                    following = plan['plan'][index]
                    dest = following['destination']
                    if dest in observed['zones']:
                        occupant = observed['zones'][dest]['object']
                        if occupant not in (None, name):
                            raise MotionExecutionError(f'DESTINATION_OCCUPIED: {dest}: {occupant}')
                    else:
                        rejection = manager.planner.candidate_rejection(resolved[dest], observed)
                        if rejection:
                            raise MotionExecutionError(f'TEMPORARY_POSITION_NO_LONGER_FREE: {rejection}')
                    skills.sync_camera_scene(observed)
                    motion_started = True
                    result = skills.pick_from_camera(name, observed)
                elif skill == 'place':
                    result = skills.place_destination(name, destination, resolved[destination])
                    if result == 'SUCCESS':
                        result = skills.camera_retreat()
                    if result == 'SUCCESS':
                        vacated = [picked_from[name]] if picked_from.get(name) not in (None, destination) else []
                        wait_camera_confirmation(manager, skills, name, destination, resolved[destination],
                                                 vacated, config['verification'])
                else:
                    if skills.attached_object is not None:
                        raise MotionExecutionError(f'OBJECT_STILL_ATTACHED: {skills.attached_object}')
                    motion_started = True
                    result = skills.home()
            except Exception as error:
                print(f'[{index}/{len(plan["plan"])}] {label} ........ FAILED: {error}', flush=True)
                raise MotionExecutionError(f'STEP_FAILED: {index} {label}: {error}') from error
            print(f'[{index}/{len(plan["plan"])}] {label} ........ {result}', flush=True)
            if result != 'SUCCESS':
                raise MotionExecutionError(f'STEP_FAILED: {index} {label}: {result}')
        print('\nTASK SUCCESS', flush=True)
        return True
    except BaseException as error:
        if not precheck_success:
            manager.planner.selected_temporary_positions = previous_selection
        stop_method = 'stop_pending_motion' if motion_started else 'cancel_pending_precheck'
        if skills is not None and hasattr(skills, stop_method):
            try:
                getattr(skills, stop_method)()
            except BaseException as stop_error:
                print(f'STOP_NOT_CONFIRMED: {stop_error}; operator recovery required', flush=True)
        held = None if skills is None else skills.attached_object
        print(f'TASK FAILED: {error}\nExecution stopped; held_object={held}. '
              'No next step or automatic release/home. Recover manually before retry.', flush=True)
        if isinstance(error, MotionExecutionError):
            raise
        raise MotionExecutionError(str(error)) from error
    finally:
        if skills is not None:
            if hasattr(skills, 'precheck_report'):
                manager.motion_precheck_report = copy.deepcopy(skills.precheck_report)
            skills.destroy_node()  # Does not open, detach or move a held object.

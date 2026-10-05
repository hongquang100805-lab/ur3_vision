"""Camera-driven adapter to the existing Gazebo/MoveIt physical skills."""

import copy
import math
import time

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (
    AllowedCollisionEntry, AttachedCollisionObject, CollisionObject, Constraints,
    ContactInformation, JointConstraint, MoveItErrorCodes, PlanningSceneComponents, RobotState,
)
from moveit_msgs.srv import GetCartesianPath, GetMotionPlan, GetPlanningScene, GetPositionFK, GetPositionIK, GetStateValidity
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters, GetParameterTypes, ListParameters
from std_msgs.msg import Float64MultiArray, String

from .environment_manager import EnvironmentError, finite_number
from .home_precheck import explicit_state, home_joint_model, joint_values
from .continuous_motion import ContinuousMotionMixin
from .joint_continuity import branch_candidates, checked_arm
from .robot_skills import RobotSkills


def extend_collision_matrix(matrix, extra_names):
    matrix = copy.deepcopy(matrix)
    names = list(matrix.entry_names)
    extra = [name for name in extra_names if name not in names]
    rows = [list(entry.enabled) + [False]*len(extra) for entry in matrix.entry_values]
    rows.extend([[False]*(len(names)+len(extra)) for _ in extra])
    matrix.entry_names = names + extra
    matrix.entry_values = [AllowedCollisionEntry(enabled=row) for row in rows]
    return matrix


def destination_center(scene, plan, destination):
    """Zone marker Z is not a cube centre. Temporary Z already is."""
    if destination in scene['zones']:
        marker = scene['zones'][destination]['position']
        z = scene['table']['center'][2] + scene['table']['size'][2]/2
        z += scene['cube_size'][2]/2 - scene.get('robot_base_height', 0.)
        return {'frame_id': 'base_link', 'x': marker[0], 'y': marker[1], 'z': z}
    point = plan['temporary_positions'].get(destination)
    if (not isinstance(point, dict) or set(point) != {'frame_id', 'x', 'y', 'z'}
            or point['frame_id'] != 'base_link'
            or not all(finite_number(point[a]) for a in ('x', 'y', 'z'))):
        raise EnvironmentError(f'INVALID_DESTINATION: {destination}')
    return copy.deepcopy(point)


class VisionRobotSkills(ContinuousMotionMixin, RobotSkills):
    def declare_parameter(self, name, value=None, descriptor=None, ignore_override=False):
        if name == 'use_sim_time':
            value = True
        return super().declare_parameter(name, value, descriptor, ignore_override)

    def __init__(self, scene_path, config):
        super().__init__(scene_path, server_timeout=config['server_timeout_s'])
        self.execution_config = config
        self.preserve_failed_grasp = True
        self.active_motion_handle = None
        self.get_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        for client, label in ((self.get_scene_client, 'PLANNING_SCENE'), (self.fk_client, 'FK')):
            if not client.wait_for_service(timeout_sec=config['server_timeout_s']):
                self.destroy_node()
                raise EnvironmentError(f'{label}_SERVER_UNAVAILABLE')

    def wait_result(self, future, timeout=60.0):
        result = super().wait_result(future, timeout)
        if hasattr(result, 'cancel_goal_async') and result.accepted:
            self.active_motion_handle = result
        elif hasattr(result, 'result') and hasattr(result, 'status'):
            self.active_motion_handle = None
        return result

    def wait_trajectory_result(self, future, timeout=60.0):
        result = super().wait_trajectory_result(future, timeout)
        self.active_motion_handle = None
        return result

    def move_cartesian(self, waypoints, diagnostic_label='Cartesian path'):
        result = self._continuous_cartesian_execution(waypoints, diagnostic_label)
        if result != 'SUCCESS' and self.active_motion_handle is not None:
            # A timeout after ExecuteTrajectory acceptance is NOT a mere
            # planning failure. Do not launch the pose fallback concurrently.
            return 'EXECUTION_STATE_UNCERTAIN'
        return result

    def cancel_pending_precheck(self):
        """Cancel plan-only action without sending any actuator command."""
        if self.active_motion_handle is not None:
            handle = self.active_motion_handle
            self.wait_result(handle.cancel_goal_async(), timeout=self.execution_config['service_timeout_s'])
            # Require a terminal action result before claiming cancellation settled.
            self.wait_result(handle.get_result_async(), timeout=self.execution_config['service_timeout_s'])
            self.active_motion_handle = None
            self._motion_submission_pending = False
        elif getattr(self, '_motion_submission_pending', False):
            raise EnvironmentError('EXECUTION_SUBMISSION_UNCERTAIN: no action handle; operator recovery required')

    def stop_pending_motion(self):
        """Failure-only cancellation; never open gripper, detach, retreat or home."""
        self.gripper_command_publisher.publish(Float64MultiArray(data=[0.0, 0.0]))
        VisionRobotSkills.cancel_pending_precheck(self)

    def world_xyz(self, point):
        return point['x'], point['y'], point['z'] + self.robot_base_height

    def pick_from_camera(self, name, state):
        obj = state['objects'].get(name, {})
        if state.get('status') != 'ok' or state.get('stale') is not False:
            return 'CAMERA_STALE'
        if obj.get('detected') is not True or obj.get('stale') is not False:
            return 'OBJECT_NOT_DETECTED_OR_STALE'
        point = obj.get('position')
        if (state.get('frame_id') != self.frame_id or not finite_number(obj.get('age_s'))
                or not 0 <= obj['age_s'] <= 1.0):
            return 'CAMERA_OBJECT_STALE_OR_INVALID_FRAME'
        if not isinstance(point, dict) or not all(finite_number(point.get(a)) for a in ('x', 'y', 'z')):
            return 'INVALID_CAMERA_POSITION'
        return super().pick(name, object_position=self.world_xyz(point))

    def place_destination(self, name, destination, point):
        if not self.simulator_is_ready():
            return 'SIMULATOR_NOT_READY'
        return self.place_at(name, self.world_xyz(point), destination, offsets=[0.0],
                             approach_clearance=self.execution_config['place_approach_clearance_m'],
                             release_clearance=self.execution_config['place_release_clearance_m'])

    def camera_retreat(self):
        return self.move_to_verified_place_pose(
            *self.world_xyz(self.execution_config['camera_retreat']), 'camera clearance retreat')

    def sync_camera_scene(self, state):
        for name, obj in state['objects'].items():
            if name != self.attached_object:
                if not self.update_attached_object(name, attached=False,
                                                   world_position=self.world_xyz(obj['position'])):
                    raise EnvironmentError(f'PLANNING_SCENE_SYNC_FAILED: {name}')

    def read_planning_scene(self):
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            | PlanningSceneComponents.ALLOWED_COLLISION_MATRIX)
        return self.wait_result(self.get_scene_client.call_async(request),
                                self.execution_config['service_timeout_s']).scene

    def tool_targets(self, point, place=False):
        x, y, z = self.world_xyz(point)
        if place:
            final = z + self.grasp_object_offset + self.execution_config['place_release_clearance_m']
            pre = final + self.execution_config['place_approach_clearance_m']
            return [('pre-place', (x, y, pre)), ('final-place', (x, y, final))]
        grasp = z + self.grasp_object_offset
        pre = self.collision_free_approach_z(z, self.cube_height, self.finger_reach, self.approach_clearance)
        return [('pre-grasp', (x, y, pre)), ('grasp', (x, y, grasp)), ('lift', (x, y, pre))]

    def _ik(self, xyz, seed, label):
        checked_arm(joint_values(seed), self.ARM_JOINT_NAMES, self._branch_model())
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name, ik.ik_link_name = self.group_name, self.ee_link
        # Carry the seed's joints only. Attached objects come from virtual scene,
        # not from an obsolete IK result returned before a simulated detach.
        ik.robot_state = RobotState(is_diff=True, joint_state=copy.deepcopy(seed.joint_state))
        ik.avoid_collisions = True
        ik.pose_stamped = PoseStamped()
        ik.pose_stamped.header.frame_id = self.frame_id
        ik.pose_stamped.pose.position.x = xyz[0]
        ik.pose_stamped.pose.position.y = xyz[1]
        ik.pose_stamped.pose.position.z = xyz[2] - self.robot_base_height
        ik.pose_stamped.pose.orientation = self.down_orientation
        ik.timeout.sec = self.execution_config['ik_timeout_s']
        result = self.wait_result(self.ik_client.call_async(request),
                                  self.execution_config['service_timeout_s'])
        passed = result.error_code.val == MoveItErrorCodes.SUCCESS
        self.last_precheck_reason = self.moveit_error_name(result.error_code.val)
        normalized = seed
        if passed:
            try:
                normalized = self._normalize_branch(result.solution, seed, xyz, label)
            except EnvironmentError as error:
                passed = False
                self.last_precheck_reason = str(error)
                print(f'{label} branch ........ FAIL: {error}', flush=True)
        print(f'{label} ........ {"PASS" if passed else "FAIL"} '
              f'pose=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]-self.robot_base_height:.3f}) '
              f'frame={self.frame_id}, link={self.ee_link}, '
              f'error_code={result.error_code.val} ({self.moveit_error_name(result.error_code.val)})', flush=True)
        return passed, normalized if passed else seed

    def _fk_matches(self, seed, xyz, label):
        request = GetPositionFK.Request()
        request.header.frame_id = self.frame_id
        request.fk_link_names = [self.ee_link]
        request.robot_state = RobotState(is_diff=True, joint_state=copy.deepcopy(seed.joint_state))
        result = self.wait_result(self.fk_client.call_async(request), self.execution_config['service_timeout_s'])
        if (result.error_code.val != MoveItErrorCodes.SUCCESS or len(result.pose_stamped) != 1
                or list(result.fk_link_names) != [self.ee_link]
                or result.pose_stamped[0].header.frame_id != self.frame_id):
            self.last_precheck_reason = 'FK_FAILED_OR_WRONG_FRAME'
            return False
        pose = result.pose_stamped[0].pose
        actual = (pose.position.x, pose.position.y, pose.position.z)
        desired = (xyz[0], xyz[1], xyz[2] - self.robot_base_height)
        q = tuple(getattr(pose.orientation, a) for a in ('x', 'y', 'z', 'w'))
        expected = tuple(getattr(self.down_orientation, a) for a in ('x', 'y', 'z', 'w'))
        if not all(finite_number(v) for v in (*actual, *q, *expected)):
            self.last_precheck_reason = 'INVALID_FK_POSE'
            return False
        norm, expected_norm = math.sqrt(sum(v*v for v in q)), math.sqrt(sum(v*v for v in expected))
        if not .99 <= norm <= 1.01 or not .99 <= expected_norm <= 1.01:
            self.last_precheck_reason = 'INVALID_FK_OR_EXPECTED_ORIENTATION'
            return False
        dot = abs(sum(a*b for a, b in zip(q, expected))) / (norm * expected_norm)
        angle = 2*math.acos(min(1., dot))
        distance = math.dist(actual, desired)
        passed = distance <= .008 and angle <= .08
        print(f'{label} FK {"PASS" if passed else "FAIL"}: pose={actual}, '
              f'position_error={distance:.6f}, orientation_error={angle:.6f}', flush=True)
        self.last_precheck_reason = 'FK_POSE_MISMATCH' if not passed else 'SUCCESS'
        return passed

    def _cartesian_precheck(self, start, target, seed, label):
        """Plan only, from the preceding IK branch, with limited contact ACM.

        Endpoint IK at grasp is deliberately NOT used: the actual pick lowers
        from pre-grasp with one continuous Cartesian path. No trajectory action.
        """
        if not self._fk_matches(seed, start, f'{label} start'):
            print(f'{label} ........ FAIL: {self.last_precheck_reason}', flush=True)
            return False, seed
        request = GetCartesianPath.Request()
        request.header.frame_id = self.frame_id
        request.group_name, request.link_name = self.group_name, self.ee_link
        request.start_state = RobotState(is_diff=True, joint_state=copy.deepcopy(seed.joint_state))
        request.avoid_collisions = True
        request.max_step, request.jump_threshold = self.execution_config['cartesian_step_m'], 0.
        request.max_velocity_scaling_factor = self.motion_scaling_factor
        request.max_acceleration_scaling_factor = self.motion_scaling_factor
        pose = Pose()
        pose.position.x, pose.position.y = target[:2]
        pose.position.z = target[2] - self.robot_base_height
        pose.orientation = copy.deepcopy(self.down_orientation)
        request.waypoints = [pose]
        result = self.wait_result(self.cartesian_client.call_async(request), self.execution_config['service_timeout_s'])
        trajectory = result.solution.joint_trajectory
        # Stricter than >=0.95: execution also requires the ENTIRE path.
        passed = (result.error_code.val == MoveItErrorCodes.SUCCESS
                  and finite_number(result.fraction) and result.fraction >= 1.0
                  and bool(trajectory.points) and bool(trajectory.joint_names))
        if passed:
            passed = self._fk_matches(result.start_state, start, f'{label} returned start')
        next_seed = copy.deepcopy(seed)
        if passed:
            try:
                next_seed = self._guard_trajectory(result.solution, seed, xyz=target, label=label)
            except EnvironmentError as error:
                self.last_precheck_reason = str(error)
                passed = False
        if not passed:
            self.last_precheck_reason = (f'CARTESIAN_INCOMPLETE_OR_INVALID fraction={result.fraction} '
                                        f'error_code={result.error_code.val}; {self.last_precheck_reason}')
        # Diagnose the SAME start branch. Geometric IK is only an instrument
        # for obtaining invalid states; every such state must then go through
        # collision checking. It is NEVER used to execute or to override fraction.
        if label.endswith('Cartesian lowering'):
            path_reason = self.last_precheck_reason
            diagnostics = self._diagnose_cartesian_waypoints(start, target, seed, label)
            if not passed:
                self.last_precheck_reason = f'{path_reason}; {diagnostics[1]}'
            elif not diagnostics[0]:
                passed = False
                self.last_precheck_reason = diagnostics[1]
        print(f'{label} fraction={result.fraction:.3f} ........ {"PASS" if passed else "FAIL"} '
              f'error_code={result.error_code.val}, frame={self.frame_id}, link={self.ee_link}, '
              f'start={start}, target={target}', flush=True)
        if hasattr(self, 'precheck_report'):
            self.precheck_report.setdefault('cartesian', []).append({'step': label, 'fraction': result.fraction, 'pass': passed})
        return passed, next_seed if passed else seed

    def _diagnose_cartesian_waypoints(self, start, target, seed, label):
        print(f'\nWAYPOINT COLLISION DIAGNOSTICS: {label} (current scoped ACM)', flush=True)
        count = max(1, math.ceil(math.dist(start, target) / self.execution_config['collision_diagnostic_step_m']))
        diagnostic_seed = copy.deepcopy(seed)
        all_valid, failures = True, []
        for index in range(count + 1):
            xyz = tuple(a + (b-a)*index/count for a, b in zip(start, target))
            prefix = f'waypoint index={index}/{count} z={xyz[2]-self.robot_base_height:.6f}'
            if index:
                request = GetPositionIK.Request()
                ik = request.ik_request
                ik.group_name, ik.ik_link_name = self.group_name, self.ee_link
                ik.robot_state = RobotState(is_diff=True, joint_state=copy.deepcopy(diagnostic_seed.joint_state))
                ik.avoid_collisions = False  # GEOMETRIC DIAGNOSTIC ONLY; not an acceptance check.
                ik.pose_stamped.header.frame_id = self.frame_id
                ik.pose_stamped.pose.position.x, ik.pose_stamped.pose.position.y = xyz[:2]
                ik.pose_stamped.pose.position.z = xyz[2] - self.robot_base_height
                ik.pose_stamped.pose.orientation = copy.deepcopy(self.down_orientation)
                ik.timeout.sec = self.execution_config['ik_timeout_s']
                result = self.wait_result(self.ik_client.call_async(request), self.execution_config['service_timeout_s'])
                if result.error_code.val != MoveItErrorCodes.SUCCESS:
                    reason = f'{prefix}: GEOMETRIC_IK_FAILED {self.moveit_error_name(result.error_code.val)}'
                    print(f'{reason}, state_valid=UNKNOWN, contacts=UNAVAILABLE (no solved state)', flush=True)
                    failures.append(reason)
                    all_valid = False
                    continue
                old = dict(zip(diagnostic_seed.joint_state.name, diagnostic_seed.joint_state.position))
                new = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
                jump = max((abs(new.get(n, old.get(n, 0.))-old.get(n, 0.))
                            for n in self.ARM_JOINT_NAMES), default=0.)
                try:
                    diagnostic_seed = self._normalize_branch(result.solution, diagnostic_seed, xyz,
                                                              prefix, diagnostic=True)
                except EnvironmentError as error:
                    failures.append(str(error))
                    all_valid = False
                    continue
                print(f'{prefix} diagnostic raw IK joint_step_max={jump:.6f} rad (normalized above; may differ from Cartesian solver)', flush=True)
            fk_ok = self._fk_matches(diagnostic_seed, xyz, f'{prefix} diagnostic')
            request = GetStateValidity.Request()
            request.group_name = ''  # Full robot, including both fingers; never disable collisions.
            request.robot_state = RobotState(is_diff=True, joint_state=copy.deepcopy(diagnostic_seed.joint_state))
            result = self.wait_result(self.state_validity_client.call_async(request), self.execution_config['service_timeout_s'])
            print(f'{prefix} state={"VALID" if result.valid else "INVALID"}, FK={"PASS" if fk_ok else "FAIL"}, '
                  f'contacts={len(result.contacts)}', flush=True)
            for contact in result.contacts:
                bodies = [(contact.contact_body_1, contact.body_type_1),
                          (contact.contact_body_2, contact.body_type_2)]
                links = [name for name, kind in bodies if kind == ContactInformation.ROBOT_LINK]
                objects = [name for name, kind in bodies if kind in
                           (ContactInformation.WORLD_OBJECT, ContactInformation.ROBOT_ATTACHED)]
                pair = f'{contact.contact_body_1} <-> {contact.contact_body_2}'
                print(f'  collision contact pair={pair}; robot_links={links}; collision_objects={objects}; '
                      f'depth={contact.depth:.6f}', flush=True)
                if not result.valid:
                    failures.append(pair)
            if not result.valid or not fk_ok:
                all_valid = False
                if not fk_ok:
                    failures.append(f'{prefix}: FK_POSE_MISMATCH')
                if not result.contacts:
                    failures.append(f'{prefix}: INVALID_WITHOUT_CONTACTS_OR_FK_MISMATCH (not proof of collision)')
        reason = 'WAYPOINTS_VALID' if all_valid else 'WAYPOINT_CHECK_FAILED: ' + '; '.join(dict.fromkeys(failures))
        print(f'WAYPOINT COLLISION DIAGNOSTICS: {"PASS" if all_valid else "FAIL"}: {reason}', flush=True)
        return all_valid, reason

    def _allow_grasp_touch(self, scene, name):
        links = ['simple_gripper_base_link', 'simple_gripper_left_finger_link',
                 'simple_gripper_right_finger_link']
        matrix = extend_collision_matrix(scene.allowed_collision_matrix,
                                         [*self.scene['objects'], *links])
        names = list(matrix.entry_names)
        for cube in self.scene['objects']:
            for other in names:
                i, j = names.index(cube), names.index(other)
                allowed = cube == name and other in links
                matrix.entry_values[i].enabled[j] = matrix.entry_values[j].enabled[i] = allowed
            if cube in matrix.default_entry_names:
                matrix.default_entry_values[matrix.default_entry_names.index(cube)] = False
        diff = copy.deepcopy(scene)
        # Only intended finger/target contact, never disable table/robot collisions.
        diff.world.collision_objects = []
        diff.robot_state = RobotState(is_diff=True)
        diff.is_diff = True
        diff.allowed_collision_matrix = matrix
        self._apply_scene(diff)

    def _apply_scene(self, scene):
        from moveit_msgs.srv import ApplyPlanningScene
        result = self.wait_result(self.scene_client.call_async(ApplyPlanningScene.Request(scene=scene)),
                                  self.execution_config['service_timeout_s'])
        if not result.success:
            raise EnvironmentError('PLANNING_SCENE_DIFF_REJECTED')

    def _reset_grasp_touch(self, baseline):
        diff = copy.deepcopy(baseline)
        diff.is_diff = True
        diff.world.collision_objects = []
        diff.robot_state = RobotState(is_diff=True)
        matrix = extend_collision_matrix(baseline.allowed_collision_matrix,
            [*self.scene['objects'], 'simple_gripper_base_link',
             'simple_gripper_left_finger_link', 'simple_gripper_right_finger_link'])
        for cube in self.scene['objects']:
            for other in matrix.entry_names:
                i, j = matrix.entry_names.index(cube), matrix.entry_names.index(other)
                matrix.entry_values[i].enabled[j] = matrix.entry_values[j].enabled[i] = False
            if cube in matrix.default_entry_names:
                matrix.default_entry_values[matrix.default_entry_names.index(cube)] = False
        diff.allowed_collision_matrix = matrix
        self._apply_scene(diff)
        print('GRASP CONTACT ACM RESET: attached-object touch_links only', flush=True)

    def _seed_gripper(self, seed, mode):
        seed = copy.deepcopy(seed)
        joints = dict(zip(seed.joint_state.name, seed.joint_state.position))
        joints.update(self.execution_config['precheck_gripper'][mode])
        seed.joint_state.name, seed.joint_state.position = list(joints), list(joints.values())
        seed.joint_state.velocity, seed.joint_state.effort = [], []
        return seed

    def _confirm_virtual_attachment(self, name):
        scene = self.read_planning_scene()
        attached = [item for item in scene.robot_state.attached_collision_objects if item.object.id == name]
        links = {'simple_gripper_base_link', 'simple_gripper_left_finger_link',
                 'simple_gripper_right_finger_link'}
        if (len(attached) != 1 or attached[0].link_name != self.ee_link
                or not links.issubset(attached[0].touch_links)
                or name in {item.id for item in scene.world.collision_objects}):
            raise EnvironmentError(f'PRECHECK_VIRTUAL_ATTACH_NOT_CONFIRMED: {name}')
        print(f'PRECHECK MoveIt attachment confirmed: {name}, link={self.ee_link}', flush=True)

    def _verify_scene_restored(self, baseline):
        actual = self.read_planning_scene()
        if actual.robot_state.attached_collision_objects:
            raise EnvironmentError('RESTORE_HAS_ATTACHED_OBJECTS')
        objects = {item.id: item for item in actual.world.collision_objects}
        # Ignore service/header timestamps, but not geometry or camera poses.
        for expected in baseline.world.collision_objects:
            item = objects.get(expected.id)
            fields = ('pose', 'primitives', 'primitive_poses', 'meshes', 'mesh_poses', 'planes', 'plane_poses')
            if item is None or any(getattr(item, field) != getattr(expected, field) for field in fields):
                raise EnvironmentError(f'RESTORE_WORLD_MISMATCH: {expected.id}')
        wanted = extend_collision_matrix(baseline.allowed_collision_matrix,
                    [*self.scene['objects'], 'simple_gripper_base_link',
                     'simple_gripper_left_finger_link', 'simple_gripper_right_finger_link'])
        matrix = actual.allowed_collision_matrix
        indices = {name: i for i, name in enumerate(matrix.entry_names)}
        for i, first in enumerate(wanted.entry_names):
            for j, second in enumerate(wanted.entry_names):
                # An omitted all-false entry is equivalent to default false.
                enabled = (matrix.entry_values[indices[first]].enabled[indices[second]]
                           if first in indices and second in indices else False)
                if enabled != wanted.entry_values[i].enabled[j]:
                    raise EnvironmentError(f'RESTORE_ACM_MISMATCH: {first}/{second}')
        if (list(matrix.default_entry_names) != list(baseline.allowed_collision_matrix.default_entry_names)
                or list(matrix.default_entry_values) != list(baseline.allowed_collision_matrix.default_entry_values)):
            raise EnvironmentError('RESTORE_ACM_DEFAULTS_MISMATCH')

    def _home_description_topic(self, name):
        """Follow MoveIt's RDFLoader: parameters may be backed by latched topics."""
        received = []
        topic = '/' + name
        subscription = self.create_subscription(
            String, topic, lambda msg: received.append(msg.data) if msg.data.strip() else None,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))
        try:
            deadline = time.monotonic() + self.execution_config['service_timeout_s']
            while rclpy.ok() and not received and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
            if not received:
                raise EnvironmentError(f'HOME_MODEL_UNAVAILABLE: {name}: no nonempty string parameter '
                                       f'or transient-local topic {topic}; description timeout')
            print(f'HOME {name} source: topic {topic} (transient-local; {len(received[0])} chars)', flush=True)
            return received[0]
        finally:
            self.destroy_subscription(subscription)

    def _home_model(self):
        """Read live URDF/SRDF from parameters or topics, never guessed files."""
        if not hasattr(self, 'home_parameter_client'):
            self.home_parameter_client = self.create_client(GetParameters, '/move_group/get_parameters')
        if not self.home_parameter_client.wait_for_service(timeout_sec=self.execution_config['server_timeout_s']):
            raise EnvironmentError('HOME_MODEL_SERVER_UNAVAILABLE: /move_group/get_parameters')
        if not hasattr(self, 'home_parameter_list_client'):
            self.home_parameter_list_client = self.create_client(ListParameters, '/move_group/list_parameters')
        if not self.home_parameter_list_client.wait_for_service(timeout_sec=self.execution_config['server_timeout_s']):
            raise EnvironmentError('HOME_MODEL_SERVER_UNAVAILABLE: /move_group/list_parameters')
        if not hasattr(self, 'home_parameter_type_client'):
            self.home_parameter_type_client = self.create_client(GetParameterTypes, '/move_group/get_parameter_types')
        if not self.home_parameter_type_client.wait_for_service(timeout_sec=self.execution_config['server_timeout_s']):
            raise EnvironmentError('HOME_MODEL_SERVER_UNAVAILABLE: /move_group/get_parameter_types')
        # Listing proves declaration, NOT initialization. Jazzy rclcpp returns
        # an empty GetParameters batch if any statically typed value is unset.
        # GetParameterTypes reports the VALUE type (NOT_SET for these values),
        # unlike DescribeParameters which can report the declared static type.
        listed = self.wait_result(self.home_parameter_list_client.call_async(ListParameters.Request(
            prefixes=['robot_description', 'robot_description_semantic', 'robot_description_planning.joint_limits'],
            depth=0)), self.execution_config['service_timeout_s'])
        declared = set(listed.result.names)
        descriptions = ['robot_description', 'robot_description_semantic']
        wanted = list(descriptions)
        for name in self.ARM_JOINT_NAMES:
            wanted.extend(f'robot_description_planning.joint_limits.{name}.{field}'
                          for field in ('has_position_limits', 'min_position', 'max_position'))
        keys = [key for key in wanted if key in declared]
        parameters = {}
        value_types = {}
        if keys:
            typed = self.wait_result(self.home_parameter_type_client.call_async(GetParameterTypes.Request(names=keys)),
                                     self.execution_config['service_timeout_s'])
            if len(typed.types) != len(keys):
                raise EnvironmentError(f'HOME_MODEL_PARAMETER_TYPES_INVALID: expected={len(keys)}, '
                                       f'received={len(typed.types)}')
            value_types = dict(zip(keys, typed.types))
        for key in keys:
            if value_types[key] == ParameterType.PARAMETER_NOT_SET:
                print(f'HOME parameter {key}: UNINITIALIZED (declared, value type=NOT_SET)', flush=True)
                continue
            result = self.wait_result(self.home_parameter_client.call_async(GetParameters.Request(names=[key])),
                                      self.execution_config['service_timeout_s'])
            if not result.values:
                # Permit only a proven transition to NOT_SET, never silently
                # drop an initialized bounds override on a malformed response.
                current = self.wait_result(self.home_parameter_type_client.call_async(
                    GetParameterTypes.Request(names=[key])), self.execution_config['service_timeout_s'])
                if list(current.types) == [ParameterType.PARAMETER_NOT_SET]:
                    value_types[key] = ParameterType.PARAMETER_NOT_SET
                    print(f'HOME parameter {key}: UNINITIALIZED during read (confirmed NOT_SET)', flush=True)
                    continue
            if len(result.values) != 1:
                raise EnvironmentError(f'HOME_MODEL_PARAMETER_RESPONSE_INVALID: parameter={key}; '
                                       f'value_type={value_types[key]}; expected=1, received={len(result.values)}')
            if result.values[0].type != value_types[key]:
                raise EnvironmentError(f'HOME_MODEL_PARAMETER_CHANGED: {key}: '
                                       f'type_before={value_types[key]}, type_after={result.values[0].type}; retry precheck')
            parameters[key] = result.values[0]
        xml = {}
        for name in descriptions:
            value = parameters.get(name)
            if value is not None and value.type not in (ParameterType.PARAMETER_NOT_SET, ParameterType.PARAMETER_STRING):
                raise EnvironmentError(f'HOME_MODEL_INVALID: {name} parameter has type={value.type}, expected STRING')
            if value is not None and value.type == ParameterType.PARAMETER_STRING and value.string_value.strip():
                xml[name] = value.string_value
                print(f'HOME {name} source: /move_group parameter ({len(value.string_value)} chars)', flush=True)
            else:
                reason = ('NOT_DECLARED' if name not in declared else 'UNINITIALIZED' if value is None
                          else 'EMPTY_STRING')
                print(f'HOME {name} parameter: {reason}; reading live topic /{name}', flush=True)
                xml[name] = self._home_description_topic(name)
        overrides = {}
        for name in self.ARM_JOINT_NAMES:
            fields = {}
            for field in ('has_position_limits', 'min_position', 'max_position'):
                value = parameters.get(f'robot_description_planning.joint_limits.{name}.{field}')
                if value is None or value.type == ParameterType.PARAMETER_NOT_SET:
                    continue
                if field == 'has_position_limits' and value.type == ParameterType.PARAMETER_BOOL:
                    fields[field] = value.bool_value
                elif field != 'has_position_limits' and value.type in (ParameterType.PARAMETER_DOUBLE, ParameterType.PARAMETER_INTEGER):
                    fields[field] = value.double_value if value.type == ParameterType.PARAMETER_DOUBLE else value.integer_value
                else:
                    raise EnvironmentError(f'HOME_MODEL_INVALID: parameter {name}.{field} has wrong type')
            overrides[name] = fields
        print(f'HOME limits/group source: live URDF + SRDF; MoveIt position-limit parameters read: '
              f'{sum(key not in descriptions for key in parameters)}/{len(wanted)-len(descriptions)} '
              '(undeclared/uninitialized overrides use URDF bounds)', flush=True)
        return home_joint_model(xml['robot_description'], xml['robot_description_semantic'],
                                self.group_name, self.ARM_JOINT_NAMES, overrides)

    def _home_state_validity(self, state, label):
        request = GetStateValidity.Request(robot_state=copy.deepcopy(state), group_name='')
        result = self.wait_result(self.state_validity_client.call_async(request), self.execution_config['service_timeout_s'])
        print(f'{label} state validity ..... {"PASS" if result.valid else "FAIL"}', flush=True)
        for contact in result.contacts:
            print(f'  HOME collision contact: {contact.contact_body_1} <-> {contact.contact_body_2}; '
                  f'body_types=({contact.body_type_1},{contact.body_type_2}); depth={contact.depth:.6f}', flush=True)
        if not result.valid and not result.contacts:
            print('  INVALID without contacts: bounds/constraints must also be checked', flush=True)
        return result.valid

    def _home_precheck(self, seed):
        try:
            return self._select_home_branch(seed)
        except EnvironmentError as error:
            self.last_precheck_reason = str(error)
            print(f'HOME FEASIBILITY: FAIL: {error}', flush=True)
            return False

    def _select_home_branch(self, seed):
        self._last_home_trajectory = None
        self._last_home_goal = None
        model = self._branch_model()
        start = checked_arm(joint_values(seed), self.ARM_JOINT_NAMES, model)
        if hasattr(self, 'precheck_report'):
            self.precheck_report['retreat_joints'] = start
        canonical = dict(zip(self.ARM_JOINT_NAMES, self.HOME_JOINT_VALUES))
        if self._home_plan_candidate(seed, canonical, 'canonical home'):
            return True
        # Different, FK-equivalent goals are deterministic alternatives, not
        # random retries of the same failed request. Never mask invalid paths.
        if not self.last_precheck_reason.startswith(('HOME_PLANNING_FAILURE', 'HOME_PLANNING_TIMEOUT')):
            return False
        config = self.execution_config['continuity']
        reference = explicit_state(seed, canonical, self.ARM_JOINT_NAMES)
        candidates = branch_candidates(canonical, start, self.ARM_JOINT_NAMES, model,
                margin=config['joint_limit_margin_rad'], maximum=config['maximum_home_candidates'])
        for goal in sorted(candidates, key=lambda v: sum(abs(v[n]-start[n]) for n in self.ARM_JOINT_NAMES)):
            if all(abs(goal[n]-canonical[n]) < 1e-8 for n in self.ARM_JOINT_NAMES):
                continue
            label = 'equivalent home branch'
            candidate = explicit_state(seed, goal, self.ARM_JOINT_NAMES)
            print(f'{label}: goal={goal}; total_joint_motion={sum(abs(goal[n]-start[n]) for n in self.ARM_JOINT_NAMES):.9f}', flush=True)
            if not self._fk_equivalent(candidate, reference, label) or not self._home_state_validity(candidate, label):
                continue
            if self._home_plan_candidate(seed, goal, label):
                return True
        return False

    def _home_plan_candidate(self, seed, goal_arm, branch_label):
        print('\nHOME FEASIBILITY:', flush=True)
        print(f'simulated start: {getattr(self, "home_start_source", "explicit supplied seed")}', flush=True)
        print('MoveIt interface: GetMotionPlan /plan_kinematic_path (planning ONLY)', flush=True)
        print('action goal status: N/A (service; no MoveGroup/ExecuteTrajectory goal sent)', flush=True)
        settings = self.execution_config['home_precheck']
        print(f'group={self.group_name}; planning_time={settings["planning_time_s"]:.1f}s; '
              f'planning_attempts={settings["planning_attempts"]}', flush=True)
        print(f'start joint state (simulated): names={list(seed.joint_state.name)}, '
              f'positions={list(seed.joint_state.position)}', flush=True)
        print(f'home joint goal: {dict(zip(self.ARM_JOINT_NAMES, self.HOME_JOINT_VALUES))}', flush=True)
        try:
            if self.group_name != 'ur_manipulator' or len(self.HOME_JOINT_VALUES) != 6:
                raise EnvironmentError('HOME_GROUP_OR_GOAL_INVALID')
            model = self._branch_model()
            start_arm = checked_arm(joint_values(seed), self.ARM_JOINT_NAMES, model)
            goal_arm = checked_arm(goal_arm, self.ARM_JOINT_NAMES, model)
            print(f'HOME target branch: {branch_label}; goal={goal_arm}', flush=True)
            for name in self.ARM_JOINT_NAMES:
                print(f'  {name}: simulated={start_arm[name]:.9f}; goal={goal_arm[name]:.9f}; '
                      f'continuous={model[name]["continuous"]}; '
                      f'limits=[{model[name]["lower"]},{model[name]["upper"]}]', flush=True)
            start = explicit_state(seed, start_arm, self.ARM_JOINT_NAMES)
            target = explicit_state(seed, goal_arm, self.ARM_JOINT_NAMES)
            print('arm goal/trajectory: six joints only; auxiliary seed joints retained ONLY for collision geometry', flush=True)
            start_ok = self._home_state_validity(start, 'retreat')
            goal_ok = self._home_state_validity(target, 'home')
            if not start_ok or not goal_ok:
                raise EnvironmentError(f'HOME_INVALID_STATE: retreat_valid={start_ok}, home_valid={goal_ok}')
            errors = {n: abs(start_arm[n]-goal_arm[n]) for n in self.ARM_JOINT_NAMES}
            tolerance = settings['joint_tolerance_rad']
            print(f'home per-joint error (rad): {errors}', flush=True)
            if all(error <= tolerance for error in errors.values()):
                self.last_precheck_reason = 'ALREADY_AT_HOME'
                print(f'plan retreat -> home ....... PASS: ALREADY_AT_HOME\n'
                      f'home trajectory points ..... 0\nhome final joint error ..... {max(errors.values()):.9f}', flush=True)
                return True
            if not hasattr(self, 'home_plan_client'):
                self.home_plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
            if not self.home_plan_client.wait_for_service(timeout_sec=self.execution_config['server_timeout_s']):
                raise EnvironmentError('HOME_PLANNING_SERVICE_UNAVAILABLE: /plan_kinematic_path')
            request = GetMotionPlan.Request()
            motion = request.motion_plan_request
            motion.group_name = self.group_name
            motion.start_state = start  # Explicit/non-diff, never actual current arm state.
            constraint = Constraints(name='home feasibility joint goal')
            constraint.joint_constraints = [JointConstraint(joint_name=n, position=goal_arm[n],
                tolerance_above=tolerance, tolerance_below=tolerance, weight=1.) for n in self.ARM_JOINT_NAMES]
            motion.goal_constraints = [constraint]
            motion.allowed_planning_time = settings['planning_time_s']
            motion.num_planning_attempts = settings['planning_attempts']
            motion.max_velocity_scaling_factor = self.motion_scaling_factor
            motion.max_acceleration_scaling_factor = self.motion_scaling_factor
            # Empty pipeline/planner IDs mean the running MoveIt's configured
            # defaults, not a hard-coded OMPL planner or a fallback mock.
            timeout = settings['planning_time_s']*settings['planning_attempts'] + 10.
            result = self.wait_result(self.home_plan_client.call_async(request), timeout).motion_plan_response
            code = result.error_code
            if hasattr(self, 'precheck_report'):
                self.precheck_report['home_error_code'] = code.val
                self.precheck_report['home_points'] = len(result.trajectory.joint_trajectory.points)
            print(f'HOME service response: MoveItErrorCodes.val={code.val} ({self.moveit_error_name(code.val)}); '
                  f'message={code.message or "<not provided by MoveIt>"!r}; '
                  f'source={code.source or "<not provided by MoveIt>"!r}; planning_time={result.planning_time:.6f}s', flush=True)
            if code.val != MoveItErrorCodes.SUCCESS:
                kind = 'PLANNING_TIMEOUT' if code.val == MoveItErrorCodes.TIMED_OUT else 'PLANNING_FAILURE'
                raise EnvironmentError(f'HOME_{kind}: error_code={code.val} ({self.moveit_error_name(code.val)}), '
                                       f'message={code.message!r}, source={code.source!r}')
            trajectory = result.trajectory.joint_trajectory
            print(f'home trajectory points ..... {len(trajectory.points)}', flush=True)
            if (result.group_name != self.group_name or len(trajectory.joint_names) != 6
                    or set(trajectory.joint_names) != set(self.ARM_JOINT_NAMES) or len(trajectory.points) < 2):
                raise EnvironmentError('HOME_INVALID_TRAJECTORY: wrong group/joints or no connecting path')
            returned_start = checked_arm(joint_values(result.trajectory_start), self.ARM_JOINT_NAMES, model)
            if max(abs(returned_start[n]-start_arm[n]) for n in self.ARM_JOINT_NAMES) > settings['start_tolerance_rad']:
                raise EnvironmentError(f'HOME_WRONG_TRAJECTORY_START: expected={start_arm}, returned={returned_start}')
            endpoint = self._guard_trajectory(result.trajectory, start, goal=goal_arm, label=branch_label)
            values = joint_values(endpoint)
            errors = {n: abs(values[n]-goal_arm[n]) for n in self.ARM_JOINT_NAMES}
            print(f'home final per-joint error ..... {errors}\nhome final joint error ..... {max(errors.values()):.9f}', flush=True)
            if any(error > tolerance for error in errors.values()):
                raise EnvironmentError('HOME_GOAL_NOT_REACHED_BY_PLAN')
            if not self._home_state_validity(explicit_state(seed, values, self.ARM_JOINT_NAMES), 'home endpoint'):
                raise EnvironmentError('HOME_ENDPOINT_INVALID')
            self.last_precheck_reason = 'HOME_PLAN_SUCCESS'
            self._last_home_trajectory = copy.deepcopy(result.trajectory)
            self._last_home_goal = copy.deepcopy(goal_arm)
            if hasattr(self, 'precheck_report'):
                self.precheck_report['home_final_error'] = max(errors.values())
            print('plan retreat -> home ....... PASS', flush=True)
            return True
        except Exception as error:
            if isinstance(error, EnvironmentError):
                self.last_precheck_reason = str(error)
            else:
                kind = 'SERVICE_TIMEOUT' if 'timeout' in str(error).lower() else 'EXCEPTION'
                self.last_precheck_reason = f'HOME_{kind}: {error} (no successful MoveIt response)'
            print(f'plan retreat -> home ....... FAIL\n{self.last_precheck_reason}', flush=True)
            return False

    def feasibility_precheck(self, plan, state, temporary_candidates=None, on_temporary_selected=None):
        self.precheck_scene_restored = False
        self.precheck_report = {'cartesian': [], 'pass': False}
        print('\nMOTION FEASIBILITY PRECHECK:', flush=True)
        if not self.simulator_is_ready():
            raise EnvironmentError('SIMULATOR_NOT_READY')
        if self.get_tool0_pose() is None:
            raise EnvironmentError('FRAME_TRANSFORM_FAILURE: base_link -> tool0')
        deadline = time.monotonic() + 2.0
        while not all(n in self.joint_states for n in self.ARM_JOINT_NAMES) and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not all(n in self.joint_states for n in self.ARM_JOINT_NAMES):
            raise EnvironmentError('JOINT_STATE_UNAVAILABLE')
        print('TF base_link -> tool0 ........ PASS', flush=True)
        original = self.read_planning_scene()
        if original.robot_state.attached_collision_objects or self.attached_object:
            raise EnvironmentError('OBJECT_ALREADY_ATTACHED: recover held object before a new plan')
        if 'table' not in {item.id for item in original.world.collision_objects}:
            raise EnvironmentError('PLANNING_SCENE_NOT_READY: table collision object missing')
        self.sync_camera_scene(state)
        baseline = self.read_planning_scene()
        seed = RobotState(is_diff=True)
        seed.joint_state.name = list(self.joint_states)
        seed.joint_state.position = [self.joint_states[n][0] for n in seed.joint_state.name]
        self.precheck_report['initial_joints'] = dict(zip(seed.joint_state.name, seed.joint_state.position))
        destinations = {}
        passed = True
        simulated_home_start = None
        self.home_start_source = 'initial state (home-only plan)'
        try:
            print('\nGRASP FEASIBILITY:', flush=True)
            for step in plan['plan']:
                name = step.get('object')
                if step['skill'] == 'pick':
                    targets = self.tool_targets(state['objects'][name]['position'])
                    # Actual pick opens before approaching; model this in the
                    # virtual seed, not by commanding the physical gripper.
                    seed = self._seed_gripper(seed, 'open')
                    self._allow_grasp_touch(baseline, name)
                    ok, seed = self._ik(targets[0][1], seed, f'{name} pre-grasp')
                    if not ok:
                        raise EnvironmentError(f'MOTION_PRECHECK_FAILED: {name} pre-grasp: {self.last_precheck_reason}')
                    center_z = state['objects'][name]['position']['z']
                    print(f'{name} GRASP GEOMETRY: cube_center_z={center_z:.6f}, '
                          f'cube_top_z={center_z+self.cube_height/2:.6f}, '
                          f'tool0_to_cube_center={self.grasp_object_offset:.6f}, finger_reach={self.finger_reach:.6f}, '
                          f'grasp_tool_z={targets[1][1][2]-self.robot_base_height:.6f}, '
                          f'finger_tip_z={targets[1][1][2]-self.robot_base_height-self.finger_reach:.6f}', flush=True)
                    ok, seed = self._cartesian_precheck(targets[0][1], targets[1][1], seed,
                                                        f'{name} Cartesian lowering')
                    if not ok:
                        raise EnvironmentError(f'MOTION_PRECHECK_FAILED: {name} lowering: {self.last_precheck_reason}')
                    if not self.remove_world_object(name) or not self.update_attached_object(name, attached=True):
                        raise EnvironmentError('PRECHECK_VIRTUAL_ATTACH_FAILED')
                    self._confirm_virtual_attachment(name)
                    self._reset_grasp_touch(baseline)
                    seed = self._seed_gripper(seed, 'holding')
                    ok, seed = self._cartesian_precheck(targets[1][1], targets[2][1], seed,
                                                        f'{name} Cartesian lift (attached cube)')
                    if not ok:
                        raise EnvironmentError(f'MOTION_PRECHECK_FAILED: {name} lift: {self.last_precheck_reason}')
                elif step['skill'] == 'place':
                    dest = step['destination']
                    center = destination_center(self.scene, plan, dest)
                    offsets = (self.bounded_place_offsets(self.zone_usable_size[1], self.cube_size[1],
                                                         margin=self.execution_config['zone_offset_margin_m'])
                               if dest in self.scene['zones'] else [0.0])
                    choices = ([(dest, dict(center, y=center['y']+offset)) for offset in offsets]
                               if dest in self.scene['zones'] else
                               list((temporary_candidates if temporary_candidates is not None
                                     else plan['temporary_positions']).items()))
                    if dest not in self.scene['zones']:
                        print('\nTEMPORARY CANDIDATE SEARCH (MoveIt attached cube):', flush=True)
                    selected = None
                    for candidate, point in choices:
                        candidate_ok = True
                        candidate_seed = copy.deepcopy(seed)
                        reasons = []
                        for label, xyz in self.tool_targets(point, place=True):
                            ok, candidate_seed = self._ik(xyz, candidate_seed, f'{candidate} {label}')
                            candidate_ok &= ok
                            if not ok:
                                reasons.append(f'{label}: {self.last_precheck_reason}')
                        if candidate_ok:
                            selected, seed = point, candidate_seed
                            if dest not in self.scene['zones']:
                                print(f'{candidate} ... PASS\nSELECTED: {candidate} {point}', flush=True)
                                step['destination'] = candidate
                                plan['temporary_positions'] = {candidate: copy.deepcopy(point)}
                                dest = candidate
                                if on_temporary_selected is not None:
                                    on_temporary_selected(plan)
                            break
                        print(f'{candidate} ... REJECTED: {"; ".join(reasons)}', flush=True)
                    if selected is None:
                        raise EnvironmentError(f'MOTION_PRECHECK_FAILED: {dest}: NO_REACHABLE_PLACE_CANDIDATE')
                    destinations[dest] = copy.deepcopy(selected)
                    if dest in self.scene['zones']:
                        print(f'{dest} selected zone y_offset={selected["y"]-center["y"]:+.3f} m', flush=True)
                    else:
                        self.precheck_report['temporary'] = dest
                        print(f'{dest} EXACT cube-centre coordinates=({selected["x"]:.3f}, '
                              f'{selected["y"]:.3f}, {selected["z"]:.3f}); NO ZONE OFFSET', flush=True)
                    if not self.update_attached_object(name, attached=False,
                                                       world_position=self.world_xyz(destinations[dest])):
                        raise EnvironmentError('PRECHECK_VIRTUAL_DETACH_FAILED')
                    seed = self._seed_gripper(seed, 'open')
                    ok, seed = self._ik(self.world_xyz(self.execution_config['camera_retreat']), seed, 'camera retreat')
                    passed &= ok
                    simulated_home_start = copy.deepcopy(seed) if ok else None
                    self.home_start_source = f'{dest} retreat endpoint (camera retreat IK)' if ok else f'{dest} retreat FAILED'
                else:
                    if destinations and simulated_home_start is None:
                        raise EnvironmentError('MOTION_PRECHECK_FAILED: HOME: latest simulated retreat did not PASS')
                    home_start = simulated_home_start if simulated_home_start is not None else copy.deepcopy(seed)
                    if not self._home_precheck(home_start):
                        raise EnvironmentError(f'MOTION_PRECHECK_FAILED: {self.last_precheck_reason}')
        finally:
            # A timed-out plan-only action must stop planning against the
            # virtual scene before rollback; this sends NO gripper commands.
            cancel_error = None
            if getattr(self, 'active_motion_handle', None) is not None:
                try:
                    self.cancel_pending_precheck()
                except Exception as error:
                    cancel_error = error
            # Roll back ALL virtual attachments, moved cubes and touch ACM.
            # Never apply simulated arm joint positions to the monitored robot.
            rollback = copy.deepcopy(baseline)
            rollback.is_diff = True
            rollback.robot_state = RobotState(is_diff=True)
            # Explicit false entries clear virtual touch pairs even if MoveIt's
            # diff merge retains names that were absent from the baseline ACM.
            rollback.allowed_collision_matrix = extend_collision_matrix(
                baseline.allowed_collision_matrix, [*state['objects'],
                    'simple_gripper_base_link', 'simple_gripper_left_finger_link',
                    'simple_gripper_right_finger_link'])
            for name in state['objects']:
                item = AttachedCollisionObject(link_name=self.ee_link)
                item.object = CollisionObject(id=name, operation=CollisionObject.REMOVE)
                rollback.robot_state.attached_collision_objects.append(item)
            try:
                self._apply_scene(rollback)
                self._verify_scene_restored(baseline)
                self.precheck_scene_restored = True
                print('PLANNING SCENE RESTORED', flush=True)
                self.precheck_report['scene_restored'] = True
            except Exception as error:
                raise EnvironmentError(f'PRECHECK_SCENE_RESTORE_FAILED: {error}') from error
            if cancel_error is not None:
                raise EnvironmentError(f'PRECHECK_PLAN_CANCEL_NOT_CONFIRMED: {cancel_error}') from cancel_error
        if not passed:
            raise EnvironmentError('MOTION_PRECHECK_FAILED: one or more targets are not collision-free/reachable')
        self.precheck_report['pass'] = True
        print('MOTION FEASIBILITY PRECHECK: PASS (Cartesian grasp/lift, attached place IK, plan-only home)', flush=True)
        return destinations

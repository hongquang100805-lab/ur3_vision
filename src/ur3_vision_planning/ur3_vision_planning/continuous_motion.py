"""Seed-aware motion adapter: all planning is separate from guarded execution."""

import copy
import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetCartesianPath, GetMotionPlan, GetPositionFK, GetStateValidity
from geometry_msgs.msg import Pose

from .environment_manager import EnvironmentError
from .home_precheck import explicit_state, joint_values
from .joint_continuity import branch_candidates, checked_arm, check_trajectory, deltas


class ContinuousMotionMixin:
    def _branch_model(self):
        if not hasattr(self, '_live_joint_model'):
            self._live_joint_model = self._home_model()
        return self._live_joint_model

    def _actual_seed(self):
        received = getattr(self, '_joint_sample_count', 0)
        deadline = time.monotonic()+self.execution_config['service_timeout_s']
        while rclpy.ok() and getattr(self, '_joint_sample_count', 0) <= received and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.05)
        if getattr(self, '_joint_sample_count', 0) <= received:
            raise EnvironmentError('ACTUAL_JOINT_STATE_TIMEOUT')
        state = RobotState(is_diff=True)
        state.joint_state.name = list(self.joint_states)
        state.joint_state.position = [self.joint_states[n][0] for n in state.joint_state.name]
        checked_arm(joint_values(state), self.ARM_JOINT_NAMES, self._branch_model())
        return state

    def gripper_state_callback(self, msg):
        super().gripper_state_callback(msg)
        if all(n in msg.name for n in self.ARM_JOINT_NAMES):
            self._joint_sample_count = getattr(self, '_joint_sample_count', 0)+1

    def _branch_validity(self, state, label):
        # Diff intentionally preserves the virtual/actual scene's attachment.
        state = copy.deepcopy(state)
        state.is_diff = True
        result = self.wait_result(self.state_validity_client.call_async(
            GetStateValidity.Request(robot_state=state, group_name='')), self.execution_config['service_timeout_s'])
        print(f'{label} state validity {"PASS" if result.valid else "FAIL"}', flush=True)
        for contact in result.contacts:
            print(f'  contact {contact.contact_body_1} <-> {contact.contact_body_2}', flush=True)
        return result.valid

    def _normalize_branch(self, raw_state, seed, xyz, label, diagnostic=False):
        names, model = self.ARM_JOINT_NAMES, self._branch_model()
        previous = checked_arm(joint_values(seed), names, model)
        raw = joint_values(raw_state)
        config = self.execution_config['continuity']
        for values in branch_candidates(raw, previous, names, model, maximum=config['maximum_equivalent_candidates']):
            physical, angular = deltas(previous, values, names)
            # Angular delta is <=pi by definition; ALSO check the unwrapped
            # command delta so a hidden 2pi jump cannot pass this test.
            if max(abs(v) for v in physical.values()) > config['maximum_ik_delta_rad']:
                continue
            state = explicit_state(seed, values, names)
            state.is_diff = True
            wrapped = {n: (values[n]-raw[n])/(2*math.pi) for n in names if abs(values[n]-raw[n]) > 1e-8}
            print(f'BRANCH CONTINUITY: {label}\n  seed={previous}\n  raw IK={ {n: raw[n] for n in names} }'
                  f'\n  normalized IK={values}\n  maximum joint delta={max(abs(v) for v in physical.values()):.9f}; '
                  f'angular_delta={angular}; wrapped joints (2*pi multiples)={wrapped}', flush=True)
            if not self._fk_matches(state, xyz, f'{label} FK equivalence'):
                continue
            if not diagnostic and not self._branch_validity(state, label):
                continue
            return state
        raise EnvironmentError(f'IK_BRANCH_CONTINUITY_FAILED: {label}: no near, in-limit, FK/validity-compatible branch')

    def _guard_trajectory(self, trajectory, seed, xyz=None, goal=None, label='trajectory', execution=False):
        config = self.execution_config['continuity']
        end, maximum = check_trajectory(trajectory.joint_trajectory, joint_values(seed), self.ARM_JOINT_NAMES,
            self._branch_model(), start_tolerance=(config['execution_start_tolerance_rad'] if execution else
                self.execution_config['home_precheck']['start_tolerance_rad']),
            point_delta_limit=config['maximum_trajectory_delta_rad'], goal=goal,
            goal_tolerance=self.execution_config['home_precheck']['joint_tolerance_rad'])
        endpoint = explicit_state(seed, end, self.ARM_JOINT_NAMES)
        endpoint.is_diff = True
        if xyz is not None and not self._fk_matches(endpoint, xyz, f'{label} endpoint'):
            raise EnvironmentError(f'TRAJECTORY_ENDPOINT_FK_FAILED: {label}')
        if not self._branch_validity(endpoint, f'{label} endpoint'):
            raise EnvironmentError(f'TRAJECTORY_ENDPOINT_COLLISION: {label}')
        print(f'TRAJECTORY GUARD: {label} PASS; points={len(trajectory.joint_trajectory.points)}, '
              f'maximum point delta={maximum:.9f}', flush=True)
        return endpoint

    def _fk_equivalent(self, first, second, label):
        poses = []
        for state in (first, second):
            request = GetPositionFK.Request(robot_state=copy.deepcopy(state), fk_link_names=[self.ee_link])
            request.header.frame_id = self.frame_id
            response = self.wait_result(self.fk_client.call_async(request), self.execution_config['service_timeout_s'])
            if (response.error_code.val != 1 or list(response.fk_link_names) != [self.ee_link]
                    or len(response.pose_stamped) != 1 or response.pose_stamped[0].header.frame_id != self.frame_id):
                raise EnvironmentError('HOME_EQUIVALENCE_FK_UNAVAILABLE')
            poses.append(response.pose_stamped[0].pose)
        a, b = poses
        distance = math.dist([a.position.x, a.position.y, a.position.z], [b.position.x, b.position.y, b.position.z])
        qa, qb = [getattr(a.orientation, k) for k in 'xyzw'], [getattr(b.orientation, k) for k in 'xyzw']
        if not all(math.isfinite(v) for v in (*qa, *qb, distance)):
            return False
        na, nb = math.sqrt(sum(v*v for v in qa)), math.sqrt(sum(v*v for v in qb))
        passed = .99 <= na <= 1.01 and .99 <= nb <= 1.01 and distance <= 1e-5
        passed = passed and 2*math.acos(min(1., abs(sum(x*y for x,y in zip(qa,qb)))/(na*nb))) <= 1e-4
        print(f'{label} FK equivalence {"PASS" if passed else "FAIL"}', flush=True)
        return passed

    def _execute_checked(self, trajectory, xyz=None, goal=None, label='motion'):
        # Read a NEW actual joint sample immediately before the ONLY arm action.
        seed = self._actual_seed()
        self._guard_trajectory(trajectory, seed, xyz, goal, label, execution=True)
        request = ExecuteTrajectory.Goal(trajectory=trajectory)
        self._motion_submission_pending = True
        handle = self.wait_result(self.execute_client.send_goal_async(request), timeout=30.)
        if not handle.accepted:
            self._motion_submission_pending = False
            return 'EXECUTION_REJECTED'
        response = self.wait_trajectory_result(handle.get_result_async(), timeout=90.)
        self._motion_submission_pending = False
        if response.status != GoalStatus.STATUS_SUCCEEDED or response.result.error_code.val != 1:
            return 'EXECUTION_FAILED'
        final = self._actual_seed()
        if goal is not None and max(abs(joint_values(final)[n]-goal[n]) for n in self.ARM_JOINT_NAMES) > .02:
            return 'EXECUTION_ENDPOINT_JOINT_MISMATCH'
        if xyz is not None and not self._fk_matches(final, xyz, f'{label} actual endpoint'):
            return 'EXECUTION_ENDPOINT_POSE_MISMATCH'
        return 'SUCCESS'

    def _plan_joint_motion(self, seed, target, label):
        if not hasattr(self, 'home_plan_client'):
            self.home_plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        if not self.home_plan_client.wait_for_service(timeout_sec=self.execution_config['server_timeout_s']):
            raise EnvironmentError('MOTION_PLAN_SERVER_UNAVAILABLE')
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = self.group_name
        motion.start_state = copy.deepcopy(seed)
        motion.start_state.is_diff = True  # Full supplied joints + live attachment.
        motion.allowed_planning_time = self.execution_config['home_precheck']['planning_time_s']
        motion.num_planning_attempts = self.execution_config['home_precheck']['planning_attempts']
        motion.max_velocity_scaling_factor = motion.max_acceleration_scaling_factor = self.motion_scaling_factor
        tolerance = self.execution_config['home_precheck']['joint_tolerance_rad']
        motion.goal_constraints = [Constraints(joint_constraints=[JointConstraint(joint_name=n, position=target[n],
            tolerance_above=tolerance, tolerance_below=tolerance, weight=1.) for n in self.ARM_JOINT_NAMES])]
        response = self.wait_result(self.home_plan_client.call_async(request), timeout=55.).motion_plan_response
        print(f'{label}: plan-only MoveIt code={response.error_code.val}; '
              f'message={response.error_code.message!r}; source={response.error_code.source!r}', flush=True)
        if response.error_code.val != 1 or response.group_name != self.group_name:
            raise EnvironmentError(f'MOTION_PLAN_FAILED: {label}: code={response.error_code.val}')
        self._guard_trajectory(response.trajectory, seed, goal=target, label=label)
        return response.trajectory

    def move_to_pose(self, x, y, z):
        try:
            seed = self._actual_seed()
            ok, target = self._ik((x,y,z), seed, 'execution pose IK')
            if not ok:
                return 'PLANNING_FAILED'
            goal = {n: joint_values(target)[n] for n in self.ARM_JOINT_NAMES}
            trajectory = self._plan_joint_motion(seed, goal, 'execution pose')
            return self._execute_checked(trajectory, xyz=(x,y,z), goal=goal, label='execution pose')
        except Exception as error:
            print(f'MOTION REFUSED: {error}', flush=True)
            return ('EXECUTION_STATE_UNCERTAIN' if self.active_motion_handle is not None
                    or getattr(self, '_motion_submission_pending', False) else 'PLANNING_FAILED')

    def move_to_verified_place_pose(self, x, y, z, label):
        return self.move_to_pose(x,y,z)

    def diagnose_place_pose(self, x, y, z, label):
        return self._ik((x,y,z), self._actual_seed(), label)[0]

    def home(self):
        try:
            seed = self._actual_seed()
            self.home_start_source = 'actual current joint state (execution home)'
            if not self._home_precheck(seed):
                return 'PLANNING_FAILED'
            if self.last_precheck_reason == 'ALREADY_AT_HOME':
                return 'SUCCESS'
            return self._execute_checked(self._last_home_trajectory, goal=self._last_home_goal, label='execution home')
        except Exception as error:
            print(f'HOME EXECUTION REFUSED: {error}', flush=True)
            return ('EXECUTION_STATE_UNCERTAIN' if self.active_motion_handle is not None
                    or getattr(self, '_motion_submission_pending', False) else 'PLANNING_FAILED')

    def _continuous_cartesian_execution(self, waypoints, diagnostic_label):
        try:
            if not waypoints:
                raise EnvironmentError('EMPTY_CARTESIAN_TARGET')
            seed = self._actual_seed()
            request = GetCartesianPath.Request()
            request.header.frame_id = self.frame_id
            request.group_name, request.link_name = self.group_name, self.ee_link
            request.start_state = copy.deepcopy(seed)
            request.start_state.is_diff = True
            request.max_step = self.execution_config['cartesian_step_m']
            request.avoid_collisions = True
            request.max_velocity_scaling_factor = request.max_acceleration_scaling_factor = self.motion_scaling_factor
            for x,y,z in waypoints:
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = x,y,z-self.robot_base_height
                pose.orientation = copy.deepcopy(self.down_orientation)
                request.waypoints.append(pose)
            result = self.wait_result(self.cartesian_client.call_async(request), timeout=30.)
            print(f'{diagnostic_label}: Cartesian fraction={result.fraction:.3f}, code={result.error_code.val}', flush=True)
            if result.error_code.val != 1 or not math.isfinite(result.fraction) or result.fraction < 1.:
                return 'PLANNING_FAILED'
            self._guard_trajectory(result.solution, seed, xyz=waypoints[-1], label=diagnostic_label)
            return self._execute_checked(result.solution, xyz=waypoints[-1], label=diagnostic_label)
        except Exception as error:
            print(f'CARTESIAN MOTION REFUSED: {error}', flush=True)
            return ('EXECUTION_STATE_UNCERTAIN' if self.active_motion_handle is not None
                    or getattr(self, '_motion_submission_pending', False) else 'PLANNING_FAILED')

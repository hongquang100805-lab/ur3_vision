"""No ROS initialization or arm motion. Pure guards + service response fixtures."""

import copy
import math
from types import MethodType, SimpleNamespace

import pytest
from builtin_interfaces.msg import Duration
from moveit_msgs.msg import MoveItErrorCodes, MotionPlanResponse, RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from test_home_precheck import NAMES, HOME, model, xml_model, home_fixture
from test_plan_execution import config
from ur3_vision_planning.environment_manager import EnvironmentError
from ur3_vision_planning.home_precheck import home_joint_model, joint_values
from ur3_vision_planning.joint_continuity import branch_candidates, check_trajectory, deltas
from ur3_vision_planning.motion_precheck import repeat_summary
from ur3_vision_planning.vision_robot_skills import VisionRobotSkills


def state(values):
    result = RobotState()
    result.joint_state.name = list(NAMES)+['simple_gripper_left_finger_joint', 'simple_gripper_right_finger_joint']
    result.joint_state.position = [values[n] for n in NAMES]+[0., .043]
    return result


def path(start=None, end=None, steps=40):
    start, end = start or HOME, end or dict(HOME, shoulder_pan_joint=.2)
    return RobotTrajectory(joint_trajectory=JointTrajectory(joint_names=list(NAMES), points=[
        JointTrajectoryPoint(positions=[start[n]+(end[n]-start[n])*i/steps for n in NAMES],
                             time_from_start=Duration(sec=i)) for i in range(steps+1)]))


def test_given_two_pi_branches_become_same_nearest_seed():
    reference = dict(zip(NAMES, (2.734, -1.551, -.896, -2.265, -4.712, -1.978)))
    raw = dict(reference, shoulder_pan_joint=reference[NAMES[0]]-2*math.pi,
               shoulder_lift_joint=reference[NAMES[1]]+2*math.pi,
               wrist_1_joint=reference[NAMES[3]]+2*math.pi)
    result = branch_candidates(raw, reference, NAMES, model())[0]
    assert result == pytest.approx(reference)
    physical, angular = deltas(reference, result, NAMES)
    assert max(abs(v) for v in physical.values()) < 1e-12
    assert max(abs(v) for v in angular.values()) < 1e-12


def test_prismatic_is_not_wrapped_and_outside_limit_rejected():
    names = ['slider']
    bounds = {'slider': {'type': 'prismatic', 'lower': -10., 'upper': 10.}}
    assert branch_candidates({'slider': 6.3}, {'slider': 0.}, names, bounds) == [{'slider': 6.3}]
    with pytest.raises(EnvironmentError, match='NO_IN_LIMIT'):
        branch_candidates({'slider': 11.}, {'slider': 0.}, names, bounds)


def test_nearest_branch_does_not_cross_limits_or_home_safety_margin():
    bounds = {'j': {'type': 'revolute', 'lower': -2*math.pi, 'upper': 2*math.pi}}
    result = branch_candidates({'j': 0.}, {'j': 6.}, ['j'], bounds, margin=.05)
    assert result == [{'j': 0.}]


def test_live_urdf_soft_limits_are_respected():
    urdf, srdf = xml_model()
    urdf = urdf.replace('<child link="link_0"/>', '<child link="link_0"/><safety_controller '
                        'soft_lower_limit="-6.0" soft_upper_limit="6.0"/>')
    result = home_joint_model(urdf, srdf, 'ur_manipulator', NAMES)
    assert result[NAMES[0]]['lower'] == -6. and result[NAMES[0]]['upper'] == 6.


@pytest.mark.parametrize('failure', ['nan', 'inf', 'names', 'start', 'jump', 'time', 'negative_time',
                                   'velocity', 'endpoint', 'outside_bounds'])
def test_trajectory_guard_rejects_before_action(failure):
    trajectory = path().joint_trajectory
    if failure in ('nan', 'inf'):
        trajectory.points[1].positions[0] = float(failure)
    elif failure == 'names':
        trajectory.joint_names[-1] = 'gripper_joint'
    elif failure == 'start':
        trajectory.points[0].positions[0] = .3
    elif failure == 'jump':
        trajectory.points[1].positions[0] = 2*math.pi-.01
    elif failure == 'time':
        trajectory.points[1].time_from_start = Duration()
    elif failure == 'negative_time':
        trajectory.points[0].time_from_start = Duration(sec=-1)
    elif failure == 'velocity':
        trajectory.points[1].velocities = [math.nan]*6
    elif failure == 'endpoint':
        trajectory.points[-1].positions[0] = .3
    else:
        trajectory.points[1].positions[2] = 4.
    with pytest.raises(EnvironmentError):
        check_trajectory(trajectory, HOME, NAMES, model(), goal=dict(HOME, shoulder_pan_joint=.2))


def test_trajectory_first_point_periodic_equivalence_is_not_command_equivalence():
    start = dict(HOME, shoulder_pan_joint=-3.)
    trajectory = path(start).joint_trajectory
    trajectory.points[0].positions[0] += 2*math.pi
    with pytest.raises(EnvironmentError, match='START_MISMATCH'):
        check_trajectory(trajectory, start, NAMES, model())


def normalized_fixture(config):
    fake = SimpleNamespace(ARM_JOINT_NAMES=NAMES, execution_config=config, _branch_model=model,
        _fk_matches=lambda *args: True, _branch_validity=lambda *args: True)
    fake._normalize_branch = MethodType(VisionRobotSkills._normalize_branch, fake)
    return fake


def test_normalization_preserves_seed_gripper_and_requires_fk_collision(config):
    seed = state(HOME)
    raw = state(dict(HOME, shoulder_lift_joint=HOME[NAMES[1]]+2*math.pi))
    raw.joint_state.position[-1] = 0.  # IK response auxiliary values must not replace virtual open fingers.
    fake = normalized_fixture(config)
    normalized = fake._normalize_branch(raw, seed, (.38, -.15, .4), 'red pre-grasp')
    assert joint_values(normalized)[NAMES[1]] == pytest.approx(HOME[NAMES[1]])
    assert joint_values(normalized)['simple_gripper_right_finger_joint'] == .043
    for field in ('_fk_matches', '_branch_validity'):
        rejected = normalized_fixture(config)
        setattr(rejected, field, lambda *args: False)
        with pytest.raises(EnvironmentError, match='IK_BRANCH_CONTINUITY_FAILED'):
            rejected._normalize_branch(raw, seed, (.38, -.15, .4), 'red pre-grasp')


def test_ik_never_uses_empty_seed_or_actual_state_for_chain(config):
    fake = normalized_fixture(config)
    seed = state(dict(HOME, shoulder_pan_joint=.7))
    sent = []
    fake.group_name, fake.ee_link, fake.frame_id, fake.robot_base_height = 'ur_manipulator', 'tool0', 'base_link', 0.
    from geometry_msgs.msg import Quaternion
    fake.down_orientation = Quaternion(x=1.)
    fake.moveit_error_name = lambda v: 'SUCCESS'
    fake.wait_result = lambda f, timeout: f
    fake.ik_client = SimpleNamespace(call_async=lambda request: sent.append(request) or SimpleNamespace(
        error_code=MoveItErrorCodes(val=1), solution=state(dict(HOME, shoulder_pan_joint=.75))))
    assert VisionRobotSkills._ik(fake, (.38,-.15,.4), seed, 'first')[0]
    assert joint_values(sent[0].ik_request.robot_state)['shoulder_pan_joint'] == .7
    with pytest.raises(EnvironmentError):
        VisionRobotSkills._ik(fake, (.38,-.15,.4), RobotState(), 'empty')
    assert len(sent) == 1


def test_execution_guard_rejects_two_pi_jump_before_send(config):
    fake = normalized_fixture(config)
    fake._actual_seed = lambda: state(HOME)
    fake._guard_trajectory = MethodType(VisionRobotSkills._guard_trajectory, fake)
    sent = []
    fake.execute_client = SimpleNamespace(send_goal_async=lambda goal: sent.append(goal))
    trajectory = path()
    trajectory.joint_trajectory.points[1].positions[0] = 2*math.pi-.01
    with pytest.raises(EnvironmentError, match='JOINT_JUMP'):
        VisionRobotSkills._execute_checked(fake, trajectory)
    assert sent == []


def test_execution_uses_guard_then_one_action_then_checks_actual_endpoint(config):
    from action_msgs.msg import GoalStatus
    fake = normalized_fixture(config)
    goal = dict(HOME, shoulder_pan_joint=.2)
    samples = iter([state(HOME), state(goal)])
    fake._actual_seed = lambda: next(samples)
    fake._guard_trajectory = MethodType(VisionRobotSkills._guard_trajectory, fake)
    sent = []
    response = SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED,
                               result=SimpleNamespace(error_code=MoveItErrorCodes(val=1)))
    handle = SimpleNamespace(accepted=True, get_result_async=lambda: response)
    fake.execute_client = SimpleNamespace(send_goal_async=lambda request: sent.append(request) or handle)
    fake.wait_result = fake.wait_trajectory_result = lambda future, timeout: future
    assert VisionRobotSkills._execute_checked(fake, path(HOME,goal), goal=goal) == 'SUCCESS'
    assert len(sent) == 1 and fake._motion_submission_pending is False


def test_unknown_submission_timeout_is_not_safe_planning_failure(config):
    fake = normalized_fixture(config)
    fake._actual_seed = lambda: state(HOME)
    fake._guard_trajectory = MethodType(VisionRobotSkills._guard_trajectory, fake)
    fake._execute_checked = MethodType(VisionRobotSkills._execute_checked, fake)
    fake._ik = lambda *args: (True, state(dict(HOME, shoulder_pan_joint=.2)))
    fake._plan_joint_motion = lambda *args: path()
    fake.active_motion_handle = None
    sent = []
    fake.execute_client = SimpleNamespace(send_goal_async=lambda request: sent.append(request) or 'future')
    fake.wait_result = lambda *args, **kw: (_ for _ in ()).throw(RuntimeError('Timeout'))
    assert VisionRobotSkills.move_to_pose(fake, .38,-.15,.4) == 'EXECUTION_STATE_UNCERTAIN'
    assert len(sent) == 1 and fake._motion_submission_pending is True
    fake.execution_config = config
    with pytest.raises(EnvironmentError, match='EXECUTION_SUBMISSION_UNCERTAIN'):
        VisionRobotSkills.cancel_pending_precheck(fake)


def test_home_equivalent_candidate_is_distinct_verified_and_nearest(config):
    fake, seed, sent, _ = home_fixture(config)
    fk = []
    fake._fk_equivalent = lambda candidate, reference, label: fk.append((candidate,reference)) or True
    def plan(request):
        sent.append(request)
        motion = request.motion_plan_request
        start = {n: joint_values(motion.start_state)[n] for n in NAMES}
        goal = {c.joint_name: c.position for c in motion.goal_constraints[0].joint_constraints}
        code = 99999 if len(sent) == 1 else 1
        return SimpleNamespace(motion_plan_response=MotionPlanResponse(group_name='ur_manipulator',
            trajectory_start=copy.deepcopy(motion.start_state), trajectory=path(start,goal),
            error_code=MoveItErrorCodes(val=code)))
    fake.home_plan_client.call_async = plan
    assert VisionRobotSkills._home_precheck(fake, seed)
    assert len(sent) == 2 and fk
    goal = fake._last_home_goal
    assert goal['wrist_3_joint'] == pytest.approx(4*math.pi)
    assert all(model()[n]['lower']+.05 <= goal[n] <= model()[n]['upper']-.05 for n in NAMES)


def test_three_unit_winding_responses_produce_identical_retreat_and_report(config):
    reference = dict(zip(NAMES, (2.734,-1.551,-.896,-2.265,-4.712,-1.978)))
    fake = normalized_fixture(config)
    reports = []
    for run in range(3):
        raw = dict(reference)
        raw[NAMES[0]] -= 2*math.pi if run % 2 else 0.
        raw[NAMES[1]] += 2*math.pi if run % 2 else 0.
        raw[NAMES[3]] += 2*math.pi if run % 2 else 0.
        normalized = fake._normalize_branch(state(raw), state(reference), (.18,-.2,.45), f'unit run {run+1}')
        reports.append(dict(**{'pass': True}, scene_restored=True, scene_resynced=True,
            temporary='temporary_8', home_error_code=1, home_points=41, home_final_error=0.,
            cartesian=[{'pass': True, 'fraction': 1.}], retreat_joints={n: joint_values(normalized)[n] for n in NAMES}))
    assert repeat_summary(reports, .02, 'temporary_8')
    reports[2]['retreat_joints'][NAMES[0]] += 2*math.pi
    assert not repeat_summary(reports, .02, 'temporary_8')


def test_repeat_report_cannot_hide_home_failure():
    reports = [{'pass': True, 'home_error_code': 99999, 'retreat_joints': dict(HOME)}]*3
    assert repeat_summary(reports, .02, 'temporary_8') is False


def test_legacy_bai03_entrypoint_also_uses_guarded_robot_adapter():
    import inspect
    from ur3_vision_planning import skill_executor
    source = inspect.getsource(skill_executor.main)
    assert 'skills = VisionRobotSkills(scene_path, motion_config)' in source
    assert 'skills = RobotSkills(' not in source

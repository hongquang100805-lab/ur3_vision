"""HOME service/model fixtures only; never initialize ROS or execute motion."""

import copy
import math
from types import MethodType, SimpleNamespace
from builtin_interfaces.msg import Duration

import pytest
from moveit_msgs.msg import (
    ContactInformation, MotionPlanResponse, MoveItErrorCodes, RobotState, RobotTrajectory,
)
from rcl_interfaces.msg import ParameterType, ParameterValue
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

from test_environment_manager import planner, state
from test_motion_precheck import fake_precheck
from test_plan_execution import config
from ur3_vision_planning.environment_manager import EnvironmentError
from ur3_vision_planning.home_precheck import (
    explicit_state, home_joint_model, joint_errors, joint_values, normalize_arm,
)
from ur3_vision_planning.robot_skills import RobotSkills
from ur3_vision_planning.vision_robot_skills import VisionRobotSkills


NAMES = RobotSkills.ARM_JOINT_NAMES
HOME = dict(zip(NAMES, RobotSkills.HOME_JOINT_VALUES))


def xml_model():
    joints = []
    parent = 'base_link'
    for index, name in enumerate(NAMES):
        child = 'tool0' if index == 5 else f'link_{index}'
        kind = 'continuous' if index == 5 else 'revolute'
        bound = math.pi if name == 'elbow_joint' else 2*math.pi
        joints.append(f'<joint name="{name}" type="{kind}"><parent link="{parent}"/>'
                      f'<child link="{child}"/><limit lower="{-bound}" upper="{bound}"/></joint>')
        parent = child
    urdf = '<robot name="fixture">' + ''.join(joints) + '</robot>'
    srdf = '<robot name="fixture"><group name="ur_manipulator"><chain base_link="base_link" tip_link="tool0"/></group></robot>'
    return urdf, srdf


def model():
    return home_joint_model(*xml_model(), 'ur_manipulator', NAMES)


def seed_state(values=None):
    values = values or dict(zip(NAMES, (1.2, -2.1, .4, 2.7, 1.1, 4*math.pi+.2)))
    seed = RobotState(is_diff=True)
    seed.joint_state.name = list(reversed(NAMES)) + ['simple_gripper_left_finger_joint', 'simple_gripper_right_finger_joint']
    seed.joint_state.position = [values[n] for n in reversed(NAMES)] + [0., .043]
    return seed


def home_fixture(config, failure=None, start_values=None):
    sent, validity = [], []
    seed = seed_state(start_values)
    fake = SimpleNamespace(group_name='ur_manipulator', ARM_JOINT_NAMES=NAMES,
        HOME_JOINT_VALUES=RobotSkills.HOME_JOINT_VALUES, motion_scaling_factor=.2,
        execution_config=config, home_start_source='zone_b retreat endpoint (camera retreat IK)',
        _home_model=model, moveit_error_name=RobotSkills.moveit_error_name,
        joint_states={n: (0., 0.) for n in NAMES})  # Different actual state: must NOT be used.
    def check(req):
        validity.append(req)
        bad = ((failure == 'retreat_collision' and len(validity) == 1)
               or (failure == 'goal_collision' and len(validity) == 2)
               or (failure == 'endpoint_collision' and len(validity) == 3))
        contacts = [ContactInformation(contact_body_1='wrist_1_link', contact_body_2='table',
            body_type_1=ContactInformation.ROBOT_LINK, body_type_2=ContactInformation.WORLD_OBJECT)] if bad else []
        return SimpleNamespace(valid=not bad, contacts=contacts)
    fake.state_validity_client = SimpleNamespace(call_async=check)
    fake._home_state_validity = MethodType(VisionRobotSkills._home_state_validity, fake)
    fake._branch_model = model
    fake._branch_validity = lambda state, label: True
    for method in ('_select_home_branch', '_home_plan_candidate', '_guard_trajectory'):
        setattr(fake, method, MethodType(getattr(VisionRobotSkills, method), fake))
    # All equivalent targets fail in these failure fixtures; this is NOT a
    # random retry or permission to accept a failed canonical response.
    fake._fk_equivalent = lambda *args: True
    def plan(request):
        sent.append(request)
        if failure == 'rpc_timeout':
            raise RuntimeError('Timeout')
        start = request.motion_plan_request.start_state
        names = list(reversed(NAMES))
        first = [joint_values(start)[n] for n in names]
        last = [HOME[n] for n in names]
        code = MoveItErrorCodes.FAILURE if failure == '99999' else (
            MoveItErrorCodes.TIMED_OUT if failure == 'planning_timeout' else MoveItErrorCodes.SUCCESS)
        result = MotionPlanResponse(group_name='ur_manipulator', trajectory_start=copy.deepcopy(start),
            planning_time=.5, error_code=MoveItErrorCodes(val=code, message='fixture planner detail', source='fixture_pipeline'))
        if failure == 'wrong_start':
            result.trajectory_start.joint_state.position[0] = 0.
        if failure == 'wrong_first':
            first[0] += .1
        if failure == 'wrong_end':
            last[0] += .02
        if failure == 'wrong_group':
            result.group_name = 'gripper'
        if failure == 'gripper_trajectory':
            names.append('simple_gripper_left_finger_joint')
            first.append(0.)
            last.append(0.)
        # Time-parameterized, continuous service trajectory fixture.
        points = [JointTrajectoryPoint(positions=[a+(b-a)*i/40 for a,b in zip(first,last)],
                                      time_from_start=Duration(sec=i)) for i in range(41)]
        if failure == 'empty_path':
            points = []
        if failure == 'invalid_midpoint':
            middle = list(first)
            middle[2] = 100.
            points.insert(1, JointTrajectoryPoint(positions=middle))
        result.trajectory = RobotTrajectory(joint_trajectory=JointTrajectory(joint_names=names, points=points))
        return SimpleNamespace(motion_plan_response=result)
    fake.home_plan_client = SimpleNamespace(wait_for_service=lambda **kw: True, call_async=plan)
    fake.wait_result = lambda f, timeout: f
    return fake, seed, sent, validity


def test_home_request_uses_retreat_not_actual_state_and_six_arm_goals(config, capsys):
    fake, seed, sent, validity = home_fixture(config)
    assert VisionRobotSkills._home_precheck(fake, seed)
    request = sent[0].motion_plan_request
    assert request.start_state.is_diff is False
    assert list(request.start_state.joint_state.name[:6]) == list(NAMES)
    assert joint_values(request.start_state)['shoulder_pan_joint'] == 1.2  # Actual value was 0.
    assert joint_values(request.start_state)['wrist_3_joint'] == pytest.approx(4*math.pi+.2)
    assert joint_values(request.start_state)['wrist_1_joint'] == 2.7  # Bounded: don't modulo its winding.
    assert request.group_name == 'ur_manipulator' and request.allowed_planning_time == 15.
    assert request.num_planning_attempts == 3 and request.pipeline_id == request.planner_id == ''
    goals = request.goal_constraints[0].joint_constraints
    assert [j.joint_name for j in goals] == list(NAMES)
    assert all('finger' not in j.joint_name for j in goals)
    assert len(validity) == 3 and all(req.group_name == '' for req in validity)
    assert all(req.robot_state.is_diff is False for req in validity)
    output = capsys.readouterr().out
    for text in ('zone_b retreat endpoint', 'retreat state validity ..... PASS',
                 'home state validity ..... PASS', 'plan retreat -> home ....... PASS',
                 'home trajectory points ..... 41', 'fixture_pipeline', 'action goal status: N/A'):
        assert text in output


@pytest.mark.parametrize('failure,reason', [
    ('99999', 'HOME_PLANNING_FAILURE'), ('planning_timeout', 'HOME_PLANNING_TIMEOUT'),
    ('rpc_timeout', 'HOME_SERVICE_TIMEOUT'), ('retreat_collision', 'HOME_INVALID_STATE'),
    ('goal_collision', 'HOME_INVALID_STATE'), ('endpoint_collision', 'HOME_ENDPOINT_INVALID'),
    ('wrong_start', 'HOME_WRONG_TRAJECTORY_START'), ('wrong_first', 'TRAJECTORY_START_MISMATCH'),
    ('wrong_end', 'TRAJECTORY_ENDPOINT_MISMATCH'), ('wrong_group', 'HOME_INVALID_TRAJECTORY'),
    ('gripper_trajectory', 'HOME_INVALID_TRAJECTORY'), ('empty_path', 'HOME_INVALID_TRAJECTORY'),
    ('invalid_midpoint', 'JOINT_LIMIT_VIOLATION'),
])
def test_home_failures_are_not_success_and_have_details(config, failure, reason, capsys):
    fake, seed, sent, validity = home_fixture(config, failure)
    assert VisionRobotSkills._home_precheck(fake, seed) is False
    assert reason in fake.last_precheck_reason
    output = capsys.readouterr().out
    assert 'plan retreat -> home ....... PASS' not in output
    assert 'start joint state (simulated)' in output and 'home joint goal:' in output
    if 'collision' in failure:
        assert 'wrist_1_link <-> table' in output
    if failure == '99999':
        assert 'MoveItErrorCodes.val=99999 (FAILURE)' in output
        assert 'fixture planner detail' in output and 'fixture_pipeline' in output
    if failure == 'planning_timeout':
        assert 'HOME_SERVICE_TIMEOUT' not in output  # Keep planner and transport timeouts distinct.
    if failure in ('retreat_collision', 'goal_collision'):
        assert not sent and len(validity) == 2


def test_already_home_requires_all_joints_and_both_valid_states(config, capsys):
    close = {name: value+.005 for name, value in HOME.items()}
    fake, seed, sent, validity = home_fixture(config, start_values=close)
    assert VisionRobotSkills._home_precheck(fake, seed)
    assert not sent and len(validity) == 2
    assert 'ALREADY_AT_HOME' in capsys.readouterr().out
    fake, seed, sent, _ = home_fixture(config, 'goal_collision', close)
    assert VisionRobotSkills._home_precheck(fake, seed) is False
    assert not sent


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'nan', 'bounded_outside'])
def test_invalid_home_seed_never_reaches_planner(config, mutation):
    fake, seed, sent, _ = home_fixture(config)
    if mutation == 'missing':
        seed.joint_state.name.pop()
    elif mutation == 'duplicate':
        seed.joint_state.name[0] = seed.joint_state.name[1]
    elif mutation == 'nan':
        seed.joint_state.position[0] = math.nan
    else:
        i = seed.joint_state.name.index('elbow_joint')
        seed.joint_state.position[i] = 2*math.pi
    assert VisionRobotSkills._home_precheck(fake, seed) is False
    assert not sent


def test_joint_model_uses_live_bounds_and_group_chain():
    urdf, srdf = xml_model()
    limits = home_joint_model(urdf, srdf, 'ur_manipulator', NAMES,
        {'elbow_joint': {'has_position_limits': True, 'min_position': -1., 'max_position': 1.}})
    assert limits['elbow_joint']['lower'] == -1. and limits['elbow_joint']['upper'] == 1.
    assert limits['wrist_3_joint']['continuous'] is True
    bad_srdf = srdf.replace('<chain', '<joint name="gripper_joint"/><chain')
    with pytest.raises(EnvironmentError, match='HOME_MODEL_INVALID'):
        home_joint_model(urdf, bad_srdf, 'ur_manipulator', NAMES)
    with pytest.raises(EnvironmentError, match='incomplete MoveIt'):
        home_joint_model(urdf, srdf, 'ur_manipulator', NAMES,
                         {'elbow_joint': {'has_position_limits': True}})


def test_continuous_normalization_does_not_change_bounded_winding():
    values = dict(HOME, wrist_1_joint=3.9, wrist_3_joint=4*math.pi+.2)
    normalized = normalize_arm(values, NAMES, model(), 'fixture')
    assert normalized['wrist_1_joint'] == 3.9
    assert normalized['wrist_3_joint'] == pytest.approx(.2)
    assert joint_errors(dict(values, wrist_3_joint=.2), normalized, NAMES, model())['wrist_3_joint'] < 1e-9


def parameter_fixture(config, parameters):
    requests = []
    def params(req):
        requests.append(req)
        assert len(req.names) == 1  # One unset C++ parameter must not poison other values.
        assert set(req.names) <= parameters.keys()  # Never request an undeclared parameter.
        return SimpleNamespace(values=[parameters[name] for name in req.names])
    def types(req):
        assert set(req.names) <= parameters.keys()
        return SimpleNamespace(types=[parameters[name].type for name in req.names])
    def listed(req):
        assert req.depth == 0
        assert req.prefixes == ['robot_description', 'robot_description_semantic',
                                'robot_description_planning.joint_limits']
        return SimpleNamespace(result=SimpleNamespace(names=list(parameters)))
    fake = SimpleNamespace(execution_config=config, group_name='ur_manipulator', ARM_JOINT_NAMES=NAMES,
        home_parameter_client=SimpleNamespace(wait_for_service=lambda **kw: True, call_async=params),
        home_parameter_list_client=SimpleNamespace(wait_for_service=lambda **kw: True, call_async=listed),
        home_parameter_type_client=SimpleNamespace(wait_for_service=lambda **kw: True, call_async=types),
        wait_result=lambda f, timeout: f)
    return fake, requests


def test_home_model_parameter_rpc_reads_real_description_and_overrides(config):
    urdf, srdf = xml_model()
    parameters = {'robot_description': ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=urdf),
                  'robot_description_semantic': ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=srdf)}
    for name in NAMES:
        for field in ('has_position_limits', 'min_position', 'max_position'):
            parameters[f'robot_description_planning.joint_limits.{name}.{field}'] = ParameterValue()
    fake, requests = parameter_fixture(config, parameters)
    result = VisionRobotSkills._home_model(fake)
    assert result['elbow_joint']['upper'] == math.pi
    assert [request.names for request in requests] == [['robot_description'], ['robot_description_semantic']]


def test_home_model_ur_launch_urdf_topic_semantic_parameter(config, capsys):
    urdf, srdf = xml_model()
    fake, requests = parameter_fixture(config, {
        'robot_description_semantic': ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=srdf),
        # Real UR MoveIt config declares acceleration limits, not position overrides.
        'robot_description_planning.joint_limits.elbow_joint.has_acceleration_limits':
            ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=True)})
    topics = []
    fake._home_description_topic = lambda name: topics.append(name) or urdf
    result = VisionRobotSkills._home_model(fake)
    assert topics == ['robot_description']
    assert requests[0].names == ['robot_description_semantic']
    assert result['elbow_joint']['upper'] == math.pi
    assert 'NOT_DECLARED' in capsys.readouterr().out


def test_home_model_both_descriptions_from_topics(config):
    fake, requests = parameter_fixture(config, {})
    xml = dict(zip(('robot_description', 'robot_description_semantic'), xml_model()))
    fake._home_description_topic = lambda name: xml[name]
    assert VisionRobotSkills._home_model(fake) == model()
    assert requests == []


@pytest.mark.parametrize('parameter', [ParameterValue(),
    ParameterValue(type=ParameterType.PARAMETER_STRING, string_value='  ')])
def test_home_model_unset_or_empty_description_uses_topic(config, parameter):
    fake, _ = parameter_fixture(config, {'robot_description': parameter})
    xml = dict(zip(('robot_description', 'robot_description_semantic'), xml_model()))
    fake._home_description_topic = lambda name: xml[name]
    assert VisionRobotSkills._home_model(fake) == model()


def test_home_model_wrong_description_type_is_not_hidden(config):
    fake, _ = parameter_fixture(config, {
        'robot_description': ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=1)})
    with pytest.raises(EnvironmentError, match='expected STRING'):
        VisionRobotSkills._home_model(fake)


def test_home_model_live_override_with_topic_descriptions(config):
    prefix = 'robot_description_planning.joint_limits.elbow_joint.'
    fake, requests = parameter_fixture(config, {
        prefix+'has_position_limits': ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=True),
        prefix+'min_position': ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=-2.),
        prefix+'max_position': ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=2.)})
    xml = dict(zip(('robot_description', 'robot_description_semantic'), xml_model()))
    fake._home_description_topic = lambda name: xml[name]
    result = VisionRobotSkills._home_model(fake)
    assert len(requests) == 3 and all(len(request.names) == 1 for request in requests)
    assert result['elbow_joint']['lower'] == -2.
    assert result['elbow_joint']['upper'] == 2.


def test_home_model_parameter_response_length_failure_is_explicit(config):
    fake, _ = parameter_fixture(config, {
        'robot_description': ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=xml_model()[0])})
    fake.home_parameter_client.call_async = lambda req: SimpleNamespace(values=[])
    with pytest.raises(EnvironmentError, match='HOME_MODEL_PARAMETER_RESPONSE_INVALID'):
        VisionRobotSkills._home_model(fake)


def test_home_model_declared_uninitialized_regression_does_not_batch_get(config, capsys):
    urdf, srdf = xml_model()
    parameters = {'robot_description': ParameterValue(),
                  'robot_description_semantic': ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=srdf)}
    for name in NAMES:
        for field in ('has_position_limits', 'min_position', 'max_position'):
            parameters[f'robot_description_planning.joint_limits.{name}.{field}'] = ParameterValue()
    fake, requests = parameter_fixture(config, parameters)
    # Match Jazzy rclcpp: declared static NOT_SET makes the ENTIRE RPC empty.
    def get(req):
        requests.append(req)
        if any(parameters[n].type == ParameterType.PARAMETER_NOT_SET for n in req.names):
            return SimpleNamespace(values=[])
        return SimpleNamespace(values=[parameters[n] for n in req.names])
    fake.home_parameter_client.call_async = get
    topics = []
    fake._home_description_topic = lambda name: topics.append(name) or urdf
    assert VisionRobotSkills._home_model(fake) == model()
    assert [request.names for request in requests] == [['robot_description_semantic']]
    assert topics == ['robot_description']
    output = capsys.readouterr().out
    assert 'UNINITIALIZED' in output and 'position-limit parameters read: 0/18' in output


def test_home_model_parameter_types_bad_response_fails(config):
    fake, _ = parameter_fixture(config, {'robot_description': ParameterValue()})
    fake.home_parameter_type_client.call_async = lambda req: SimpleNamespace(types=[])
    with pytest.raises(EnvironmentError, match='HOME_MODEL_PARAMETER_TYPES_INVALID'):
        VisionRobotSkills._home_model(fake)


def test_home_model_race_to_uninitialized_is_confirmed_before_topic(config, capsys):
    urdf, srdf = xml_model()
    fake, _ = parameter_fixture(config, {
        'robot_description': ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=urdf)})
    probes = []
    def types(req):
        probes.append(req)
        return SimpleNamespace(types=[ParameterType.PARAMETER_STRING if len(probes) == 1
                                      else ParameterType.PARAMETER_NOT_SET])
    fake.home_parameter_type_client.call_async = types
    fake.home_parameter_client.call_async = lambda req: SimpleNamespace(values=[])
    fake._home_description_topic = lambda name: urdf if name == 'robot_description' else srdf
    assert VisionRobotSkills._home_model(fake) == model()
    assert len(probes) == 2 and probes[-1].names == ['robot_description']
    assert 'confirmed NOT_SET' in capsys.readouterr().out


def test_home_model_initialized_limit_empty_response_cannot_be_ignored(config):
    prefix = 'robot_description_planning.joint_limits.elbow_joint.'
    fake, _ = parameter_fixture(config, {
        prefix+'min_position': ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=-2.)})
    fake.home_parameter_client.call_async = lambda req: SimpleNamespace(values=[])
    with pytest.raises(EnvironmentError, match='HOME_MODEL_PARAMETER_RESPONSE_INVALID.*elbow_joint.min_position'):
        VisionRobotSkills._home_model(fake)


def test_home_model_parameter_type_change_cannot_be_ignored(config):
    fake, _ = parameter_fixture(config, {
        'robot_description': ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=xml_model()[0])})
    fake.home_parameter_client.call_async = lambda req: SimpleNamespace(values=[
        ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=1)])
    with pytest.raises(EnvironmentError, match='HOME_MODEL_PARAMETER_CHANGED'):
        VisionRobotSkills._home_model(fake)


@pytest.mark.parametrize('field,value,side', [('min_position', -2., 'lower'), ('max_position', 2., 'upper')])
def test_home_model_individual_position_override_without_flag(field, value, side):
    result = home_joint_model(*xml_model(), 'ur_manipulator', NAMES, {'elbow_joint': {field: value}})
    assert result['elbow_joint'][side] == value


@pytest.mark.parametrize('content', ['<robot name="live"/>', '', '   '])
def test_description_topic_latched_qos_and_subscription_cleanup(config, monkeypatch, content):
    import ur3_vision_planning.vision_robot_skills as module
    created, destroyed = [], []
    subscription = object()
    def subscribe(msg_type, topic, callback, qos):
        created.append((msg_type, topic, qos))
        callback(String(data=content))
        return subscription
    fake = SimpleNamespace(execution_config=config, create_subscription=subscribe,
                           destroy_subscription=destroyed.append)
    monkeypatch.setattr(module.rclpy, 'ok', lambda: False)
    if content.strip():
        assert VisionRobotSkills._home_description_topic(fake, 'robot_description') == content
    else:
        with pytest.raises(EnvironmentError, match='description timeout'):
            VisionRobotSkills._home_description_topic(fake, 'robot_description')
    msg_type, topic, qos = created[0]
    assert msg_type is String and topic == '/robot_description'
    assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert qos.reliability == ReliabilityPolicy.RELIABLE and qos.depth == 1
    assert destroyed == [subscription]


def test_home_model_invalid_topic_xml_is_rejected(config):
    fake, _ = parameter_fixture(config, {})
    fake._home_description_topic = lambda name: 'not XML'
    with pytest.raises(EnvironmentError, match='HOME_MODEL_INVALID'):
        VisionRobotSkills._home_model(fake)


def test_pipeline_home_uses_last_zone_retreat_endpoint(planner, state, config):
    plan = planner.resolve_goal('red_cube', 'zone_b', state)
    fake, _, _, _ = fake_precheck(planner, state, config)
    original_ik = fake._ik
    retreats = []
    def ik(xyz, seed, label):
        ok, result = original_ik(xyz, seed, label)
        if label == 'camera retreat':
            result = copy.deepcopy(result)
            result.joint_state.position[0] = .6 if not retreats else 1.2
            retreats.append(copy.deepcopy(result))
        return ok, result
    fake._ik = ik
    seen = []
    fake._home_precheck = lambda seed: seen.append(copy.deepcopy(seed)) or True
    VisionRobotSkills.feasibility_precheck(fake, plan, state, planner.free_temporary_candidates(state))
    assert len(retreats) == 2 and seen == [retreats[-1]]
    assert fake.home_start_source == 'zone_b retreat endpoint (camera retreat IK)'
    assert fake.joint_states[NAMES[0]][0] == 0.  # Actual robot state never changes.


def test_home_failure_still_restores_virtual_scene(planner, state, config, capsys):
    plan = planner.resolve_goal('red_cube', 'zone_b', state)
    fake, scene, baseline, _ = fake_precheck(planner, state, config)
    def failed_home(seed):
        fake.last_precheck_reason = 'HOME_PLANNING_FAILURE: error_code=99999, source=fixture'
        return False
    fake._home_precheck = failed_home
    with pytest.raises(EnvironmentError, match='HOME_PLANNING_FAILURE'):
        VisionRobotSkills.feasibility_precheck(fake, plan, state, planner.free_temporary_candidates(state))
    assert scene.world == baseline.world and scene.robot_state.attached_collision_objects == []
    assert 'PLANNING SCENE RESTORED' in capsys.readouterr().out

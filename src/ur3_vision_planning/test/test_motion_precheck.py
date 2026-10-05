"""Service fixtures only: no rclpy.init, Gazebo, actuator or motion action calls."""

import copy
from types import MethodType, SimpleNamespace
from builtin_interfaces.msg import Duration

import pytest
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.msg import (
    AttachedCollisionObject, CollisionObject, MoveItErrorCodes, PlanningScene,
    RobotState, RobotTrajectory,
)
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from test_environment_manager import planner, state
from test_plan_execution import config
from ur3_vision_planning.environment_manager import EnvironmentError
from ur3_vision_planning.plan_execution import execute_validated_plan, MotionExecutionError
from ur3_vision_planning.robot_skills import RobotSkills
from ur3_vision_planning.vision_robot_skills import VisionRobotSkills


def fake_precheck(planner, state, config, rejection=None, cartesian_ok=True):
    scene = PlanningScene()
    scene.world.collision_objects = [CollisionObject(id='table')]
    for name, obj in state['objects'].items():
        pose = Pose()
        for axis in ('x', 'y', 'z'):
            setattr(pose.position, axis, obj['position'][axis])
        pose.orientation.w = 1.
        scene.world.collision_objects.append(CollisionObject(id=name,
            primitives=[SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.04]*3)],
            primitive_poses=[pose]))
    scene.robot_state.is_diff = True
    baseline = copy.deepcopy(scene)
    events = []
    fake = SimpleNamespace(scene=planner.scene, attached_object=None,
        execution_config=config, ARM_JOINT_NAMES=RobotSkills.ARM_JOINT_NAMES,
        joint_states={n: (0., 0.) for n in RobotSkills.ARM_JOINT_NAMES},
        ee_link='tool0', frame_id='base_link', robot_base_height=0.,
        grasp_object_offset=.16, cube_height=.04, finger_reach=.15, approach_clearance=.01,
        zone_usable_size=[.075]*2, cube_size=[.04]*3, active_motion_handle=None,
        simulator_is_ready=lambda: True, get_tool0_pose=lambda: ((0., 0., .5), (1., 0., 0., 0.)),
        sync_camera_scene=lambda s: None, read_planning_scene=lambda: copy.deepcopy(scene),
        collision_free_approach_z=RobotSkills.collision_free_approach_z,
        bounded_place_offsets=lambda *a, **k: [0.],
        world_xyz=lambda p: (p['x'], p['y'], p['z']),
        last_precheck_reason='NO_IK_SOLUTION')
    for method in ('tool_targets', '_seed_gripper', '_allow_grasp_touch', '_reset_grasp_touch',
                   '_confirm_virtual_attachment', '_verify_scene_restored'):
        setattr(fake, method, MethodType(getattr(VisionRobotSkills, method), fake))
    def apply(diff):
        events.append('apply')
        scene.allowed_collision_matrix = copy.deepcopy(diff.allowed_collision_matrix)
        if diff.world.collision_objects:
            scene.world = copy.deepcopy(diff.world)
        if diff.robot_state.attached_collision_objects:
            scene.robot_state.attached_collision_objects = [copy.deepcopy(item)
                for item in diff.robot_state.attached_collision_objects
                if item.object.operation != CollisionObject.REMOVE]
    fake._apply_scene = apply
    def remove(name):
        scene.world.collision_objects = [item for item in scene.world.collision_objects if item.id != name]
        return True
    fake.remove_world_object = remove
    def attach(name, attached, world_position=None):
        events.append(('attach' if attached else 'detach', name))
        remove(name)
        if attached:
            item = AttachedCollisionObject(link_name='tool0', touch_links=[
                'simple_gripper_base_link', 'simple_gripper_left_finger_link', 'simple_gripper_right_finger_link'])
            item.object.id = name
            scene.robot_state.attached_collision_objects = [item]
        else:
            scene.robot_state.attached_collision_objects = []
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = world_position
            pose.orientation.w = 1.
            scene.world.collision_objects.append(CollisionObject(id=name,
                primitives=[SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.04]*3)], primitive_poses=[pose]))
        return True
    fake.update_attached_object = attach
    def ik(xyz, seed, label):
        events.append(label)
        if 'temporary_' in label:
            assert [o.object.id for o in scene.robot_state.attached_collision_objects] == ['blue_cube']
            assert 'blue_cube' not in [o.id for o in scene.world.collision_objects]
        passed = not (rejection and rejection(label))
        fake.last_precheck_reason = 'SUCCESS' if passed else 'NO_IK_SOLUTION'
        return passed, seed
    fake._ik = ik
    def cartesian(start, target, seed, label):
        events.append(label)
        assert start != target and start[:2] == target[:2]
        assert start[2] == pytest.approx(.4 if 'lowering' in label else .38)
        assert target[2] == pytest.approx(.38 if 'lowering' in label else .4)
        if 'lowering' in label:
            obj = label.split()[0]
            matrix = scene.allowed_collision_matrix
            a, b = matrix.entry_names.index(obj), matrix.entry_names.index('simple_gripper_left_finger_link')
            assert matrix.entry_values[a].enabled[b] is True
            assert obj in [o.id for o in scene.world.collision_objects]
        fake.last_precheck_reason = 'CARTESIAN_INCOMPLETE'
        return cartesian_ok, seed
    fake._cartesian_precheck = cartesian
    fake._home_precheck = lambda seed: events.append('home plan-only') or True
    return fake, scene, baseline, events


def test_candidate_rejection_then_selection_and_revalidation(planner, state, config, capsys):
    goal = {'object': 'red_cube', 'target_zone': 'zone_b'}
    plan = planner.resolve_goal(**{'object_name': 'red_cube', 'target_zone': 'zone_b', 'environment_state': state})
    skills, scene, baseline, events = fake_precheck(planner, state, config,
        rejection=lambda label: label.startswith('temporary_1 '))
    candidates = planner.free_temporary_candidates(state)
    resolved = VisionRobotSkills.feasibility_precheck(skills, plan, state, candidates)
    assert plan['plan'][1]['destination'] == 'temporary_2'
    assert plan['temporary_positions'] == {'temporary_2': candidates['temporary_2']}
    assert resolved['temporary_2'] == candidates['temporary_2']
    assert not any('temporary_3' in str(e) for e in events)
    assert not any('red_cube grasp' == e for e in events)  # no independent grasp IK
    assert 'red_cube Cartesian lowering' in events
    planner.adopt_prechecked_plan(plan, goal, state, candidates)
    assert planner.validate_plan(plan, goal, state)[0]
    assert scene.world == baseline.world
    assert scene.robot_state.attached_collision_objects == []
    output = capsys.readouterr().out
    assert 'temporary_1 ... REJECTED: pre-place: NO_IK_SOLUTION' in output
    assert 'SELECTED: temporary_2' in output and 'PLANNING SCENE RESTORED' in output


@pytest.mark.parametrize('failure', ['all_candidates', 'cartesian', 'restore'])
def test_precheck_failure_restores_scene_without_gazebo_or_motion(planner, state, config, failure, capsys):
    plan = planner.resolve_goal('red_cube', 'zone_b', state)
    skills, scene, baseline, events = fake_precheck(planner, state, config,
        rejection=(lambda label: label.startswith('temporary_')) if failure == 'all_candidates' else None,
        cartesian_ok=failure != 'cartesian')
    if failure == 'restore':
        skills._verify_scene_restored = lambda s: (_ for _ in ()).throw(EnvironmentError('bad restore'))
    with pytest.raises(EnvironmentError, match='PRECHECK_SCENE_RESTORE_FAILED' if failure == 'restore' else 'MOTION_PRECHECK_FAILED'):
        VisionRobotSkills.feasibility_precheck(skills, plan, state, planner.free_temporary_candidates(state))
    assert scene.world == baseline.world
    assert scene.robot_state.attached_collision_objects == []
    if failure != 'restore':
        assert 'home plan-only' not in events
    output = capsys.readouterr().out
    assert ('PLANNING SCENE RESTORED' in output) == (failure != 'restore')
    if failure == 'all_candidates':
        assert 'temporary_12 ... REJECTED' in output


@pytest.mark.parametrize('fraction,code,valid_fk,passed', [
    (1., MoveItErrorCodes.SUCCESS, True, True),
    (.95, MoveItErrorCodes.SUCCESS, True, False),
    (.1, MoveItErrorCodes.SUCCESS, True, False),
    (1., MoveItErrorCodes.NO_IK_SOLUTION, True, False),
    (1., MoveItErrorCodes.SUCCESS, False, False),
    (float('nan'), MoveItErrorCodes.SUCCESS, True, False),
])
def test_cartesian_is_seeded_collision_checked_and_complete(config, fraction, code, valid_fk, passed):
    requests = []
    seed = RobotState()
    names = list(RobotSkills.ARM_JOINT_NAMES)
    seed.joint_state.name, seed.joint_state.position = names, [.25]+[0.]*5
    trajectory = RobotTrajectory(joint_trajectory=JointTrajectory(joint_names=names,
        points=[JointTrajectoryPoint(positions=[.25]+[0.]*5),
                JointTrajectoryPoint(positions=[.26]+[0.]*5, time_from_start=Duration(sec=1))]))
    fk_calls = []
    def fk(s, xyz, label):
        fk_calls.append((copy.deepcopy(s), xyz))
        return valid_fk
    fake = SimpleNamespace(frame_id='base_link', group_name='ur_manipulator', ee_link='tool0',
        robot_base_height=0., execution_config=config, motion_scaling_factor=.2,
        down_orientation=Quaternion(x=1., w=0.), _fk_matches=fk, last_precheck_reason='FK_POSE_MISMATCH',
        cartesian_client=SimpleNamespace(call_async=lambda req: requests.append(req) or SimpleNamespace(
            fraction=fraction, error_code=MoveItErrorCodes(val=code), solution=trajectory, start_state=copy.deepcopy(seed))),
        wait_result=lambda f, timeout: f)
    fake.ARM_JOINT_NAMES = RobotSkills.ARM_JOINT_NAMES
    fake._branch_model = lambda: {n: {'type': 'revolute', 'lower': -7., 'upper': 7.} for n in names}
    fake._branch_validity = lambda state, label: True
    fake._guard_trajectory = MethodType(VisionRobotSkills._guard_trajectory, fake)
    result, end = VisionRobotSkills._cartesian_precheck(fake, (.38, -.15, .4), (.38, -.15, .38), seed, 'red_cube lowering')
    assert result is passed
    if not valid_fk:
        assert not requests  # wrong start seed cannot be used
        return
    request = requests[0]
    assert request.avoid_collisions is True and list(request.start_state.joint_state.position) == [.25]+[0.]*5
    assert request.start_state.attached_collision_objects == []
    assert request.waypoints[0].position.z == .38
    assert request.waypoints[0].orientation == Quaternion(x=1., w=0.)
    assert request.header.frame_id == 'base_link' and request.link_name == 'tool0'
    if passed:
        assert list(end.joint_state.position) == [.26]+[0.]*5 and len(fk_calls) == 3


@pytest.mark.parametrize('fail', [False, True])
def test_precheck_only_never_sends_actuator_commands(planner, state, config, fail, capsys):
    plan = planner.resolve_goal('red_cube', 'zone_b', state)
    original = copy.deepcopy(plan)
    goal = {'object': 'red_cube', 'target_zone': 'zone_b'}
    events = []
    def precheck(working, observed, temporary_candidates, on_temporary_selected=None):
        assert temporary_candidates
        if fail:
            raise EnvironmentError('MOTION_PRECHECK_FAILED')
        working['plan'][1]['destination'] = 'temporary_2'
        working['temporary_positions'] = {'temporary_2': temporary_candidates['temporary_2']}
        return {'temporary_2': temporary_candidates['temporary_2']}
    forbidden = lambda *a, **k: pytest.fail('precheck-only cannot move or send gripper commands')
    skills = SimpleNamespace(feasibility_precheck=precheck, attached_object=None,
        cancel_pending_precheck=lambda: events.append('cancel plan-only'),
        stop_pending_motion=forbidden, pick_from_camera=forbidden, home=forbidden,
        place_destination=forbidden, sync_camera_scene=forbidden,
        destroy_node=lambda: events.append('destroy'))
    manager = SimpleNamespace(planner=planner, fresh_snapshot=forbidden,
        get_parameter=lambda n: SimpleNamespace(value={
            'dry_run': False, 'execute_robot': True, 'precheck_only': True}[n]))
    if fail:
        with pytest.raises(MotionExecutionError):
            execute_validated_plan(manager, goal, state, plan, config, lambda *a: skills)
        assert events == ['cancel plan-only', 'destroy']
        assert planner.selected_temporary_positions == original['temporary_positions']
    else:
        assert execute_validated_plan(manager, goal, state, plan, config, lambda *a: skills) is False
        assert planner.selected_temporary_positions == {'temporary_2': planner.config['temporary_positions']['temporary_2']}
        assert 'PRECHECK ONLY — ROBOT MOTION NOT STARTED' in capsys.readouterr().out
    assert plan == original


def test_unknown_precheck_slot_cannot_enter_trusted_ledger(planner, state):
    plan = planner.resolve_goal('red_cube', 'zone_b', state)
    original = copy.deepcopy(planner.selected_temporary_positions)
    plan['temporary_positions']['temporary_1']['x'] += .001
    with pytest.raises(EnvironmentError, match='UNTRUSTED_COORDINATES'):
        planner.adopt_prechecked_plan(plan, {'object': 'red_cube', 'target_zone': 'zone_b'}, state,
                                      planner.free_temporary_candidates(state))
    assert planner.selected_temporary_positions == original


def test_virtual_open_seed_matches_asymmetric_controller(config):
    fake = SimpleNamespace(execution_config=config)
    seed = RobotState()
    updated = VisionRobotSkills._seed_gripper(fake, seed, 'open')
    assert dict(zip(updated.joint_state.name, updated.joint_state.position)) == {
        'simple_gripper_left_finger_joint': 0., 'simple_gripper_right_finger_joint': .043}
    assert seed.joint_state.name == []


@pytest.mark.parametrize('case,passed', [('good', True), ('negative_q', True),
    ('wrong_frame', False), ('wrong_link', False), ('wrong_z', False),
    ('wrong_orientation', False), ('invalid_q', False), ('error', False)])
def test_fk_verifies_start_and_end_pose(config, case, passed):
    from geometry_msgs.msg import PoseStamped
    point = PoseStamped()
    point.header.frame_id = 'base_link'
    point.pose.position.x, point.pose.position.y, point.pose.position.z = .38, -.15, .38
    point.pose.orientation = Quaternion(x=-1. if case == 'negative_q' else 1., w=0.)
    if case == 'wrong_frame':
        point.header.frame_id = 'world'
    if case == 'wrong_z':
        point.pose.position.z += .02
    if case == 'wrong_orientation':
        point.pose.orientation = Quaternion(w=1.)
    if case == 'invalid_q':
        point.pose.orientation = Quaternion(w=0.)
    requests = []
    result = SimpleNamespace(pose_stamped=[point],
        fk_link_names=['wrist_3_link' if case == 'wrong_link' else 'tool0'],
        error_code=MoveItErrorCodes(val=MoveItErrorCodes.FAILURE if case == 'error' else MoveItErrorCodes.SUCCESS))
    fake = SimpleNamespace(frame_id='base_link', ee_link='tool0', robot_base_height=0.,
        execution_config=config, down_orientation=Quaternion(x=1., w=0.),
        fk_client=SimpleNamespace(call_async=lambda req: requests.append(req) or result),
        wait_result=lambda f, timeout: f)
    assert VisionRobotSkills._fk_matches(fake, RobotState(), (.38, -.15, .38), 'test') is passed
    assert requests[0].header.frame_id == 'base_link'


@pytest.mark.parametrize('case', ['moved_object', 'left_attached', 'left_touch'])
def test_restore_verification_rejects_stale_virtual_scene(planner, state, config, case):
    skills, scene, baseline, _ = fake_precheck(planner, state, config)
    if case == 'moved_object':
        scene.world.collision_objects[1].primitive_poses[0].position.x += .01
    elif case == 'left_attached':
        scene.robot_state.attached_collision_objects = [AttachedCollisionObject()]
    else:
        skills._allow_grasp_touch(baseline, 'blue_cube')
    with pytest.raises(EnvironmentError, match='RESTORE_'):
        skills._verify_scene_restored(baseline)


def test_unconfirmed_virtual_attach_is_rejected(planner, state, config):
    skills, scene, _, _ = fake_precheck(planner, state, config)
    with pytest.raises(EnvironmentError, match='VIRTUAL_ATTACH_NOT_CONFIRMED'):
        skills._confirm_virtual_attachment('blue_cube')


def test_no_physical_commands_in_precheck_source():
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(VisionRobotSkills.feasibility_precheck)))
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not calls.intersection({'set_gazebo_attachment', 'load_gazebo_attachment_plugin',
        'command_gripper', 'move_cartesian', 'move_to_pose', 'home', 'publish'})

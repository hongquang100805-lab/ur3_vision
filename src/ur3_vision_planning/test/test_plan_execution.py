"""No ROS initialization, servers, simulator or physical commands in these tests."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from moveit_msgs.msg import CollisionObject, MoveItErrorCodes, PlanningScene, PlanningSceneWorld, RobotState

from test_environment_manager import planner, state
from ur3_vision_planning.environment_manager import EnvironmentError
from ur3_vision_planning.plan_execution import (
    CameraPlacementConfirmation, MotionExecutionError, execution_enabled, execute_validated_plan,
    validate_execution_config,
)
from ur3_vision_planning.vision_robot_skills import VisionRobotSkills, destination_center
from ur3_vision_planning.robot_skills import RobotSkills


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / 'config' / 'execution.yaml'
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize('dry,confirmed,enabled', [(True, False, False), (True, True, False), (False, True, True)])
def test_double_gate(dry, confirmed, enabled):
    assert execution_enabled(dry, confirmed) is enabled


def test_unconfirmed_execution_is_refused():
    with pytest.raises(EnvironmentError, match='EXECUTION_NOT_CONFIRMED'):
        execution_enabled(False, False)


@pytest.mark.parametrize('dry,confirmed', [(1, True), (False, 'true'), (None, False)])
def test_non_boolean_gate_cannot_enable_motion(dry, confirmed):
    with pytest.raises(EnvironmentError):
        execution_enabled(dry, confirmed)


def placed_state(state, name, destination, point, stamp):
    state = copy.deepcopy(state)
    state['stamp'] = stamp
    state['published_at'] = stamp+.1
    for obj in state['objects'].values():
        obj['stamp'] = stamp
    obj = state['objects'][name]
    obj['position'] = {a: point[a] for a in ('x', 'y', 'z')}
    obj['zone'] = destination if destination in state['zones'] else None
    for zone_name, zone in state['zones'].items():
        occupants = [n for n, o in state['objects'].items() if o['zone'] == zone_name]
        zone.update(objects=occupants, object=occupants[0] if occupants else None,
                    occupied=bool(occupants), status='occupied' if occupants else 'empty')
    return state


def test_confirmation_requires_three_distinct_post_retreat_frames(planner, state, config):
    point = planner.config['temporary_positions']['temporary_1']
    check = CameraPlacementConfirmation('blue_cube', 'temporary_1', point, ['zone_b'], config['verification'], 10.)
    observed = placed_state(state, 'blue_cube', 'temporary_1', point, 11.)
    assert not check.observe(observed)
    assert not check.observe(observed)  # same image published repeatedly is only one frame
    assert check.count == 1
    observed['objects']['blue_cube']['stamp'] = 12.
    assert not check.observe(observed)
    observed['objects']['blue_cube']['stamp'] = 13.
    assert check.observe(observed)


@pytest.mark.parametrize('failure', ['stale', 'far', 'zone_not_empty', 'wrong_zone', 'old_frame'])
def test_bad_camera_frame_does_not_verify(planner, state, config, failure):
    point = planner.config['temporary_positions']['temporary_1']
    check = CameraPlacementConfirmation('blue_cube', 'temporary_1', point, ['zone_b'], config['verification'], 10.)
    observed = placed_state(state, 'blue_cube', 'temporary_1', point, 11.)
    if failure == 'stale':
        observed['stale'] = True
    elif failure == 'far':
        observed['objects']['blue_cube']['position']['x'] += .1
    elif failure == 'zone_not_empty':
        observed['zones']['zone_b']['occupied'] = True
    elif failure == 'wrong_zone':
        observed['objects']['blue_cube']['zone'] = 'zone_a'
    else:
        observed['objects']['blue_cube']['stamp'] = 9.
    for _ in range(5):
        assert not check.observe(observed)


def test_red_confirmation_requires_both_object_zone_and_zone_occupant(planner, state, config):
    point = destination_center(planner.scene, {'temporary_positions': {}}, 'zone_b')
    check = CameraPlacementConfirmation('red_cube', 'zone_b', point, [], config['verification'], 10.)
    state = placed_state(state, 'blue_cube', 'temporary_1', planner.config['temporary_positions']['temporary_1'], 11.)
    state = placed_state(state, 'red_cube', 'zone_b', point, 12.)
    state['zones']['zone_b']['object'] = 'blue_cube'
    assert not check.observe(state)
    state['zones']['zone_b']['object'] = 'red_cube'
    for stamp in (13., 14., 15.):
        state['objects']['red_cube']['stamp'] = stamp
        result = check.observe(state)
    assert result


def test_zone_and_temporary_z_are_cube_centres(planner, config):
    plan = {'temporary_positions': planner.config['temporary_positions']}
    assert destination_center(planner.scene, plan, 'zone_b')['z'] == pytest.approx(.22)
    point = destination_center(planner.scene, plan, 'temporary_1')
    fake = SimpleNamespace(robot_base_height=0., grasp_object_offset=.16, execution_config=config,
                           cube_height=.04, finger_reach=.15, approach_clearance=.01,
                           world_xyz=lambda p: (p['x'], p['y'], p['z']),
                           collision_free_approach_z=RobotSkills.collision_free_approach_z)
    targets = VisionRobotSkills.tool_targets(fake, point, place=True)
    assert targets[0][1][2] == pytest.approx(.400)
    assert targets[1][1][2] == pytest.approx(.385)
    assert VisionRobotSkills.tool_targets(fake, point)[1][1][2] == pytest.approx(.38)


@pytest.mark.parametrize('failure', [None, 'precheck', 'pick', 'place', 'verification', 'home'])
def test_execution_order_and_stop_on_first_failure(planner, state, config, monkeypatch, capsys, failure):
    goal = {'object': 'red_cube', 'target_zone': 'zone_b'}
    plan = planner.resolve_goal('red_cube', 'zone_b', state)
    points = {d: destination_center(planner.scene, plan, d) for d in ('temporary_1', 'zone_b')}
    blue_placed = placed_state(state, 'blue_cube', 'temporary_1', points['temporary_1'], 11.)
    snapshots = iter((state, state, blue_placed))
    events = []
    manager = SimpleNamespace(planner=planner,
        get_parameter=lambda n: SimpleNamespace(value={'dry_run': False, 'execute_robot': True, 'precheck_only': False}[n]),
        fresh_snapshot=lambda: copy.deepcopy(next(snapshots)))
    class FakeSkills:
        attached_object = None
        def feasibility_precheck(self, plan, observed, temporary_candidates=None, on_temporary_selected=None):
            events.append('precheck')
            if failure == 'precheck':
                raise EnvironmentError('MOTION_PRECHECK_FAILED')
            return points
        def sync_camera_scene(self, observed):
            events.append('sync')
        def pick_from_camera(self, name, observed):
            events.append(f'pick {name}')
            self.attached_object = name
            return 'GRASP_FAILED' if failure == 'pick' else 'SUCCESS'
        def place_destination(self, name, destination, point):
            events.append(f'place {name} {destination}')
            if failure == 'place':
                return 'PLANNING_FAILED'
            self.attached_object = None
            return 'SUCCESS'
        def camera_retreat(self):
            events.append('retreat')
            return 'SUCCESS'
        def home(self):
            events.append('home')
            return 'EXECUTION_FAILED' if failure == 'home' else 'SUCCESS'
        def destroy_node(self):
            events.append('destroy')
    skills = FakeSkills()
    def verify(manager, skills, name, destination, point, vacated, cfg):
        events.append(f'verify {name}')
        if failure == 'verification':
            raise MotionExecutionError('CAMERA_VERIFICATION_FAILED')
        if name == 'blue_cube':
            assert vacated == ['zone_b']
    monkeypatch.setattr('ur3_vision_planning.plan_execution.wait_camera_confirmation', verify)
    if failure:
        with pytest.raises(MotionExecutionError):
            execute_validated_plan(manager, goal, state, plan, config, lambda m, c: skills)
        assert 'TASK SUCCESS' not in capsys.readouterr().out
        if failure not in ('home',):
            assert 'pick red_cube' not in events and 'home' not in events
        if failure in ('pick', 'place'):
            assert skills.attached_object == 'blue_cube'
    else:
        assert execute_validated_plan(manager, goal, state, plan, config, lambda m, c: skills)
        assert events == ['precheck', 'sync', 'sync', 'pick blue_cube', 'place blue_cube temporary_1',
                          'retreat', 'verify blue_cube', 'sync', 'pick red_cube',
                          'place red_cube zone_b', 'retreat', 'verify red_cube', 'home', 'destroy']
        assert 'TASK SUCCESS' in capsys.readouterr().out


@pytest.mark.parametrize('dry,confirmed', [(True, False), (True, True), (False, False)])
def test_no_skills_factory_without_double_confirmation(dry, confirmed, config):
    manager = SimpleNamespace(get_parameter=lambda n: SimpleNamespace(value={'dry_run': dry, 'execute_robot': confirmed}[n]))
    def forbidden(*args):
        pytest.fail('RobotSkills must never be constructed without the two flags')
    if dry:
        assert execute_validated_plan(manager, {}, {}, {}, config, forbidden) is False
    else:
        with pytest.raises(EnvironmentError):
            execute_validated_plan(manager, {}, {}, {}, config, forbidden)


def test_home_precheck_sets_plan_only_not_execute():
    import inspect
    source = inspect.getsource(VisionRobotSkills._home_plan_candidate)
    assert 'GetMotionPlan.Request()' in source  # Planning-only service, not a motion action.
    assert 'motion.start_state = start' in source
    assert 'send_goal_async' not in source and 'execute_trajectory' not in source


@pytest.mark.parametrize('mutation', [
    lambda c: c['verification'].update(minimum_frames=1),
    lambda c: c.update(place_release_clearance_m=-.1),
    lambda c: c.update(camera_retreat=None),
])
def test_unsafe_config_rejected_before_factory(config, mutation):
    mutation(config)
    with pytest.raises(EnvironmentError, match='INVALID_EXECUTION_CONFIG'):
        validate_execution_config(config)


def logger():
    return SimpleNamespace(info=lambda *a: None, warning=lambda *a: None, error=lambda *a: None)


def test_pick_uses_supplied_camera_position_not_spawn_or_gazebo(monkeypatch):
    fake = SimpleNamespace(simulator_is_ready=lambda: True, attached_object=None,
        scene={'objects': {'green_cube': {'position': [999, 999, 999]}}},
        get_logger=logger, command_gripper=lambda *a: False,
        get_gazebo_model_pose=lambda *a: pytest.fail('Gazebo must not select the pick target'))
    assert RobotSkills.pick(fake, 'green_cube', object_position=(.4, -.1, .22)) == 'EXECUTION_FAILED'


def test_lift_failure_keeps_physical_and_moveit_attachment():
    paths = iter(('SUCCESS', 'PLANNING_FAILED'))
    fake = SimpleNamespace(simulator_is_ready=lambda: True, attached_object=None,
        scene={'objects': {'purple_cube': {}}}, preserve_failed_grasp=True,
        get_logger=logger, command_gripper=lambda *a, **k: True,
        grasp_object_offset=.16, cube_height=.04, finger_reach=.15, approach_clearance=.01,
        collision_free_approach_z=RobotSkills.collision_free_approach_z,
        move_to_pose=lambda *a: 'SUCCESS', remove_world_object=lambda n: True,
        move_cartesian=lambda *a: next(paths),
        get_tool0_position=lambda: (.4, -.1, .38),
        get_gazebo_model_pose=lambda n: (.4, -.1, .22),
        load_gazebo_attachment_plugin=lambda n: True,
        set_gazebo_attachment=lambda *a, **k: True,
        update_attached_object=lambda *a, **k: True,
        restore_grasp_object=lambda *a: pytest.fail('Failed lift must not detach/teleport the object'))
    assert RobotSkills.pick(fake, 'purple_cube', (.4, -.1, .22)) == 'PLANNING_FAILED'
    assert fake.attached_object == 'purple_cube' and fake.gazebo_cube_attached is True


@pytest.mark.parametrize('reachable', [False, True])
def test_place_cannot_release_before_verified_arrival(reachable):
    fake = SimpleNamespace(attached_object='green_cube', grasp_object_offset=.16,
        cube_height=.04, robot_base_height=0., frame_id='base_link', ee_link='tool0',
        scene={'table': {'center': [.38, 0., .185], 'size': [.3, .65, .03]}},
        zone_marker_size=[.09]*2, zone_usable_size=[.075]*2, cube_size=[.04]*3, finger_reach=.15,
        get_logger=logger, get_tool0_pose=lambda: ((.4, -.1, .4), (1., 0., 0., 0.)),
        diagnose_place_pose=lambda *a: reachable, move_cartesian=lambda *a, **k: 'SUCCESS',
        verify_tool_pose=lambda *a: False,
        command_gripper=lambda *a: pytest.fail('Must not open before verified arrival'),
        set_gazebo_attachment=lambda *a, **k: pytest.fail('Must not detach before verified arrival'))
    result = RobotSkills.place_at(fake, 'green_cube', (.43, -.23, .22), 'temporary_1')
    assert result == ('PLACE_POSE_NOT_REACHED' if reachable else 'PLANNING_FAILED')
    assert fake.attached_object == 'green_cube'


def test_collision_aware_ik_is_mandatory(config):
    sent = []
    from geometry_msgs.msg import Quaternion
    fake = SimpleNamespace(group_name='ur_manipulator', ee_link='tool0', frame_id='base_link',
        robot_base_height=0., execution_config=config, down_orientation=Quaternion(x=1.),
        moveit_error_name=RobotSkills.moveit_error_name,
        ik_client=SimpleNamespace(call_async=lambda req: sent.append(req) or SimpleNamespace(
            error_code=MoveItErrorCodes(val=MoveItErrorCodes.NO_IK_SOLUTION), solution=RobotState())),
        wait_result=lambda f, timeout: f)
    fake.ARM_JOINT_NAMES = RobotSkills.ARM_JOINT_NAMES
    fake._branch_model = lambda: {n: {'type': 'revolute', 'lower': -7., 'upper': 7.} for n in fake.ARM_JOINT_NAMES}
    seed = RobotState()
    seed.joint_state.name, seed.joint_state.position = list(fake.ARM_JOINT_NAMES), [0.]*6
    assert VisionRobotSkills._ik(fake, (.43, -.23, .4), seed, 'temporary_1 pre-place')[0] is False
    assert sent[0].ik_request.avoid_collisions is True


def test_failure_stop_cancels_action_without_open_detach_or_home(config):
    events = []
    handle = SimpleNamespace(cancel_goal_async=lambda: events.append('cancel') or 'cancel_future',
                             get_result_async=lambda: events.append('terminal_result') or 'result_future')
    fake = SimpleNamespace(active_motion_handle=handle, execution_config=config,
        gripper_command_publisher=SimpleNamespace(publish=lambda msg: events.append(tuple(msg.data))),
        wait_result=lambda future, timeout: events.append(future))
    VisionRobotSkills.stop_pending_motion(fake)
    assert events == [(0., 0.), 'cancel', 'cancel_future', 'terminal_result', 'result_future']
    assert fake.active_motion_handle is None


def test_collision_matrix_extension_preserves_robot_pairs_and_clears_new_touch():
    from moveit_msgs.msg import AllowedCollisionEntry, AllowedCollisionMatrix
    from ur3_vision_planning.vision_robot_skills import extend_collision_matrix
    original = AllowedCollisionMatrix(entry_names=['wrist_3_link', 'simple_gripper_base_link'],
        entry_values=[AllowedCollisionEntry(enabled=[False, True]),
                      AllowedCollisionEntry(enabled=[True, False])])
    matrix = extend_collision_matrix(original, ['blue_cube'])
    assert matrix.entry_values[0].enabled[1] is True
    assert list(matrix.entry_values[2].enabled) == [False, False, False]
    assert list(original.entry_names) == ['wrist_3_link', 'simple_gripper_base_link']


def test_cartesian_timeout_cannot_be_treated_as_safe_fallback(monkeypatch):
    monkeypatch.setattr(RobotSkills, 'move_cartesian', lambda *a, **k: 'PLANNING_FAILED')
    fake = object.__new__(VisionRobotSkills)
    fake.active_motion_handle = object()  # accepted action still lacks terminal result
    assert fake.move_cartesian([]) == 'EXECUTION_STATE_UNCERTAIN'
    fake.active_motion_handle = None  # fraction<1 without execution may use collision-aware fallback
    assert fake.move_cartesian([]) == 'PLANNING_FAILED'


@pytest.mark.parametrize('failure', [False, True])
def test_precheck_restores_virtual_scene_on_success_and_failure(planner, state, config, failure):
    plan = planner.resolve_goal('red_cube', 'zone_b', state)
    diffs = []
    checks = []
    fake = SimpleNamespace(
        scene=planner.scene, attached_object=None, execution_config=config,
        ARM_JOINT_NAMES=RobotSkills.ARM_JOINT_NAMES,
        joint_states={n: (0., 0.) for n in RobotSkills.ARM_JOINT_NAMES}, ee_link='tool0',
        simulator_is_ready=lambda: True, get_tool0_pose=lambda: ((0., 0., .5), (1., 0., 0., 0.)),
        sync_camera_scene=lambda s: None,
        read_planning_scene=lambda: PlanningScene(world=PlanningSceneWorld(
            collision_objects=[CollisionObject(id='table')])),
        tool_targets=lambda p, place=False: [('pre-place' if place else 'pre-grasp', (p['x'], p['y'], .4)),
                                           ('final-place' if place else 'grasp', (p['x'], p['y'], .385))],
        zone_usable_size=[.075, .075], cube_size=[.04]*3,
        bounded_place_offsets=lambda *a, **k: [0.],
        _allow_grasp_touch=lambda *a: None,
        remove_world_object=lambda n: True,
        update_attached_object=lambda *a, **k: True,
        _home_precheck=lambda seed: True,
        _apply_scene=lambda s: diffs.append(s),
        world_xyz=lambda p: (p['x'], p['y'], p['z']))
    fake.robot_base_height, fake.grasp_object_offset, fake.finger_reach, fake.cube_height = 0., .16, .15, .04
    fake._cartesian_precheck = lambda start, target, seed, label: (True, seed)
    fake._seed_gripper = lambda seed, mode: seed
    fake._confirm_virtual_attachment = lambda name: None
    fake._reset_grasp_touch = lambda baseline: None
    fake._verify_scene_restored = lambda baseline: None
    fake.last_precheck_reason = 'NO_IK_SOLUTION'
    fake.tool_targets = lambda p, place=False: (
        [('pre-place', (p['x'], p['y'], .4)), ('final-place', (p['x'], p['y'], .385))] if place else
        [('pre-grasp', (p['x'], p['y'], .4)), ('grasp', (p['x'], p['y'], .38)), ('lift', (p['x'], p['y'], .4))])
    def ik(xyz, seed, label):
        checks.append(label)
        return not failure, seed
    fake._ik = ik
    if failure:
        with pytest.raises(EnvironmentError, match='MOTION_PRECHECK_FAILED'):
            VisionRobotSkills.feasibility_precheck(fake, plan, state)
    else:
        assert VisionRobotSkills.feasibility_precheck(fake, plan, state)
    if not failure:
        assert any('temporary_1 final-place' in c for c in checks)
        assert any('zone_b pre-place' in c for c in checks)
    assert len(diffs) == 1
    assert diffs[0].robot_state.joint_state.name == []
    assert len(diffs[0].robot_state.attached_collision_objects) == 5

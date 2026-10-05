"""Offline service/geometry tests, not simulator collision measurements."""

import copy
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from geometry_msgs.msg import Quaternion
from moveit_msgs.msg import ContactInformation, MoveItErrorCodes, RobotState, RobotTrajectory

from test_environment_manager import planner, state
from test_motion_precheck import fake_precheck
from test_plan_execution import config
from ur3_vision_planning.environment_manager import EnvironmentError
from ur3_vision_planning.plan_execution import execute_validated_plan, MotionExecutionError
from ur3_vision_planning.scene_config import load_scene
from ur3_vision_planning.vision_robot_skills import VisionRobotSkills
from ur3_vision_planning.vision_robot_skills import extend_collision_matrix


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('body', ['green_cube', 'table', 'red_cube', None, 'ik_error'])
def test_waypoint_diagnostics_show_real_service_contacts(config, body, capsys):
    ik_requests, validity_requests, fk_points = [], [], []
    seed = RobotState()
    seed.joint_state.name, seed.joint_state.position = ['shoulder_pan_joint'], [.3]
    def ik(req):
        ik_requests.append(req)
        return SimpleNamespace(solution=copy.deepcopy(seed), error_code=MoveItErrorCodes(
            val=MoveItErrorCodes.NO_IK_SOLUTION if body == 'ik_error' else MoveItErrorCodes.SUCCESS))
    def validity(req):
        validity_requests.append(req)
        contacts = [] if body in (None, 'ik_error') else [ContactInformation(
            contact_body_1='simple_gripper_left_finger_link', body_type_1=ContactInformation.ROBOT_LINK,
            contact_body_2=body, body_type_2=ContactInformation.WORLD_OBJECT, depth=.003)]
        return SimpleNamespace(valid=not contacts, contacts=contacts)
    fake = SimpleNamespace(execution_config=config, group_name='ur_manipulator', frame_id='base_link',
        ee_link='tool0', robot_base_height=0., down_orientation=Quaternion(x=1., w=0.),
        ARM_JOINT_NAMES=['shoulder_pan_joint'], moveit_error_name=lambda code: 'NO_IK_SOLUTION',
        ik_client=SimpleNamespace(call_async=ik), state_validity_client=SimpleNamespace(call_async=validity),
        wait_result=lambda f, timeout: f,
        _fk_matches=lambda s, xyz, label: fk_points.append(xyz) or True)
    fake._normalize_branch = lambda result, seed, xyz, label, diagnostic=False: result
    passed, reason = VisionRobotSkills._diagnose_cartesian_waypoints(fake,
        (.38, -.148, .4), (.38, -.148, .38), seed, 'red_cube Cartesian lowering')
    assert passed is (body is None)
    assert ik_requests and validity_requests
    assert all(req.ik_request.avoid_collisions is False for req in ik_requests)
    assert all(req.group_name == '' for req in validity_requests)
    assert all(req.ik_request.pose_stamped.pose.orientation == Quaternion(x=1., w=0.) for req in ik_requests)
    assert fk_points[0] == (.38, -.148, .4)
    if body != 'ik_error':
        assert fk_points[-1] == (.38, -.148, .38)
        assert len(validity_requests) == len(ik_requests)+1
        for a, b in zip(fk_points, fk_points[1:]):
            assert math.dist(a, b) <= .00200001
    output = capsys.readouterr().out
    assert 'waypoint index=0/' in output and 'z=0.380000' in output
    if body not in (None, 'ik_error'):
        assert f'collision contact pair=simple_gripper_left_finger_link <-> {body}' in output
        assert 'robot_links=' in output and f"collision_objects=['{body}']" in output
        assert body in reason
    elif body == 'ik_error':
        assert 'state_valid=UNKNOWN' in output and 'GEOMETRIC_IK_FAILED' in reason


def test_contact_acm_only_target_and_resets_after_attach(planner, state, config):
    skills, scene, baseline, _ = fake_precheck(planner, state, config)
    matrix = extend_collision_matrix(baseline.allowed_collision_matrix, ['red_cube', 'wrist_3_link',
        'simple_gripper_base_link'])
    r, w, g = (matrix.entry_names.index(n) for n in ('red_cube', 'wrist_3_link', 'simple_gripper_base_link'))
    matrix.entry_values[r].enabled[w] = matrix.entry_values[w].enabled[r] = True
    matrix.entry_values[w].enabled[g] = matrix.entry_values[g].enabled[w] = True
    baseline.allowed_collision_matrix = matrix
    skills._allow_grasp_touch(baseline, 'red_cube')
    matrix = scene.allowed_collision_matrix
    assert matrix.entry_values[matrix.entry_names.index('red_cube')].enabled[matrix.entry_names.index('wrist_3_link')] is False
    assert matrix.entry_values[matrix.entry_names.index('wrist_3_link')].enabled[matrix.entry_names.index('simple_gripper_base_link')] is True
    links = ['simple_gripper_base_link', 'simple_gripper_left_finger_link', 'simple_gripper_right_finger_link']
    for cube in state['objects']:
        for link in links:
            i, j = matrix.entry_names.index(cube), matrix.entry_names.index(link)
            assert matrix.entry_values[i].enabled[j] is (cube == 'red_cube')
    skills._reset_grasp_touch(baseline)
    matrix = scene.allowed_collision_matrix
    for cube in state['objects']:
        for link in links:
            assert matrix.entry_values[matrix.entry_names.index(cube)].enabled[matrix.entry_names.index(link)] is False


def test_valid_diagnostic_ik_cannot_override_quarter_cartesian_path(config):
    fake = SimpleNamespace(execution_config=config, frame_id='base_link', group_name='ur_manipulator',
        ee_link='tool0', robot_base_height=0., motion_scaling_factor=.2, down_orientation=Quaternion(x=1., w=0.),
        _fk_matches=lambda *a: True, _diagnose_cartesian_waypoints=lambda *a: (True, 'WAYPOINTS_VALID'),
        cartesian_client=SimpleNamespace(call_async=lambda req: SimpleNamespace(
            solution=RobotTrajectory(), fraction=.25, error_code=MoveItErrorCodes(val=MoveItErrorCodes.SUCCESS))),
        wait_result=lambda f, timeout: f)
    fake.last_precheck_reason = 'SUCCESS'
    passed, _ = VisionRobotSkills._cartesian_precheck(fake, (.38, -.148, .4), (.38, -.148, .38),
                                                     RobotState(), 'red_cube Cartesian lowering')
    assert passed is False and 'fraction=0.25' in fake.last_precheck_reason


def test_diagnostic_entrypoint_cannot_enable_execution(monkeypatch, capsys):
    from ur3_vision_planning import motion_precheck
    calls = []
    monkeypatch.setattr(motion_precheck.rclpy, 'init', lambda **kw: None)
    monkeypatch.setattr(motion_precheck.rclpy, 'ok', lambda: False)
    monkeypatch.setattr(motion_precheck, 'NaturalLanguagePlanner', lambda: SimpleNamespace(
        get_parameter=lambda n: SimpleNamespace(value=False), destroy_node=lambda: calls.append('destroy')))
    monkeypatch.setattr(motion_precheck, 'execute_validated_plan', lambda *a: pytest.fail('execution must be refused'))
    assert motion_precheck.main() == 1
    assert calls == ['destroy']
    assert 'DIAGNOSTIC_NODE_REQUIRES_PRECHECK_ONLY' in capsys.readouterr().out


def test_slot_eight_exact_point_and_final_plan_before_red_failure(planner, state, config, capsys):
    goal = {'object': 'red_cube', 'target_zone': 'zone_b'}
    plan = planner.resolve_goal('red_cube', 'zone_b', state)
    skills, _, _, _ = fake_precheck(planner, state, config,
        rejection=lambda label: any(label.startswith(f'temporary_{i} ') for i in range(1, 8)))
    original_cartesian = skills._cartesian_precheck
    def cartesian(start, target, seed, label):
        if label == 'red_cube Cartesian lowering':
            skills.last_precheck_reason = 'green_cube <-> simple_gripper_left_finger_link'
            return False, seed
        return original_cartesian(start, target, seed, label)
    skills._cartesian_precheck = cartesian
    skills.feasibility_precheck = lambda plan, state, **kw: VisionRobotSkills.feasibility_precheck(skills, plan, state, **kw)
    synced = []
    skills.sync_camera_scene = lambda observed: synced.append(copy.deepcopy(observed))
    skills.destroy_node = lambda: None
    manager = SimpleNamespace(planner=planner, fresh_snapshot=lambda: copy.deepcopy(state),
        get_parameter=lambda n: SimpleNamespace(value={'dry_run': False, 'execute_robot': True, 'precheck_only': True}[n]))
    previous = copy.deepcopy(planner.selected_temporary_positions)
    with pytest.raises(MotionExecutionError, match='green_cube'):
        execute_validated_plan(manager, goal, state, plan, config, lambda *a: skills)
    assert planner.selected_temporary_positions == previous
    assert skills.precheck_scene_restored is True and synced
    output = capsys.readouterr().out
    assert '2. place(blue_cube, temporary_8)' in output
    assert '4. place(red_cube, zone_b)' in output
    assert 'VALIDATION AFTER TEMPORARY SELECTION: PLAN VALID' in output
    assert 'temporary_8 EXACT cube-centre coordinates=(0.280, -0.270, 0.220); NO ZONE OFFSET' in output
    assert 'temporary_8 selected y_offset' not in output
    assert 'PLANNING SCENE RESTORED' in output and 'PLANNING SCENE RESYNCED FROM CAMERA AFTER PRECHECK' in output
    assert 'TASK SUCCESS' not in output


def finger_box(tool_pose, side, q):
    """AABB of actual URDF box for tool orientation (1,0,0,0), not arm IK."""
    root = ET.parse(ROOT/'urdf/simple_parallel_gripper.urdf.xacro').getroot()
    link = next(e for e in root.iter('link') if e.attrib['name'] == '${name}_'+side+'_finger_link')
    joint = next(e for e in root.iter('joint') if e.attrib['name'] == '${name}_'+side+'_finger_joint')
    origin = [float(v) for v in joint.find('origin').attrib['xyz'].split()]
    axis = [float(v) for v in joint.find('axis').attrib['xyz'].split()]
    center = [float(v) for v in link.find('collision/origin').attrib['xyz'].split()]
    size = [float(v) for v in link.find('collision/geometry/box').attrib['size'].split()]
    local = [a + b + q*c for a, b, c in zip(origin, center, axis)]
    position = [a + sign*b for a, b, sign in zip(tool_pose, local, (1., -1., -1.))]
    return [(c-s/2, c+s/2) for c, s in zip(position, size)]


def intersects(finger, object_point, cube_size):
    return all(min(b, c+s/2) > max(a, c-s/2) for (a, b), c, s in zip(finger, object_point, cube_size))


def test_source_geometry_proves_old_layout_overlap_and_new_layout_clearance(planner, state):
    old = load_scene(ROOT/'config/scene.yaml', 'default')
    # This is an offline source-geometry result, NOT a GetStateValidity contact.
    finger = finger_box((.38, -.148, .38), 'left', 0.)
    assert intersects(finger, old['objects']['green_cube']['position'], old['cube_size'])
    assert finger[2][0] == pytest.approx(.23)  # Table top .20: no finger/table overlap.
    current = load_scene(ROOT/'config/scene.yaml', 'zone_b_occupied')
    positions = [obj['position'] for obj in current['objects'].values()]
    assert min(math.dist(a[:2], b[:2]) for i, a in enumerate(positions) for b in positions[i+1:]) >= .12 - 1e-9
    for name, obj in current['objects'].items():
        center = obj['position']
        assert all(abs(c-t)+s/2 <= extent/2 for c, t, s, extent in
                   zip(center[:2], current['table']['center'][:2], current['cube_size'][:2], current['table']['size'][:2]))
        # No new spawn is farther radially than the unchanged red target.
        # This is a geometric bound, NOT a new IK PASS claim.
        assert math.hypot(*center[:2]) <= math.hypot(.38, -.15) + 1e-9
        camera = planner.camera
        rotation = planner.rpy_rotation(camera['rpy']) @ planner.rpy_rotation(camera['optical_rpy'])
        tx = math.tan(camera['horizontal_fov']/2)
        ty = tx*camera['height']/camera['width']
        for dx in (-current['cube_size'][0]/2, current['cube_size'][0]/2):
            for dy in (-current['cube_size'][1]/2, current['cube_size'][1]/2):
                point = np.array([center[0]+dx, center[1]+dy, center[2]+current['cube_size'][2]/2])
                optical = rotation.T @ (point-np.array(camera['position']))
                assert camera['near_clip'] <= optical[2] <= camera['far_clip']
                assert abs(optical[0]/optical[2]) <= tx and abs(optical[1]/optical[2]) <= ty
        for other, target in current['objects'].items():
            if name == other:
                continue
            for side, q in (('left', 0.), ('right', .043)):
                assert not intersects(finger_box((center[0], center[1], center[2]+.16), side, q),
                                      target['position'], current['cube_size'])

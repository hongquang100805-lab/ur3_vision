"""Read-only visualization fixtures; no ROS init or simulator start."""

import ast
from pathlib import Path

import pytest
from visualization_msgs.msg import Marker
import yaml

from ur3_vision_planning.scene_config import load_scene
from ur3_vision_planning.scene_visualization import scene_markers

ROOT = Path(__file__).resolve().parents[1]


def configs():
    return (load_scene(ROOT/'config/scene.yaml', 'zone_b_occupied'),
            yaml.safe_load((ROOT/'config/camera.yaml').read_text())['camera'],
            yaml.safe_load((ROOT/'config/visualization.yaml').read_text()))


def observed():
    return {'frame_id': 'base_link', 'stale': False,
            'objects': {'blue_cube': {'detected': True, 'stale': False,
                                    'position': {'x': .28, 'y': -.27, 'z': .22}}},
            'zones': {'zone_b': {'stale': False, 'status': 'empty'}}}


def test_initial_full_scene_labels_and_camera():
    scene, camera, config = configs()
    markers = scene_markers(scene, camera, config).markers
    names = {m.ns: m for m in markers if m.action == Marker.ADD}
    assert len([n for n in names if n.endswith('_cube_visual')]) == 5
    assert names['blue_cube_visual'].pose.position.x == pytest.approx(.28)
    assert all(f'zone_{z}_label' in names for z in ('a', 'b', 'c'))
    assert names['zone_b_label'].text == 'Zone B - Blue\nunknown'
    assert names['camera_body'].header.frame_id == camera['link_frame']
    assert all(m.pose.orientation.w == 1. for m in names.values())
    assert all('initial spawn' in names[n+'_label'].text for n in scene['objects'])


def test_camera_positions_not_spawn_and_no_phantom_missing_cubes():
    markers = scene_markers(*configs(), observed(), True).markers
    names = {m.ns: m for m in markers if m.action == Marker.ADD}
    assert names['blue_cube_visual'].pose.position.y == pytest.approx(-.27)
    assert 'red_cube_visual' not in names
    assert names['zone_b_label'].text.endswith('empty')
    assert 'initial spawn' not in names['blue_cube_label'].text
    assert markers[0].action == Marker.DELETEALL


@pytest.mark.parametrize('base_height', [0., .1])
def test_four_table_legs_match_gazebo_floor_and_table_bottom(base_height):
    scene, camera, config = configs()
    scene['robot_base_height'] = base_height
    markers = scene_markers(scene, camera, config).markers
    legs = [m for m in markers if m.ns.startswith('table_leg_')]
    assert len(legs) == 4
    assert {(round(m.pose.position.x, 3), round(m.pose.position.y, 3)) for m in legs} == {
        (.25, -.28), (.25, .28), (.51, -.28), (.51, .28)}
    for leg in legs:
        assert leg.header.frame_id == 'base_link'
        assert leg.scale.x == pytest.approx(.04)
        assert leg.scale.y == pytest.approx(.04)
        assert leg.scale.z == pytest.approx(.17)
        assert leg.pose.position.z-leg.scale.z/2. == pytest.approx(-base_height)
        assert leg.pose.position.z+leg.scale.z/2. == pytest.approx(.17-base_height)


def test_stale_camera_hides_old_cube_and_marks_unknown():
    markers = scene_markers(*configs(), observed(), False).markers
    assert not any(m.ns.endswith('_cube_visual') for m in markers)
    assert next(m for m in markers if m.ns == 'zone_b_label').text.endswith('unknown')
    assert next(m for m in markers if m.ns == 'vision_status').text.endswith('STALE / UNKNOWN')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), None, True])
def test_bad_camera_coordinate_not_rendered(value):
    state = observed()
    state['objects']['blue_cube']['position']['x'] = value
    assert not any(m.ns == 'blue_cube_visual' for m in scene_markers(*configs(), state, True).markers)


def test_single_owned_rviz_and_inherited_windows_disabled():
    source = (ROOT/'launch/llm_robot.launch.py').read_text()
    tree = ast.parse(source)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'Node'
             and any(k.arg == 'package' and isinstance(k.value, ast.Constant)
                     and k.value.value == 'rviz2' for k in n.keywords)]
    assert len(nodes) == 1
    assert "'launch_rviz': 'false'" in source
    assert "dashboard_rviz = LaunchConfiguration('launch_rviz').perform(context)" in source
    assert 'condition=IfCondition(dashboard_rviz)' in source
    assert 'GroupAction(' not in source


@pytest.mark.parametrize('parent_rviz', ['true', 'false'])
def test_dashboard_choice_survives_include_and_delayed_ur_callback(parent_rviz, monkeypatch, tmp_path):
    # Exercise Jazzy include context writes and delayed configuration access.
    # A scoped group would lose ur_type before MoveIt's OnProcessExit callback.
    # No process, simulator, or ROS initialization.
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path))
    from launch import LaunchContext, LaunchDescription, LaunchDescriptionSource
    from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
    from launch.conditions import IfCondition
    from launch.substitutions import LaunchConfiguration

    seen = []
    def observe(context):
        seen.append(LaunchConfiguration('launch_rviz').perform(context))
    deferred = []
    def register_deferred(context):
        deferred.append(lambda: LaunchConfiguration('ur_type').perform(context))
    child = LaunchDescriptionSource(LaunchDescription([
        DeclareLaunchArgument('ur_type'),
        OpaqueFunction(function=observe), OpaqueFunction(function=register_deferred)]))
    included = IncludeLaunchDescription(child, launch_arguments={
        'launch_rviz': 'false', 'ur_type': 'ur3'}.items())
    context = LaunchContext()
    context.launch_configurations['launch_rviz'] = parent_rviz
    dashboard_rviz = LaunchConfiguration('launch_rviz').perform(context)

    def visit(entity):
        for child_entity in entity.visit(context) or []:
            visit(child_entity)
    visit(included)
    assert seen == ['false']  # both inherited RViz instances disabled
    assert context.launch_configurations['launch_rviz'] == 'false'
    assert IfCondition(dashboard_rviz).evaluate(context) == (parent_rviz == 'true')
    assert deferred[0]() == 'ur3'  # available AFTER include finishes


def test_native_jazzy_display_topics_qos():
    rviz = yaml.safe_load((ROOT/'rviz/vision.rviz').read_text())['Visualization Manager']
    displays = rviz['Displays']
    assert rviz['Global Options']['Fixed Frame'] == 'base_link'
    images = [d for d in displays if d['Class'] == 'rviz_default_plugins/Image']
    assert {d['Topic']['Value'] for d in images} == {
        '/camera/image_raw', '/camera/depth/image_raw', '/vision/debug_image'}
    assert all(d['Topic']['Reliability Policy'] == 'Best Effort' for d in images)
    depth = next(d for d in displays if d['Class'] == 'rviz_default_plugins/DepthCloud')
    camera = configs()[1]
    assert depth['CameraInfo Topic'] == camera['info_topic']
    assert depth['Depth Map Topic'] == camera['depth_topic']
    assert depth['Color Image Topic'] == camera['image_topic']
    assert depth['Reliability Policy'] == 'Best effort'
    markers = next(d for d in displays if d['Class'] == 'rviz_default_plugins/MarkerArray')
    assert markers['Topic']['Durability Policy'] == 'Transient Local'

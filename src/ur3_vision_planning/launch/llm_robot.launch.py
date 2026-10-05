import os
import tempfile
import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, OpaqueFunction
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ur3_vision_planning.scene_config import load_scene

def launch_setup(context):
    pkg_ur_simulation = get_package_share_directory('ur_simulation_gz')
    pkg_ur3_vision = get_package_share_directory('ur3_vision_planning')
    
    # Load scene.yaml
    scenario = LaunchConfiguration('scenario').perform(context)
    scene = load_scene(os.path.join(pkg_ur3_vision, 'config', 'scene.yaml'), scenario)

    camera_config = os.path.join(pkg_ur3_vision, 'config', 'camera.yaml')
    with open(camera_config, encoding='utf-8') as f:
        camera = yaml.safe_load(f)['camera']
    world_xml = xacro.process_file(
        os.path.join(pkg_ur3_vision, 'worlds', 'vision_world.sdf.xacro'),
        mappings={'camera_config': camera_config},
    ).toxml()
    # Render from the installed template/config into a unique runtime file.
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.sdf', prefix='ur3_vision_world_', delete=False,
        encoding='utf-8',
    ) as world_file:
        world_file.write(world_xml)
        world_path = world_file.name

    def cleanup_world(event, context):
        if os.path.exists(world_path):
            os.unlink(world_path)
        return []
        
    ur_sim_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ur_simulation, 'launch', 'ur_sim_moveit.launch.py')
        ),
        launch_arguments={
            'ur_type': 'ur3',
            'description_file': os.path.join(
                pkg_ur3_vision, 'urdf', 'ur3_table_mount.urdf.xacro'
            ),
            'controllers_file': os.path.join(
                pkg_ur3_vision, 'config', 'ur3_gripper_controllers.yaml'
            ),
            'launch_rviz': LaunchConfiguration('launch_rviz'),
            'gazebo_gui': LaunchConfiguration('gazebo_gui'),
            'world_file': world_path,
        }.items(),
    )
    
    nodes = [
      ur_sim_moveit,
      Node(
        package='ur3_vision_planning', executable='camera_perception',
        output='screen', parameters=[{'use_sim_time': True}],
      ),
      RegisterEventHandler(OnShutdown(on_shutdown=cleanup_world)),
      Node(
        package='ur3_vision_planning',
        executable='scene_publisher',
        output='screen',
        parameters=[{'scenario': scenario}],
      ),
      Node(
        package='controller_manager',
        executable='spawner',
        arguments=['simple_gripper_controller', '-c', '/controller_manager'],
        output='screen',
      ),
    ]

    def static_tf(name, parent, child, position, rpy):
        return Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=name, output='screen',
            arguments=[
                '--x', str(position[0]), '--y', str(position[1]),
                '--z', str(position[2]), '--roll', str(rpy[0]),
                '--pitch', str(rpy[1]), '--yaw', str(rpy[2]),
                '--frame-id', parent, '--child-frame-id', child,
            ],
        )

    nodes.extend([
        static_tf('overhead_camera_tf', camera['parent_frame'],
                  camera['link_frame'], camera['position'], camera['rpy']),
        static_tf('overhead_camera_optical_tf', camera['link_frame'],
                  camera['optical_frame'], [0.0, 0.0, 0.0], camera['optical_rpy']),
        Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name='camera_bridge', output='screen',
            parameters=[{
                'use_sim_time': True,
                f"qos_overrides.{camera['image_topic']}.publisher.reliability": 'best_effort',
            }],
            arguments=[
                camera['gz_topic'] + '/image@sensor_msgs/msg/Image[gz.msgs.Image',
                camera['gz_topic'] + '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                camera['gz_topic'] + '/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            ],
            remappings=[
                (camera['gz_topic'] + '/image', camera['image_topic']),
                (camera['gz_topic'] + '/camera_info', camera['info_topic']),
                (camera['gz_topic'] + '/depth_image', camera['depth_topic']),
            ],
        ),
    ])
    
    # Function to create an SDF string for a colored box
    cube_size = " ".join(str(value) for value in scene.get(
        'cube_size', [0.04, 0.04, 0.04]
    ))
    mass = scene.get('cube_mass', 0.1)
    sx, sy, sz = scene['cube_size']
    inertia = [mass * (sy**2 + sz**2) / 12.0,
               mass * (sx**2 + sz**2) / 12.0,
               mass * (sx**2 + sy**2) / 12.0]

    def get_box_sdf(name, color_rgba, size=cube_size):
        return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{inertia[0]}</ixx> <iyy>{inertia[1]}</iyy> <izz>{inertia[2]}</izz>
          <ixy>0</ixy> <ixz>0</ixz> <iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{size}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>{size}</size></box></geometry>
        <material>
          <ambient>{color_rgba}</ambient>
          <diffuse>{color_rgba}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    def get_table_sdf():
        table_center = scene['table']['center']
        table_size = scene['table']['size']
        leg_height = table_center[2] - table_size[2] / 2
        leg_center_z = leg_height / 2
        leg_x_offset = table_size[0] / 2 - 0.02
        leg_y_offset = table_size[1] / 2 - 0.045
        leg1_x = table_center[0] - leg_x_offset
        leg2_x = table_center[0] + leg_x_offset
        leg1_y = table_center[1] - leg_y_offset
        leg2_y = table_center[1] + leg_y_offset
        return """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="table">
    <static>true</static>
    <link name="link">
      <!-- Table geometry is expressed in the Gazebo world frame. -->
      <visual name="top_visual">
        <pose>{table_center[0]} {table_center[1]} {table_center[2]} 0 0 0</pose>
        <geometry><box><size>{table_size[0]} {table_size[1]} {table_size[2]}</size></box></geometry>
        <material>
          <ambient>0.35 0.35 0.38 1</ambient>
          <diffuse>0.60 0.60 0.65 1</diffuse>
        </material>
      </visual>
      <collision name="top_collision">
        <pose>{table_center[0]} {table_center[1]} {table_center[2]} 0 0 0</pose>
        <geometry><box><size>{table_size[0]} {table_size[1]} {table_size[2]}</size></box></geometry>
      </collision>
      <!-- 4 Legs -->
      <visual name="leg1">
        <pose>{leg1_x} {leg1_y} {leg_center_z} 0 0 0</pose>
        <geometry><box><size>0.04 0.04 {leg_height}</size></box></geometry>
        <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material>
      </visual>
      <visual name="leg2">
        <pose>{leg1_x} {leg2_y} {leg_center_z} 0 0 0</pose>
        <geometry><box><size>0.04 0.04 {leg_height}</size></box></geometry>
        <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material>
      </visual>
      <visual name="leg3">
        <pose>{leg2_x} {leg1_y} {leg_center_z} 0 0 0</pose>
        <geometry><box><size>0.04 0.04 {leg_height}</size></box></geometry>
        <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material>
      </visual>
      <visual name="leg4">
        <pose>{leg2_x} {leg2_y} {leg_center_z} 0 0 0</pose>
        <geometry><box><size>0.04 0.04 {leg_height}</size></box></geometry>
        <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>""".format(
            table_center=table_center,
            table_size=table_size,
            leg1_x=leg1_x,
            leg2_x=leg2_x,
            leg1_y=leg1_y,
            leg2_y=leg2_y,
            leg_center_z=leg_center_z,
            leg_height=leg_height,
        )

    def get_zone_sdf(name, color_rgba):
        marker_size = scene.get('zone_marker_size', [0.09, 0.09])
        usable_size = scene.get('zone_usable_size', [0.075, 0.075])
        return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="border">
        <pose>0 0 0 0 0 0</pose>
        <geometry><box><size>{marker_size[0]} {marker_size[1]} 0.001</size></box></geometry>
        <material>
          <ambient>0.1 0.1 0.1 1</ambient>
          <diffuse>0.15 0.15 0.15 1</diffuse>
        </material>
      </visual>
      <visual name="pad">
        <pose>0 0 0.0005 0 0 0</pose>
        <geometry><box><size>{usable_size[0]} {usable_size[1]} 0.001</size></box></geometry>
        <material>
          <ambient>{color_rgba}</ambient>
          <diffuse>{color_rgba}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    # Spawn table
    nodes.append(Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'table', '-string', get_table_sdf(), '-x', '0', '-y', '0', '-z', '0'],
        output='screen'
    ))

    # Spawn zones (zone_a, zone_b, zone_c)
    zone_colors = {
        'zone_a': '1.0 0.85 0.0 0.9',  # Yellow (matches student mapping)
        'zone_b': '0.1 0.45 1.0 0.9',  # Blue
        'zone_c': '1.0 0.2 0.2 0.9'    # Red
    }
    for zone_name, zone_data in scene.get('zones', {}).items():
        pos = zone_data['position']
        color = zone_colors.get(zone_name, '0.5 0.5 0.5 0.9')
        nodes.append(Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', zone_name,
                '-string', get_zone_sdf(zone_name, color),
                '-x', str(pos[0]),
                '-y', str(pos[1]),
                '-z', str(pos[2])
            ],
            output='screen'
        ))

    # Spawn all five cubes using the single scene configuration.
    for obj_name, obj_data in scene.get('objects', {}).items():
        pos = obj_data['position']
        color = ' '.join(str(value) for value in obj_data['color'])
        nodes.append(Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', obj_name,
                '-string', get_box_sdf(obj_name, color),
                '-x', str(pos[0]),
                '-y', str(pos[1]),
                '-z', str(pos[2])
            ],
            output='screen'
        ))

    return nodes


def generate_launch_description():
    launch_arguments = [
      DeclareLaunchArgument('launch_rviz', default_value='true'),
      DeclareLaunchArgument('gazebo_gui', default_value='true'),
      DeclareLaunchArgument('scenario', default_value='default',
                           choices=['default', 'zone_b_occupied'],
                           description='Initial spawn layout; applied before simulation objects are created'),
    ]
    return LaunchDescription(launch_arguments + [OpaqueFunction(function=launch_setup)])

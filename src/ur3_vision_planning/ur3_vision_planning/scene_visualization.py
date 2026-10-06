"""Read-only RViz markers. Never modify MoveIt/Gazebo or send motion commands."""

import json
import math
import os
import time

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from .scene_config import load_scene


def scene_markers(scene, camera, config, state=None, fresh=False):
    """Pure marker construction; stale/missing detections are never spawn poses."""
    result = MarkerArray()
    # A complete replace also removes disappeared cubes without stale leftovers.
    result.markers.append(Marker(action=Marker.DELETEALL))
    frame = config['frame_id']
    base_height = scene.get('robot_base_height', 0.)

    def add(name, kind, position, size, color, text='', parent=frame):
        marker = Marker()
        marker.header.frame_id = parent
        marker.ns, marker.id, marker.type, marker.action = name, 0, kind, Marker.ADD
        marker.pose.orientation.w = 1.
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, position)
        marker.scale.x, marker.scale.y, marker.scale.z = map(float, size)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = map(float, color)
        marker.text, marker.frame_locked = text, True
        result.markers.append(marker)

    def label(name, position, text, parent=frame):
        add(name, Marker.TEXT_VIEW_FACING, position, [0., 0., config['label_height_m']],
            [1., 1., 1., 1.], text, parent)

    table = scene['table']
    add('table_visual', Marker.CUBE,
        [*table['center'][:2], table['center'][2]-base_height], table['size'], [.35, .35, .38, 1.])
    legs = config['table_legs']
    table_bottom = table['center'][2] - table['size'][2]/2.
    floor_z = legs['floor_z_world_m']
    leg_height = table_bottom - floor_z
    if leg_height > 0.:
        dx = table['size'][0]/2. - legs['edge_inset_xy_m'][0]
        dy = table['size'][1]/2. - legs['edge_inset_xy_m'][1]
        for index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)), 1):
            add(f'table_leg_{index}', Marker.CUBE,
                [table['center'][0]+sx*dx, table['center'][1]+sy*dy,
                 (floor_z+table_bottom)/2.-base_height],
                [legs['width_m'], legs['width_m'], leg_height], legs['color'])
    for name, zone in scene['zones'].items():
        pos = [*zone['position'][:2], zone['position'][2]-base_height]
        add(name, Marker.CUBE, pos, [*scene['zone_marker_size'], .001],
            [*config['zone_colors'][name][:3], config['zone_alpha']])
        status = 'unknown'
        if fresh:
            observed = state.get('zones', {}).get(name, {})
            if not observed.get('stale', True):
                status = str(observed.get('status', 'unknown'))
        label(name+'_label', [*pos[:2], pos[2]+config['zone_label_offset_m']],
              config['zone_labels'][name]+'\n'+status)

    for name, obj in scene['objects'].items():
        if state is None:
            position = [*obj['position'][:2], obj['position'][2]-base_height]
            suffix = ' (initial spawn; no camera state yet)'
        else:
            observed = state.get('objects', {}).get(name, {})
            if not fresh or not observed.get('detected', False) or observed.get('stale', True):
                continue
            p = observed.get('position')
            if not isinstance(p, dict) or not all(
                    isinstance(p.get(a), (int, float)) and not isinstance(p[a], bool)
                    and math.isfinite(p[a]) for a in ('x', 'y', 'z')):
                continue
            position = [p[a] for a in ('x', 'y', 'z')]
            suffix = ''
        add(name+'_visual', Marker.CUBE, position, scene['cube_size'], obj['color'])
        label(name+'_label', [*position[:2], position[2]+config['object_label_offset_m']], name+suffix)

    add('camera_body', Marker.CUBE, [0., 0., 0.], config['camera_body_size'],
        [.15, .15, .2, 1.], parent=camera['link_frame'])
    label('camera_label', [0., 0., .07], 'RGB-D camera', camera['link_frame'])
    label('vision_status', [table['center'][0], -.4, table['center'][2]+.1-base_height],
          'Camera state: '+('LIVE' if fresh else ('STALE / UNKNOWN' if state is not None else 'WAITING')))
    return result


class SceneVisualization(Node):
    def __init__(self):
        super().__init__('scene_visualization')
        share = get_package_share_directory('ur3_vision_planning')
        def read(name):
            with open(os.path.join(share, 'config', name), encoding='utf-8') as stream:
                return yaml.safe_load(stream)
        self.config, self.camera = read('visualization.yaml'), read('camera.yaml')['camera']
        self.declare_parameter('scenario', 'default')
        self.scene = load_scene(os.path.join(share, 'config', 'scene.yaml'),
                                self.get_parameter('scenario').value)
        self.state, self.received_at = None, None
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(MarkerArray, self.config['marker_topic'], qos)
        self.subscription = self.create_subscription(String, self.config['environment_topic'], self.on_state, 10)
        self.timer = self.create_timer(1./self.config['update_rate_hz'], self.publish_markers,
                                      clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.publish_markers()

    def on_state(self, msg):
        try:
            state = json.loads(msg.data)
            if (not isinstance(state, dict) or state.get('frame_id') != self.config['frame_id']
                    or not isinstance(state.get('objects'), dict) or not isinstance(state.get('zones'), dict)):
                raise ValueError('Invalid environment state/frame')
            if not all(isinstance(item, dict) for collection in ('objects', 'zones')
                       for item in state[collection].values()):
                raise ValueError('Invalid object/zone record')
            self.state, self.received_at = state, time.monotonic()
        except (ValueError, TypeError) as error:
            self.get_logger().warning(f'Visualization ignored invalid camera state: {error}')

    def publish_markers(self):
        fresh = (self.state is not None and not self.state.get('stale', True)
                 and time.monotonic()-self.received_at <= self.config['state_timeout_s'])
        self.publisher.publish(scene_markers(self.scene, self.camera, self.config, self.state, fresh))


def main(args=None):
    rclpy.init(args=args)
    node = SceneVisualization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

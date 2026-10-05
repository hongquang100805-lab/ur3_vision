"""Camera-only HSV/depth perception; no simulator model-pose inputs."""

from collections import deque
import json
import math
import os
import time

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import yaml


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def transform_matrix(transform):
    q = transform.rotation
    x, y, z, w = np.array([q.x, q.y, q.z, q.w]) / np.linalg.norm(
        [q.x, q.y, q.z, q.w])
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    t = transform.translation
    return rotation, np.array([t.x, t.y, t.z])


def median_depth(depth, u, v, config):
    radius = config['patch_radius_px']
    patch = depth[max(0, v-radius):v+radius+1, max(0, u-radius):u+radius+1]
    values = patch[np.isfinite(patch) & (patch > 0)
                   & (patch >= config['min_m']) & (patch <= config['max_m'])]
    if values.size < config['min_valid_samples']:
        return None
    return float(np.median(values))


def object_zone(position, scene, config, block_size):
    table_top = scene['table']['center'][2] + scene['table']['size'][2]/2
    if abs(position[2] - (table_top + block_size[2]/2)) > config['support_z_tolerance_m']:
        return None
    usable = scene['zone_usable_size']
    for name, data in scene['zones'].items():
        if all(abs(position[i] - data['position'][i]) + block_size[i]/2
               <= usable[i]/2 - config['footprint_margin_m'] for i in (0, 1)):
            return name
    return None


class CameraPerception(Node):
    def __init__(self):
        super().__init__('camera_perception')
        share = get_package_share_directory('ur3_vision_planning')
        self.declare_parameter('config_file', os.path.join(share, 'config', 'perception.yaml'))
        self.declare_parameter('scene_file', os.path.join(share, 'config', 'scene.yaml'))
        with open(self.get_parameter('config_file').value, encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        with open(self.get_parameter('scene_file').value, encoding='utf-8') as f:
            self.scene = yaml.safe_load(f)
        if self.config['block_size'] != self.scene['cube_size']:
            raise ValueError('perception block_size must match scene cube_size')
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.info = None
        self.depth_queue = deque(maxlen=self.config['depth']['queue_length'])
        self.rgb_queue = deque(maxlen=self.config['depth']['queue_length'])
        self.tracks = {}
        self.last_image = None
        self.last_depth = None
        self.last_processed = None
        self.last_measurement_stamp = None
        self.last_log = -math.inf
        self.reason = 'waiting_for_camera'
        topics = self.config['topics']
        self.image_sub = self.create_subscription(Image, topics['image'], self.on_image,
                                                   qos_profile_sensor_data)
        self.info_sub = self.create_subscription(CameraInfo, topics['camera_info'],
                                                  self.on_info, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, topics['depth'], self.on_depth,
                                                   qos_profile_sensor_data)
        self.debug_pub = self.create_publisher(Image, topics['debug'], qos_profile_sensor_data)
        self.state_pub = self.create_publisher(String, topics['state'], 10)
        # Wall/steady timer still publishes stale status when /clock stops.
        self.timer = self.create_timer(1/self.config['temporal']['publish_rate_hz'],
                                      self.publish_state,
                                      clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info('Camera perception ready: HSV + synchronized RGB-D + TF')

    def on_info(self, message):
        self.info = message

    def on_image(self, message):
        self.last_image = time.monotonic()
        self.rgb_queue.append((message, self.last_image))
        self.process_pending()

    def on_depth(self, message):
        self.last_depth = time.monotonic()
        self.depth_queue.append((message, self.last_depth))
        self.process_pending()

    def process_pending(self):
        if self.info is None or not self.depth_queue:
            self.reason = 'waiting_for_camera_info_or_depth'
            return
        while self.rgb_queue:
            rgb, received = self.rgb_queue[0]
            rgb_stamp = stamp_seconds(rgb.header.stamp)
            depth_msg, depth_received = min(
                self.depth_queue,
                key=lambda item: abs(stamp_seconds(item[0].header.stamp) - rgb_stamp))
            if abs(stamp_seconds(depth_msg.header.stamp) - rgb_stamp) > self.config['depth']['max_rgb_depth_delta_s']:
                if stamp_seconds(self.depth_queue[-1][0].header.stamp) < rgb_stamp:
                    return
                self.rgb_queue.popleft()
                self.reason = 'rgb_depth_not_synchronized'
                continue
            self.rgb_queue.popleft()
            if time.monotonic() - min(received, depth_received) > self.config['depth']['timeout_s']:
                self.reason = 'old_rgb_depth_pair'
                continue
            measurement_age = self.get_clock().now().nanoseconds*1e-9 - rgb_stamp
            if abs(measurement_age) > self.config['temporal']['max_measurement_age_s']:
                self.reason = 'old_image_stamp'
                continue
            if self.last_measurement_stamp is not None and rgb_stamp <= self.last_measurement_stamp:
                continue
            try:
                self.process_pair(rgb, depth_msg)
            except Exception as error:
                self.reason = 'processing_failed'
                self.warn(f'Camera perception rejected frame: {error}')

    def warn(self, message):
        now = time.monotonic()
        if now - self.last_log >= self.config['temporal']['log_interval_s']:
            self.get_logger().warning(message)
            self.last_log = now

    def process_pair(self, rgb, depth_msg):
        if rgb.header.frame_id != depth_msg.header.frame_id or rgb.header.frame_id != self.info.header.frame_id:
            raise ValueError('RGB/depth/CameraInfo frame mismatch; aligned RGB-D required')
        if (rgb.width, rgb.height) != (depth_msg.width, depth_msg.height) or (rgb.width, rgb.height) != (self.info.width, self.info.height):
            raise ValueError('RGB/depth/CameraInfo dimensions mismatch')
        k = np.array(self.info.k).reshape(3, 3)
        if k[0, 0] <= 0 or k[1, 1] <= 0 or any(abs(v) > 1e-8 for v in self.info.d):
            raise ValueError('Valid zero-distortion calibrated RGB-D CameraInfo required')
        transform = self.tf_buffer.lookup_transform(
            self.config['target_frame'], rgb.header.frame_id,
            rclpy.time.Time.from_msg(rgb.header.stamp),
            timeout=Duration(seconds=self.config['temporal']['tf_timeout_s']))
        rotation, translation = transform_matrix(transform.transform)
        bgr = self.bridge.imgmsg_to_cv2(rgb, desired_encoding=self.config['image_encoding'])
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)
        if depth_msg.encoding == '16UC1':
            depth *= self.config['depth']['unit_scale_16uc1']
        elif depth_msg.encoding != '32FC1':
            raise ValueError(f'Unsupported depth encoding {depth_msg.encoding}')
        rows, cols = np.indices(depth.shape)
        rays = np.stack(((cols-k[0, 2])/k[0, 0], (rows-k[1, 2])/k[1, 1], np.ones_like(cols)), axis=-1)
        points = (rays * depth[..., None]) @ rotation.T + translation
        table = self.scene['table']
        center = np.array(table['center'])
        center[2] -= self.scene.get('robot_base_height', 0.0)
        table_top = center[2] + table['size'][2]/2
        roi_cfg = self.config['roi']
        valid = (np.isfinite(depth) & (depth > 0)
                 & (depth >= self.config['depth']['min_m'])
                 & (depth <= self.config['depth']['max_m']))
        for axis in (0, 1):
            valid &= abs(points[..., axis]-center[axis]) <= table['size'][axis]/2 + roi_cfg['table_margin_m']
        valid &= points[..., 2] >= table_top + roi_cfg['block_top_above_table_m'][0]
        valid &= points[..., 2] <= table_top + roi_cfg['block_top_above_table_m'][1]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        debug = bgr.copy()
        contour_cfg = self.config['contours']
        kernel = np.ones((contour_cfg['morphology_kernel'],)*2, dtype=np.uint8)
        now = time.monotonic()
        stamp = stamp_seconds(rgb.header.stamp)
        for object_index, (name, ranges) in enumerate(self.config['hsv'].items()):
            mask = np.zeros(depth.shape, dtype=np.uint8)
            for bounds in ranges:
                mask |= cv2.inRange(hsv, np.array(bounds['lower']), np.array(bounds['upper']))
            mask[~valid] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                    iterations=contour_cfg['open_iterations'])
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                    iterations=contour_cfg['close_iterations'])
            candidates = []
            for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                area = cv2.contourArea(contour)
                x, y, width, height = cv2.boundingRect(contour)
                aspect = width/height
                fill = area/(width*height)
                if not (contour_cfg['min_area_px'] <= area <= contour_cfg['max_area_px']
                        and contour_cfg['min_aspect'] <= aspect <= contour_cfg['max_aspect']
                        and fill >= contour_cfg['min_fill_ratio']):
                    continue
                moments = cv2.moments(contour)
                u, v = int(round(moments['m10']/moments['m00'])), int(round(moments['m01']/moments['m00']))
                distance = median_depth(depth, u, v, self.config['depth'])
                if distance is None:
                    continue
                physical_size = np.array([width/k[0, 0], height/k[1, 1]])*distance
                ratios = physical_size/np.array(self.config['block_size'][:2])
                low, high = contour_cfg['physical_size_ratio']
                if np.any(ratios < low) or np.any(ratios > high):
                    continue
                optical = np.array([(u-k[0, 2])*distance/k[0, 0],
                                    (v-k[1, 2])*distance/k[1, 1], distance])
                position = rotation @ optical + translation
                # Depth hits the top face. Report cube centre in base_link.
                position[2] -= self.config['block_size'][2]/2
                confidence = float(np.clip(fill * min(aspect, 1/aspect)
                                             * min(1.0, float(np.min(ratios))), 0, 1))
                candidates.append((confidence, position, (u, v), (x, y, width, height)))
            if not candidates:
                continue
            # Score all physically-valid candidates; never select biggest colour blob.
            confidence, position, pixel, bbox = max(candidates, key=lambda c: c[0])
            track = self.tracks.setdefault(name, {
                'history': deque(maxlen=self.config['temporal']['history_length'])})
            track['history'].append((now, position))
            while track['history'] and now-track['history'][0][0] > self.config['temporal']['object_timeout_s']:
                track['history'].popleft()
            track.update(last_seen=now, stamp=stamp, confidence=confidence, pixel=pixel, bbox=bbox)
            filtered = np.median([sample[1] for sample in track['history']], axis=0)
            zone = object_zone(filtered + np.array([0, 0, self.scene.get('robot_base_height', 0.0)]),
                               self.scene, self.config['zones'], self.config['block_size'])
            x, y, width, height = bbox
            style = self.config['debug']
            cv2.rectangle(debug, (x, y), (x+width, y+height), (255, 255, 255), style['line_thickness'])
            cv2.putText(debug, str(object_index+1), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, style['font_scale'],
                        (0, 0, 0), style['line_thickness'])
            for index, label in enumerate([
                f'{object_index+1} {name}', f'{filtered[0]:.3f},{filtered[1]:.3f},{filtered[2]:.3f}',
                zone or 'no zone']):
                cv2.putText(debug, label,
                            (style['label_panel_x_px'], style['label_panel_y_px']
                             + object_index*style['object_spacing_px']
                             + index*style['text_line_spacing_px']),
                            cv2.FONT_HERSHEY_SIMPLEX, style['font_scale'], (0, 0, 0), style['line_thickness'])
        self.last_processed = now
        self.last_measurement_stamp = stamp
        self.reason = 'ok'
        output = self.bridge.cv2_to_imgmsg(debug, encoding=self.config['image_encoding'])
        output.header = rgb.header
        self.debug_pub.publish(output)

    def publish_state(self):
        now = time.monotonic()
        stale = (self.last_image is None or self.last_depth is None or self.last_processed is None
                 or now-self.last_image > self.config['temporal']['image_timeout_s']
                 or now-self.last_depth > self.config['depth']['timeout_s']
                 or now-self.last_processed > self.config['temporal']['image_timeout_s'])
        objects = {}
        for name in self.config['hsv']:
            track = self.tracks.get(name)
            age = None if track is None else now-track['last_seen']
            detected = not stale and track is not None and age <= self.config['temporal']['object_timeout_s']
            item = {'detected': detected, 'stale': not detected, 'position': None,
                    'zone': None, 'confidence': 0.0, 'age_s': age,
                    'stamp': None if track is None else track['stamp'],
                    'pixel': None, 'bbox': None}
            if detected:
                position = np.median([sample[1] for sample in track['history']], axis=0)
                item.update(position=dict(zip(('x', 'y', 'z'), map(float, position))),
                            zone=object_zone(position + np.array([0, 0, self.scene.get('robot_base_height', 0.0)]),
                                             self.scene, self.config['zones'], self.config['block_size']),
                            confidence=track['confidence'] * max(0, 1-age/self.config['temporal']['object_timeout_s']),
                            pixel={'u': track['pixel'][0], 'v': track['pixel'][1]},
                            bbox=dict(zip(('x', 'y', 'width', 'height'), track['bbox'])))
            objects[name] = item
        complete = all(item['detected'] for item in objects.values())
        zones = {}
        for name in self.scene['zones']:
            occupants = [obj for obj, item in objects.items() if item['detected'] and item['zone'] == name]
            zones[name] = {'occupied': True if occupants else (False if complete else None),
                           'object': occupants[0] if len(occupants) == 1 else None,
                           'objects': occupants, 'stale': stale,
                           'status': 'occupied' if occupants else ('empty' if complete else 'unknown')}
        state = {'stamp': self.last_measurement_stamp,
                 'published_at': self.get_clock().now().nanoseconds*1e-9,
                 'frame_id': self.config['target_frame'], 'stale': stale,
                 'status': 'stale' if stale else ('ok' if complete else 'partial'),
                 'reason': self.reason if not stale else 'camera_depth_or_processing_timeout',
                 'objects': objects, 'zones': zones}
        self.state_pub.publish(String(data=json.dumps(state, allow_nan=False)))
        if stale:
            self.warn('Camera data stale: positions invalidated; zone occupancy unknown')


def main(args=None):
    rclpy.init(args=args)
    node = CameraPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

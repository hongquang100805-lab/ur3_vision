import math
import re
import subprocess
import time

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetCartesianPath,
    GetPositionIK,
    GetStateValidity,
)
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformListener
import yaml


class RobotSkills(Node):
    ARM_JOINT_NAMES = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')
    HOME_JOINT_VALUES = (0.0, -1.5707, 0.0, 0.0, 0.0, 0.0)

    def __init__(self, scene_config_path, server_timeout=None):
        super().__init__('robot_skills')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Load scene config
        with open(scene_config_path, 'r') as f:
            self.scene = yaml.safe_load(f)
        self.robot_base_height = self.scene.get('robot_base_height', 0.0)
            
        self.group_name = "ur_manipulator"
        self.ee_link = "tool0"
        self.frame_id = "base_link"
        self.motion_scaling_factor = 0.2
        self.grasp_object_offset = 0.16
        self.grasp_object_x_offset = 0.0
        # The finger collision geometry extends 0.15 m along tool +Z.  With the
        # down-facing tool orientation that is 0.15 m below tool0 in base_link.
        self.finger_reach = 0.15
        self.cube_size = self.scene.get('cube_size', [0.04, 0.04, 0.04])
        self.cube_height = self.cube_size[2]
        self.zone_marker_size = self.scene.get('zone_marker_size', [0.09, 0.09])
        self.zone_usable_size = self.scene.get('zone_usable_size', [0.075, 0.075])
        self.approach_clearance = 0.01
        
        # Action clients
        self.move_client = ActionClient(self, MoveGroup, "/move_action")
        self.execute_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        self.cartesian_client = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.state_validity_client = self.create_client(
            GetStateValidity, "/check_state_validity"
        )
        self.scene_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.gripper_joint_name = "simple_gripper_left_finger_joint"
        self.gripper_joint_states = {}
        self.joint_states = {}
        self.gripper_joint_names = [
            "simple_gripper_left_finger_joint",
            "simple_gripper_right_finger_joint",
        ]
        self.gripper_command_publisher = self.create_publisher(
            Float64MultiArray, "/simple_gripper_controller/commands", 10
        )
        self.gripper_state_subscription = self.create_subscription(
            JointState, "/joint_states", self.gripper_state_callback, 10
        )
        
        # Wait for servers
        self.get_logger().info("Waiting for MoveIt servers...")
        for client, action in ((self.move_client, True), (self.cartesian_client, False),
                               (self.ik_client, False), (self.state_validity_client, False),
                               (self.execute_client, True), (self.scene_client, False)):
            ready = (client.wait_for_server(timeout_sec=server_timeout) if action
                     else client.wait_for_service(timeout_sec=server_timeout))
            if not ready:
                self.destroy_node()
                raise RuntimeError('MOVEIT_SERVER_UNAVAILABLE: waiting for action/service timed out')
        self.get_logger().info("MoveIt servers ready.")
        
        # Hardcoded orientation for pointing down
        self.down_orientation = Pose().orientation
        self.down_orientation.x = 1.0
        self.down_orientation.y = 0.0
        self.down_orientation.z = 0.0
        self.down_orientation.w = 0.0
        self.gazebo_cube_name = None
        self.gazebo_cube_attached = False
        self.last_simulator_pose_update = False
        self.simulator_pose_failed = False
        self.attached_object = None
        self.gazebo_attachment_plugins = set()
        # Only acknowledgements observed by this instance. Subscriber discovery
        # alone cannot prove the mechanical joint is attached or detached.
        self.gazebo_attachment_states = {}
        self.grasp_world_offset = None

    @staticmethod
    def attachment_subscriber_present(output, action):
        """Parse subscriber rows, not publisher types elsewhere in gz output.

        Gazebo 8's attach callback subscribes to generic ProtoMsg, advertised as
        google.protobuf.Message. Its detach callback is typed gz.msgs.Empty.
        """
        if action not in ('attach', 'detach'):
            raise ValueError('Invalid Gazebo attachment action')
        accepted_types = {'gz.msgs.Empty'}
        if action == 'attach':
            accepted_types.add('google.protobuf.Message')
        in_subscribers = False
        for line in output.splitlines():
            stripped = line.strip()
            if stripped == 'Subscribers [Address, Message Type]:':
                in_subscribers = True
                continue
            if not in_subscribers:
                continue
            if not line[:1].isspace():
                in_subscribers = False
                continue
            address, comma, message_type = stripped.rpartition(',')
            if comma and address.strip() and message_type.strip() in accepted_types:
                return True
        return False

    def gazebo_attachment_topic_has_subscriber(self, model_name, action='attach'):
        """Check whether this cube already has a DetachableJoint system."""
        if action not in ('attach', 'detach'):
            raise ValueError('Invalid Gazebo attachment action')
        topic = f"/ur3_vision_planning/grasp/{model_name}/{action}"
        try:
            result = subprocess.run(
                ["gz", "topic", "-i", "-t", topic],
                capture_output=True,
                text=True,
                timeout=6.0,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return (
            result.returncode == 0
            and self.attachment_subscriber_present(result.stdout, action)
        )

    def get_gazebo_model_entity_id(self, model_name):
        """Resolve a Gazebo entity ID for an unambiguous system/add target."""
        try:
            result = subprocess.run(
                ["gz", "model", "-m", model_name],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError) as error:
            self.get_logger().error(
                f"Cannot resolve Gazebo entity for {model_name}: {error}"
            )
            return None
        match = re.search(r"^Model:\s*\[(\d+)\]", result.stdout, re.MULTILINE)
        if result.returncode != 0 or match is None:
            details = result.stderr.strip() or "model entity ID missing"
            self.get_logger().error(
                f"Cannot resolve Gazebo entity for {model_name}: {details}"
            )
            return None
        return int(match.group(1))

    def load_gazebo_attachment_plugin(self, model_name):
        """Load one initially-attached joint only after physical finger contact."""
        if model_name in self.gazebo_attachment_plugins:
            return True
        if self.gazebo_attachment_topic_has_subscriber(model_name):
            self.gazebo_attachment_plugins.add(model_name)
            return True

        robot_entity_id = self.get_gazebo_model_entity_id("ur")
        if robot_entity_id is None:
            return False

        topic_root = f"/ur3_vision_planning/grasp/{model_name}"
        inner_xml = (
            "<parent_link>wrist_3_link</parent_link>"
            f"<child_model>{model_name}</child_model>"
            "<child_link>link</child_link>"
            f"<attach_topic>{topic_root}/attach</attach_topic>"
            f"<detach_topic>{topic_root}/detach</detach_topic>"
            f"<output_topic>{topic_root}/state</output_topic>"
        )
        escaped_xml = inner_xml.replace('\\', '\\\\').replace('"', '\\"')
        request = (
            f'entity: {{id: {robot_entity_id}}} '
            'plugins: {'
            'name: "gz::sim::systems::DetachableJoint" '
            'filename: "gz-sim-detachable-joint-system" '
            f'innerxml: "{escaped_xml}"'
            '}'
        )
        listener = None
        try:
            listener = subprocess.Popen(
                ["gz", "topic", "-e", "-t", f"{topic_root}/state", "-d", "3.0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(0.15)
            result = subprocess.run(
                [
                    "gz", "service", "-s", "/world/empty/entity/system/add",
                    "--reqtype", "gz.msgs.EntityPlugin_V",
                    "--reptype", "gz.msgs.Boolean",
                    "--timeout", "3000", "--req", request,
                ],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            output, error_output = listener.communicate(timeout=4.0)
            state = self.parse_detachable_joint_state(output)
            service_accepted = result.returncode == 0 and "data: true" in result.stdout
            plugin_loaded = (
                state is True
                or self.gazebo_attachment_topic_has_subscriber(model_name)
            )
            if not service_accepted or not plugin_loaded:
                details = (
                    error_output.strip() or result.stderr.strip()
                    or output.strip() or result.stdout.strip() or "no response"
                )
                self.get_logger().error(
                    f"Gazebo dynamic attach failed for {model_name}: {details}"
                )
                return False
            self.gazebo_attachment_plugins.add(model_name)
            if not hasattr(self, 'gazebo_attachment_states'):
                self.gazebo_attachment_states = {}
            self.gazebo_attachment_states[model_name] = state
            if state is True:
                self.get_logger().info(
                    f"Gazebo attach confirmed for model={model_name}, "
                    "parent_link=wrist_3_link, child_link=link"
                )
            else:
                self.get_logger().info(
                    f"Gazebo grasp joint loaded for model={model_name}; "
                    "subscriber confirmed but state UNKNOWN; attachment "
                    "requires a state acknowledgement before lift"
                )
            return True
        except (OSError, subprocess.SubprocessError) as error:
            self.get_logger().error(
                f"Cannot load Gazebo grasp joint for {model_name}: {error}"
            )
            return False
        finally:
            if listener is not None and listener.poll() is None:
                listener.kill()
                listener.communicate()

    def simulator_is_ready(self):
        """Refuse all motion unless this workspace's Gazebo scene is present."""
        try:
            result = subprocess.run(
                ["gz", "model", "--list"],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError) as error:
            self.get_logger().error(f"Cannot verify Gazebo simulator: {error}")
            return False
        models = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", result.stdout))
        required = {"table", *self.scene.get("objects", {})}
        missing = sorted(required - models)
        if result.returncode != 0 or missing:
            details = result.stderr.strip() or f"missing models: {missing}"
            self.get_logger().error(
                "Motion refused: the expected Gazebo scene is not ready " + details
            )
            return False
        return True

    def get_occupied_zones(self):
        """Return occupancy tracked from the launched scene and successful places."""
        occupied = {}
        for object_name, object_data in self.scene.get("objects", {}).items():
            object_position = object_data["position"]
            for zone_name, zone_data in self.scene.get("zones", {}).items():
                zone_position = zone_data["position"]
                if (
                    abs(object_position[0] - zone_position[0]) <= 0.045
                    and abs(object_position[1] - zone_position[1]) <= 0.045
                ):
                    occupied[zone_name] = object_name
        return occupied

    def set_gazebo_model_pose(self, model_name, x, y, z, attempts=1):
        """Dat vi tri model trong Gazebo."""
        request = (
            f'name: "{model_name}" '
            f'position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}} '
            f'orientation: {{w: 1.0}}'
        )

        details = "unknown error"
        for attempt in range(attempts):
            try:
                result = subprocess.run(
                    [
                        "gz",
                        "service",
                        "-s", "/world/empty/set_pose",
                        "--reqtype", "gz.msgs.Pose",
                        "--reptype", "gz.msgs.Boolean",
                        "--timeout", "1500",
                        "--req", request
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2.5
                )
                if "data: true" in result.stdout:
                    return True
                details = result.stderr.strip() or result.stdout.strip()
                if not details:
                    details = f"exit code {result.returncode}"
            except Exception as error:
                details = str(error)
            if attempt + 1 < attempts:
                time.sleep(0.1)
        self.get_logger().error(f"Gazebo set_pose failed: {details}")
        return False

    @staticmethod
    def parse_gazebo_model_pose(output):
        """Parse the XYZ row printed by `gz model -m NAME --pose`."""
        match = re.search(
            r"Pose\s*\[\s*XYZ.*?\]\s*:\s*\n\s*"
            r"\[\s*([-+0-9.eE]+)(?:\s*\|\s*|\s+)"
            r"([-+0-9.eE]+)(?:\s*\|\s*|\s+)"
            r"([-+0-9.eE]+)\s*\]",
            output,
            flags=re.DOTALL,
        )
        if match is None:
            return None
        return tuple(float(value) for value in match.groups())

    def get_gazebo_model_pose(self, model_name):
        """Read the simulator's actual model pose rather than trusting set_pose."""
        try:
            result = subprocess.run(
                ["gz", "model", "-m", model_name, "--pose"],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError) as error:
            self.get_logger().error(
                f"Cannot read Gazebo pose for {model_name}: {error}"
            )
            return None
        pose = self.parse_gazebo_model_pose(result.stdout)
        if result.returncode != 0 or pose is None:
            details = result.stderr.strip() or "unrecognized gz model --pose output"
            self.get_logger().error(
                f"Cannot verify Gazebo pose for {model_name}: {details}"
            )
            return None
        return pose

    @staticmethod
    def parse_detachable_joint_state(output):
        """Return the last state emitted by Gazebo 8 DetachableJoint."""
        named_states = re.findall(
            r'\bdata\s*:\s*"(attached|detached)"', output.lower()
        )
        if named_states:
            return named_states[-1] == "attached"
        states = re.findall(r"\bdata\s*:\s*(true|false)\b", output.lower())
        if not states:
            return None
        return states[-1] == "true"

    def set_gazebo_attachment(self, model_name, attached):
        """Attach / detach a cube and require Gazebo state confirmation."""
        if model_name not in self.scene.get("objects", {}):
            self.get_logger().error(
                f"Gazebo attachment refused for unknown model {model_name}"
            )
            return False

        action = "attach" if attached else "detach"
        topic_root = f"/ur3_vision_planning/grasp/{model_name}"
        state_topic = f"{topic_root}/state"
        command_topic = f"{topic_root}/{action}"
        if not hasattr(self, 'gazebo_attachment_states'):
            self.gazebo_attachment_states = {}
        # Initial dynamic loading already emits 'attached'. Do not send the
        # same command again: DetachableJoint ignores an idempotent command
        # without publishing another state. Require both that earlier explicit
        # acknowledgement and a live, correctly typed command subscriber.
        if not self.gazebo_attachment_topic_has_subscriber(model_name, action):
            self.gazebo_attachment_states[model_name] = None
            self.get_logger().error(
                f"Gazebo {action} subscriber unavailable or incompatible: "
                f"topic={command_topic}; attachment state UNKNOWN"
            )
            return False
        if self.gazebo_attachment_states.get(model_name) is attached:
            self.get_logger().info(
                f"Gazebo {action} confirmed for model={model_name}: "
                "retaining acknowledged state from this session; "
                "no duplicate command sent"
            )
            return True
        listener = None
        # Never reuse a previous acknowledgement across an attempted transition
        # that may fail or time out. Absence of a state message is NOT success.
        self.gazebo_attachment_states[model_name] = None
        try:
            # Subscribe before sending the command so the transition message
            # cannot be missed. Gazebo 8 publishes "attached" / "detached".
            listener = subprocess.Popen(
                ["gz", "topic", "-e", "-t", state_topic, "-d", "2.0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(0.15)
            result = subprocess.run(
                [
                    "gz", "topic", "-t", command_topic,
                    "-m", "gz.msgs.Empty", "-p", "unused: true",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if result.returncode != 0:
                details = result.stderr.strip() or result.stdout.strip()
                self.get_logger().error(
                    f"Gazebo {action} command failed for {model_name}: {details}"
                )
                return False
            output, error_output = listener.communicate(timeout=3.0)
            state = self.parse_detachable_joint_state(output)
            if state is None:
                self.get_logger().error(
                    f"Gazebo {action} unconfirmed for model={model_name}: "
                    "no state acknowledgement; subscriber presence is not "
                    "proof of attachment. Stop for manual recovery."
                )
                return False
            self.gazebo_attachment_states[model_name] = state
            if state is not attached:
                details = error_output.strip() or output.strip() or "no state response"
                self.get_logger().error(
                    f"Gazebo {action} was not confirmed for {model_name}: {details}"
                )
                return False
            self.get_logger().info(
                f"Gazebo {action} confirmed for model={model_name}, "
                "parent_link=wrist_3_link, child_link=link"
            )
            return True
        except (OSError, subprocess.SubprocessError) as error:
            self.get_logger().error(
                f"Gazebo {action} failed for {model_name}: {error}"
            )
            return False
        finally:
            if listener is not None and listener.poll() is None:
                listener.kill()
                listener.communicate()

    @staticmethod
    def collision_free_approach_z(
        object_center_z, cube_height=0.04, finger_reach=0.15, clearance=0.01
    ):
        """Lowest tool0 height that keeps the finger tips above the cube."""
        return object_center_z + cube_height / 2.0 + finger_reach + clearance

    def get_tool0_pose(self):
        """Return tool0 translation and quaternion in the planning frame."""
        deadline = time.monotonic() + 2.0
        last_error = "TF timeout"
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.frame_id,
                    self.ee_link,
                    rclpy.time.Time()
                )
                t = transform.transform.translation
                q = transform.transform.rotation
                return (t.x, t.y, t.z), (q.x, q.y, q.z, q.w)
            except Exception as error:
                last_error = error
                rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().warning(
            f"Cannot resolve TF {self.frame_id} -> {self.ee_link}: {last_error}"
        )
        return None

    def get_tool0_position(self):
        tool_pose = self.get_tool0_pose()
        return None if tool_pose is None else tool_pose[0]

    def gripper_state_callback(self, message):
        for index, name in enumerate(message.name):
            if index >= len(message.position):
                continue
            velocity = message.velocity[index] if index < len(message.velocity) else 0.0
            self.joint_states[name] = (message.position[index], velocity)
            if name not in self.gripper_joint_names:
                continue
            self.gripper_joint_states[name] = (message.position[index], velocity)

    def command_gripper(self, position, require_object=False):
        # Command velocity and close the loop on both reported joint positions.
        # Direct position control in gz_ros2_control could leave one prismatic
        # joint pinned at a limit; a velocity controller provides deterministic
        # reversal while retaining position-based success/contact checks.
        command_speed = 0.03
        joint_targets = [position, 0.043 - position]
        connection_deadline = time.monotonic() + 3.0
        while (
            rclpy.ok()
            and self.gripper_command_publisher.get_subscription_count() == 0
            and time.monotonic() < connection_deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

        subscriber_count = self.gripper_command_publisher.get_subscription_count()
        if subscriber_count == 0:
            self.get_logger().error(
                "Gripper controller is not subscribed to "
                "/simple_gripper_controller/commands; check controller spawner"
            )
            return False

        deadline = time.monotonic() + (8.0 if require_object else 5.0)

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

            if not all(
                name in self.gripper_joint_states
                for name in self.gripper_joint_names
            ):
                continue

            states = [
                self.gripper_joint_states[name]
                for name in self.gripper_joint_names
            ]
            positions = [state[0] for state in states]
            velocities = [state[1] for state in states]
            commands = []
            for value, target in zip(positions, joint_targets):
                error = target - value
                if abs(error) < 0.001:
                    commands.append(0.0)
                else:
                    commands.append(command_speed if error > 0.0 else -command_speed)
            self.gripper_command_publisher.publish(Float64MultiArray(data=commands))

            # Gripper mở
            if position == 0.0:
                if all(
                    abs(value - target) < 0.003
                    for value, target in zip(positions, joint_targets)
                ):
                    self.gripper_command_publisher.publish(
                        Float64MultiArray(data=[0.0, 0.0])
                    )
                    self.get_logger().info("Gripper opened.")
                    return True
                continue

            # Gripper đang đóng
            closure_positions = [positions[0], 0.043 - positions[1]]
            grasp_contact = (
                require_object
                and all(
                    0.005 < abs(value) < position - 0.005
                    for value in closure_positions
                )
                and abs(closure_positions[0] - closure_positions[1]) < 0.004
            )
            reached_target = all(
                abs(value - target) < 0.002
                for value, target in zip(positions, joint_targets)
            )
            if all(abs(value) < 0.01 for value in velocities) and (
                (not require_object and reached_target) or grasp_contact
            ):
                self.gripper_command_publisher.publish(
                    Float64MultiArray(data=[0.0, 0.0])
                )
                self.get_logger().info(
                    "Gripper stopped at closure positions: "
                    + ", ".join(f"{value:.3f}" for value in closure_positions)
                )
                return True

        states = {
            name: self.gripper_joint_states.get(name)
            for name in self.gripper_joint_names
        }
        if any(state is None for state in states.values()):
            self.get_logger().error(
                f"Gripper timed out at target {position:.3f}; missing joint states for "
                + ", ".join(name for name, state in states.items() if state is None)
            )
        else:
            self.get_logger().error(
                f"Gripper timed out at target {position:.3f}; "
                + ", ".join(
                    f"{name}: position={state[0]:.3f}, velocity={state[1]:.3f}"
                    for name, state in states.items()
                )
                + ", "
                f"subscribers={subscriber_count}"
            )
        self.gripper_command_publisher.publish(Float64MultiArray(data=[0.0, 0.0]))
        return False

    @staticmethod
    def moveit_error_name(error_code):
        for name in dir(MoveItErrorCodes):
            if name.isupper() and getattr(MoveItErrorCodes, name) == error_code:
                return name
        return "UNKNOWN"

    def remove_world_object(self, obj_name):
        scene = PlanningScene()
        scene.is_diff = True
        world_object = CollisionObject()
        world_object.id = obj_name
        world_object.operation = CollisionObject.REMOVE
        scene.world.collision_objects = [world_object]
        request = ApplyPlanningScene.Request(scene=scene)
        try:
            result = self.wait_result(self.scene_client.call_async(request), timeout=5.0)
            return result.success
        except Exception as error:
            self.get_logger().error(f"Failed to remove grasp target from planning scene: {error}")
            return False

    def restore_grasp_object(self, obj_name, fallback_position):
        if self.gazebo_cube_attached:
            self.set_gazebo_attachment(obj_name, attached=False)
        self.gazebo_cube_attached = False
        self.gazebo_cube_name = None
        self.grasp_world_offset = None
        tool_position = self.get_tool0_position()
        if tool_position is None:
            world_position = fallback_position
        else:
            world_position = (
                tool_position[0] + self.grasp_object_x_offset,
                tool_position[1],
                tool_position[2] - self.grasp_object_offset + self.robot_base_height,
            )
        self.set_gazebo_model_pose(obj_name, *world_position)
        restored = self.update_attached_object(
            obj_name, attached=False, world_position=world_position
        )
        self.attached_object = None
        return restored

    def remove_attached_object(self, obj_name):
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        attached_object = AttachedCollisionObject()
        attached_object.link_name = self.ee_link
        attached_object.object.id = obj_name
        attached_object.object.operation = CollisionObject.REMOVE
        scene.robot_state.attached_collision_objects = [attached_object]
        request = ApplyPlanningScene.Request(scene=scene)
        try:
            result = self.wait_result(self.scene_client.call_async(request), timeout=5.0)
            return result.success
        except Exception as error:
            self.get_logger().error(f"Failed to detach object for placement: {error}")
            return False

    def update_attached_object(self, obj_name, attached, world_position=None):
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        attached_object = AttachedCollisionObject()
        attached_object.link_name = self.ee_link
        attached_object.object.id = obj_name

        if attached:
            attached_object.object.header.frame_id = self.ee_link
            box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=self.cube_size)
            box_pose = Pose()
            box_pose.position.x = self.grasp_object_x_offset
            box_pose.position.y = 0.0
            box_pose.position.z = self.grasp_object_offset
            attached_object.object.primitives = [box]
            attached_object.object.primitive_poses = [box_pose]
            attached_object.object.operation = CollisionObject.ADD
            attached_object.touch_links = [
                "simple_gripper_base_link",
                "simple_gripper_left_finger_link",
                "simple_gripper_right_finger_link",
            ]
        else:
            attached_object.object.operation = CollisionObject.REMOVE
            world_object = CollisionObject()
            world_object.id = obj_name
            world_object.header.frame_id = self.frame_id
            world_object.operation = CollisionObject.ADD
            box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=self.cube_size)
            pose = Pose()
            pose.position.x = world_position[0]
            pose.position.y = world_position[1]
            pose.position.z = world_position[2] - self.robot_base_height
            world_object.primitives = [box]
            world_object.primitive_poses = [pose]
            scene.world.collision_objects.append(world_object)

        scene.robot_state.attached_collision_objects = [attached_object]
        request = ApplyPlanningScene.Request(scene=scene)
        try:
            result = self.wait_result(self.scene_client.call_async(request), timeout=5.0)
            if result.success:
                if attached:
                    self.get_logger().info(
                        f"MoveIt attach confirmed: object={obj_name}, "
                        f"link={self.ee_link}, touch_links="
                        f"{','.join(attached_object.touch_links)}"
                    )
                else:
                    self.get_logger().info(
                        f"MoveIt detach confirmed: object={obj_name}, world_pose="
                        f"({world_position[0]:.3f}, {world_position[1]:.3f}, "
                        f"{world_position[2]:.3f})"
                    )
            return result.success
        except Exception as error:
            self.get_logger().error(f"Failed to update planning scene: {error}")
            return False

    def wait_result(self, future, timeout=60.0):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not future.done():
            raise RuntimeError("Timeout")
        return future.result()

    def home(self):
        if not self.simulator_is_ready():
            return "SIMULATOR_NOT_READY"
        self.get_logger().info("Moving to HOME...")
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        # Joint constraints implement the package's fixed home state.
        joints = self.ARM_JOINT_NAMES
        vals = self.HOME_JOINT_VALUES

        constraints = Constraints()
        constraints.name = "home"
        for j, v in zip(joints, vals):
            jc = JointConstraint()
            jc.joint_name = j
            jc.position = v
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
            
        goal.request.goal_constraints = [constraints]
        goal.request.start_state.is_diff = True
        goal.request.allowed_planning_time = 5.0
        goal.request.num_planning_attempts = 3
        goal.request.max_velocity_scaling_factor = self.motion_scaling_factor
        goal.request.max_acceleration_scaling_factor = self.motion_scaling_factor
        
        goal_future = self.move_client.send_goal_async(goal)
        try:
            handle = self.wait_result(goal_future, timeout=10.0)
            if not handle.accepted:
                return "PLANNING_FAILED"
            result_future = handle.get_result_async()
            result = self.wait_result(result_future, timeout=30.0)
            if result.result.error_code.val == MoveItErrorCodes.SUCCESS:
                return "SUCCESS"
            else:
                self.get_logger().error(
                    f"Home MoveGroup failed with error code {result.result.error_code.val}"
                )
                return "EXECUTION_FAILED"
        except Exception as e:
            self.get_logger().error(f"Home failed: {e}")
            return "EXECUTION_FAILED"

    def move_to_pose(self, x, y, z):
        current_tool = self.get_tool0_position()
        if current_tool is None:
            self.get_logger().error(
                f"Pose goal refused: TF {self.frame_id} -> {self.ee_link} is unavailable"
            )
            return "FRAME_TRANSFORM_FAILURE"
        arm_joint_names = (
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        )
        arm_state = ", ".join(
            f"{name}={self.joint_states[name][0]:.3f}"
            for name in arm_joint_names if name in self.joint_states
        )
        self.get_logger().info(
            f"Pose goal frame={self.frame_id}, link={self.ee_link}, "
            f"target=({x:.3f}, {y:.3f}, {z - self.robot_base_height:.3f}), "
            "orientation=(1.000, 0.000, 0.000, 0.000), "
            f"current_tool=({current_tool[0]:.3f}, {current_tool[1]:.3f}, "
            f"{current_tool[2]:.3f}); joints=[{arm_state}]"
        )
        target = Pose()
        target.position.x = x
        target.position.y = y
        target.position.z = z - self.robot_base_height
        target.orientation = self.down_orientation

        region = BoundingVolume()
        region.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])]
        region.primitive_poses = [target]

        header = PoseStamped().header
        header.frame_id = self.frame_id

        position_constraint = PositionConstraint(
            header=header,
            link_name=self.ee_link,
            constraint_region=region,
            weight=1.0,
        )

        for attempt, tolerance in enumerate((0.15, 0.25)):
            orientation_constraint = OrientationConstraint(
                header=header,
                link_name=self.ee_link,
                orientation=target.orientation,
                absolute_x_axis_tolerance=tolerance,
                absolute_y_axis_tolerance=tolerance,
                absolute_z_axis_tolerance=tolerance,
                weight=1.0,
            )
            constraints = Constraints(
                position_constraints=[position_constraint],
                orientation_constraints=[orientation_constraint],
            )

            goal = MoveGroup.Goal()
            goal.request.group_name = self.group_name
            goal.request.pipeline_id = "ompl"
            goal.request.goal_constraints = [constraints]
            goal.request.start_state.is_diff = True
            goal.request.allowed_planning_time = 20.0
            goal.request.num_planning_attempts = 20
            goal.request.max_velocity_scaling_factor = self.motion_scaling_factor
            goal.request.max_acceleration_scaling_factor = self.motion_scaling_factor

            try:
                handle = self.wait_result(
                    self.move_client.send_goal_async(goal), timeout=30.0
                )
                if not handle.accepted:
                    self.get_logger().error(
                        f"MoveIt rejected pose goal at ({x:.3f}, {y:.3f}, {z:.3f})"
                    )
                    return "PLANNING_FAILED"
                result = self.wait_result(handle.get_result_async(), timeout=60.0)
                error_code = result.result.error_code.val
                if error_code == MoveItErrorCodes.SUCCESS:
                    return "SUCCESS"
                if error_code not in (
                    MoveItErrorCodes.PLANNING_FAILED,
                    MoveItErrorCodes.FAILURE,
                    MoveItErrorCodes.INVALID_MOTION_PLAN,
                ):
                    self.get_logger().error(
                        f"MoveIt failed at ({x:.3f}, {y:.3f}, {z:.3f}) "
                        f"with error code: {error_code} "
                        f"({self.moveit_error_name(error_code)})"
                    )
                    return "EXECUTION_FAILED"
                if attempt == 0:
                    self.get_logger().warn(
                        "Planning failed with 0.15 rad orientation tolerance; "
                        "retrying with 0.25 rad"
                    )
                else:
                    self.get_logger().error(
                        f"MoveIt failed at ({x:.3f}, {y:.3f}, {z:.3f}) "
                        f"with error code: {error_code} "
                        f"({self.moveit_error_name(error_code)})"
                    )
            except Exception as e:
                self.get_logger().error(f"Move failed: {e}")
                return "PLANNING_FAILED"

        self.get_logger().warn(
            "Pose planning failed; trying a collision-aware IK joint goal"
        )
        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = self.group_name
        ik_request.ik_request.robot_state.is_diff = True
        ik_request.ik_request.avoid_collisions = True
        ik_request.ik_request.ik_link_name = self.ee_link
        ik_request.ik_request.pose_stamped = PoseStamped()
        ik_request.ik_request.pose_stamped.header.frame_id = self.frame_id
        ik_request.ik_request.pose_stamped.pose = target
        ik_request.ik_request.timeout.sec = 2

        try:
            ik_result = self.wait_result(
                self.ik_client.call_async(ik_request), timeout=5.0
            )
            if ik_result.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error(
                    f"No collision-free IK solution at ({x:.3f}, {y:.3f}, {z:.3f}); "
                    f"error_code={ik_result.error_code.val} "
                    f"({self.moveit_error_name(ik_result.error_code.val)})"
                )
                return "PLANNING_FAILED"

            joint_constraints = []
            for joint_name, joint_position in zip(
                ik_result.solution.joint_state.name,
                ik_result.solution.joint_state.position,
            ):
                constraint = JointConstraint()
                constraint.joint_name = joint_name
                constraint.position = joint_position
                constraint.tolerance_above = 0.2
                constraint.tolerance_below = 0.2
                constraint.weight = 1.0
                joint_constraints.append(constraint)

            constraints = Constraints(joint_constraints=joint_constraints)
            goal = MoveGroup.Goal()
            goal.request.group_name = self.group_name
            goal.request.pipeline_id = "ompl"
            goal.request.goal_constraints = [constraints]
            goal.request.start_state.is_diff = True
            goal.request.allowed_planning_time = 20.0
            goal.request.num_planning_attempts = 10
            goal.request.max_velocity_scaling_factor = self.motion_scaling_factor
            goal.request.max_acceleration_scaling_factor = self.motion_scaling_factor

            handle = self.wait_result(
                self.move_client.send_goal_async(goal), timeout=30.0
            )
            if not handle.accepted:
                self.get_logger().error("MoveIt rejected IK-seeded joint goal")
                return "PLANNING_FAILED"
            result = self.wait_result(handle.get_result_async(), timeout=60.0)
            error_code = result.result.error_code.val
            if error_code == MoveItErrorCodes.SUCCESS:
                return "SUCCESS"
            self.get_logger().error(
                f"IK-seeded joint planning/execution failed with error_code={error_code} "
                f"({self.moveit_error_name(error_code)})"
            )
            return (
                "EXECUTION_FAILED"
                if error_code == MoveItErrorCodes.CONTROL_FAILED
                else "PLANNING_FAILED"
            )
        except Exception as e:
            self.get_logger().error(f"IK-seeded planning failed: {e}")
            return "PLANNING_FAILED"

    def wait_trajectory_result(self, future, timeout=60.0):
        """Cho trajectory chay va cap nhat vat the dang gap."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            # spin_once also services gazebo_cube_timer. Do not invoke
            # update_gazebo_cube a second time from this loop.
            rclpy.spin_once(self, timeout_sec=0.05)

        if not future.done():
            raise RuntimeError("Trajectory execution timeout")

        return future.result()

    @staticmethod
    def bounded_place_offsets(zone_width, cube_width, margin=0.002):
        """Small Y offsets whose complete cube footprint stays in the zone."""
        maximum = max(0.0, (zone_width - cube_width) / 2.0 - margin)
        preferred = min(0.015, maximum)
        if preferred < 0.005:
            return [0.0]
        return [0.0, preferred, -preferred]

    @staticmethod
    def first_unreached_cartesian_point(start, target, fraction, max_step):
        """Estimate the first rejected interpolation sample from MoveIt's fraction."""
        distance = sum((b - a) ** 2 for a, b in zip(start, target)) ** 0.5
        if distance <= 1e-9:
            return target
        sample_count = max(1, math.ceil(distance / max_step - 1e-9))
        first_rejected_index = min(
            sample_count, max(1, int(fraction * sample_count) + 1)
        )
        ratio = first_rejected_index / sample_count
        return tuple(a + ratio * (b - a) for a, b in zip(start, target))

    def diagnose_place_pose(self, x, y, z, label):
        """Separate geometric IK failure from collision-invalid final states."""
        target = PoseStamped()
        target.header.frame_id = self.frame_id
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z - self.robot_base_height
        target.pose.orientation = self.down_orientation

        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = self.group_name
        ik_request.ik_request.robot_state.is_diff = True
        ik_request.ik_request.avoid_collisions = False
        ik_request.ik_request.ik_link_name = self.ee_link
        ik_request.ik_request.pose_stamped = target
        ik_request.ik_request.timeout.sec = 2
        try:
            ik_result = self.wait_result(
                self.ik_client.call_async(ik_request), timeout=5.0
            )
        except Exception as error:
            self.get_logger().error(f"{label}: IK diagnostic service failed: {error}")
            return False

        if ik_result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().warning(
                f"{label}: no geometric IK solution; error_code="
                f"{ik_result.error_code.val} "
                f"({self.moveit_error_name(ik_result.error_code.val)})"
            )
            return False

        collision_aware_request = GetPositionIK.Request()
        collision_aware_request.ik_request.group_name = self.group_name
        collision_aware_request.ik_request.robot_state.is_diff = True
        collision_aware_request.ik_request.avoid_collisions = True
        collision_aware_request.ik_request.ik_link_name = self.ee_link
        collision_aware_request.ik_request.pose_stamped = target
        collision_aware_request.ik_request.timeout.sec = 2
        try:
            collision_aware_result = self.wait_result(
                self.ik_client.call_async(collision_aware_request), timeout=5.0
            )
        except Exception as error:
            self.get_logger().error(
                f"{label}: collision-aware IK service failed: {error}"
            )
            return False

        if collision_aware_result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(
                f"{label}: final pose has geometric IK and collision-aware IK"
            )
            return True

        # Show contacts for the geometric solution as a diagnostic clue. This
        # is not used to claim that every IK branch has the same collision.
        validity_request = GetStateValidity.Request()
        validity_request.robot_state = ik_result.solution
        validity_request.robot_state.is_diff = True
        validity_request.group_name = self.group_name
        try:
            validity = self.wait_result(
                self.state_validity_client.call_async(validity_request), timeout=5.0
            )
        except Exception as error:
            self.get_logger().error(
                f"{label}: state-validity diagnostic service failed: {error}"
            )
            return False

        if not validity.valid:
            contacts = sorted(
                {
                    f"{contact.contact_body_1}<->{contact.contact_body_2}"
                    for contact in validity.contacts
                }
            )
            details = ", ".join(contacts[:8]) or "contacts not reported"
            self.get_logger().warning(
                f"{label}: geometric IK exists, but collision-aware IK failed "
                f"with error_code={collision_aware_result.error_code.val} "
                f"({self.moveit_error_name(collision_aware_result.error_code.val)}); "
                f"representative rejected branch contacts: {details}"
            )
            return False

        self.get_logger().warning(
            f"{label}: collision-aware IK failed with error_code="
            f"{collision_aware_result.error_code.val} "
            f"({self.moveit_error_name(collision_aware_result.error_code.val)})"
        )
        return False

    def move_to_verified_place_pose(self, x, y, z, label):
        """Collision-aware non-Cartesian fallback with strict pose verification."""
        target = Pose()
        target.position.x = x
        target.position.y = y
        target.position.z = z - self.robot_base_height
        target.orientation = self.down_orientation

        header = PoseStamped().header
        header.frame_id = self.frame_id
        region = BoundingVolume()
        region.primitives = [
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.004])
        ]
        region.primitive_poses = [target]
        constraints = Constraints(
            position_constraints=[
                PositionConstraint(
                    header=header,
                    link_name=self.ee_link,
                    constraint_region=region,
                    weight=1.0,
                )
            ],
            orientation_constraints=[
                OrientationConstraint(
                    header=header,
                    link_name=self.ee_link,
                    orientation=target.orientation,
                    absolute_x_axis_tolerance=0.05,
                    absolute_y_axis_tolerance=0.05,
                    absolute_z_axis_tolerance=0.05,
                    weight=1.0,
                )
            ],
        )
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.pipeline_id = "ompl"
        goal.request.goal_constraints = [constraints]
        goal.request.start_state.is_diff = True
        goal.request.allowed_planning_time = 20.0
        goal.request.num_planning_attempts = 20
        goal.request.max_velocity_scaling_factor = self.motion_scaling_factor
        goal.request.max_acceleration_scaling_factor = self.motion_scaling_factor
        try:
            handle = self.wait_result(
                self.move_client.send_goal_async(goal), timeout=30.0
            )
            if not handle.accepted:
                self.get_logger().error(f"Pose fallback rejected for {label}")
                return "PLANNING_FAILED"
            result = self.wait_result(handle.get_result_async(), timeout=60.0)
            error_code = result.result.error_code.val
            if error_code != MoveItErrorCodes.SUCCESS:
                self.get_logger().error(
                    f"Pose fallback failed for {label}: error_code={error_code} "
                    f"({self.moveit_error_name(error_code)})"
                )
                return (
                    "EXECUTION_FAILED"
                    if error_code == MoveItErrorCodes.CONTROL_FAILED
                    else "PLANNING_FAILED"
                )
        except Exception as error:
            self.get_logger().error(f"Pose fallback failed for {label}: {error}")
            return "PLANNING_FAILED"

        actual = self.get_tool0_pose()
        if actual is None:
            return "FRAME_TRANSFORM_FAILURE"
        position_error = math.sqrt(
            (actual[0][0] - target.position.x) ** 2
            + (actual[0][1] - target.position.y) ** 2
            + (actual[0][2] - target.position.z) ** 2
        )
        orientation_dot = abs(
            actual[1][0] * target.orientation.x
            + actual[1][1] * target.orientation.y
            + actual[1][2] * target.orientation.z
            + actual[1][3] * target.orientation.w
        )
        orientation_dot = min(1.0, orientation_dot)
        orientation_error = 2.0 * math.acos(orientation_dot)
        if position_error > 0.008 or orientation_error > 0.08:
            self.get_logger().error(
                f"Pose fallback verification failed for {label}: "
                f"position_error={position_error:.4f} m, "
                f"orientation_error={orientation_error:.4f} rad; gripper stays closed"
            )
            return "EXECUTION_FAILED"
        self.get_logger().info(
            f"Pose fallback verified for {label}: position_error="
            f"{position_error:.4f} m, orientation_error={orientation_error:.4f} rad"
        )
        return "SUCCESS"

    def move_cartesian(self, waypoints, diagnostic_label="Cartesian path"):
        start_position = self.get_tool0_position()
        request = GetCartesianPath.Request()
        request.header.frame_id = self.frame_id
        request.header.stamp = self.get_clock().now().to_msg()
        request.start_state.is_diff = True
        request.group_name = self.group_name
        request.link_name = self.ee_link
        request.max_step = 0.005
        request.jump_threshold = 0.0
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = self.motion_scaling_factor
        request.max_acceleration_scaling_factor = self.motion_scaling_factor

        for x, y, z in waypoints:
            waypoint = Pose()
            waypoint.position.x = x
            waypoint.position.y = y
            waypoint.position.z = z - self.robot_base_height
            waypoint.orientation = self.down_orientation
            request.waypoints.append(waypoint)

        try:
            path = self.wait_result(
                self.cartesian_client.call_async(request), timeout=30.0
            )
            if path.error_code.val != MoveItErrorCodes.SUCCESS or path.fraction < 1.0:
                failure_detail = ""
                if start_position is not None and len(waypoints) == 1:
                    target = (
                        waypoints[0][0],
                        waypoints[0][1],
                        waypoints[0][2] - self.robot_base_height,
                    )
                    failed = self.first_unreached_cartesian_point(
                        start_position, target, path.fraction, request.max_step
                    )
                    failure_detail = (
                        "; estimated_first_rejected_waypoint="
                        f"({failed[0]:.3f}, {failed[1]:.3f}, {failed[2]:.3f})"
                    )
                self.get_logger().error(
                    f"{diagnostic_label} incomplete: fraction={path.fraction:.3f}, "
                    f"error_code={path.error_code.val} "
                    f"({self.moveit_error_name(path.error_code.val)})"
                    f"{failure_detail}; trajectory was not executed"
                )
                return "PLANNING_FAILED"

            goal = ExecuteTrajectory.Goal()
            goal.trajectory = path.solution
            goal_future = self.execute_client.send_goal_async(goal)
            handle = self.wait_result(goal_future, timeout=30.0)
            if not handle.accepted:
                self.get_logger().error("MoveIt rejected Cartesian trajectory")
                return "PLANNING_FAILED"

            result = self.wait_trajectory_result(
                handle.get_result_async(), timeout=60.0
            )
            if result.result.error_code.val == MoveItErrorCodes.SUCCESS:
                return "SUCCESS"
            self.get_logger().error(
                f"Cartesian execution failed with error code "
                f"{result.result.error_code.val}"
            )
            return "EXECUTION_FAILED"
        except Exception as e:
            self.get_logger().error(f"Cartesian motion failed: {e}")
            return "PLANNING_FAILED"

    def pick(self, obj_name, object_position=None):
        if not self.simulator_is_ready():
            return "SIMULATOR_NOT_READY"
        if self.attached_object is not None:
            self.get_logger().error(
                f"Cannot pick {obj_name}: {self.attached_object} is still held; place it first"
            )
            return "OBJECT_STILL_ATTACHED"
        if obj_name not in self.scene.get('objects', {}):
            return "INVALID_OBJECT"
            
        if object_position is None:
            object_position = self.get_gazebo_model_pose(obj_name)
            if object_position is None:
                return "SIMULATOR_STATE_UNAVAILABLE"
            source = 'Gazebo (legacy skill_executor)'
        else:
            source = 'camera /vision/environment_state'
        if len(object_position) != 3 or not all(math.isfinite(v) for v in object_position):
            return "INVALID_OBJECT_POSITION"
        x, y, z = object_position
        self.get_logger().info(f"Using {source} for {obj_name}: ({x:.3f}, {y:.3f}, {z:.3f})")
        
        self.get_logger().info("Opening gripper...")
        if not self.command_gripper(0.0):
            return "EXECUTION_FAILED"

        self.get_logger().info(f"Moving above {obj_name}...")
        grasp_z = z + self.grasp_object_offset
        # Keep the complete finger collision geometry above the cube.  The old
        # grasp_z + 0.02 target put the 0.15 m finger tips at z=0.23 while the
        # cube top is z=0.24, which MoveIt correctly rejected as a collision.
        approach_z = self.collision_free_approach_z(
            z, self.cube_height, self.finger_reach, self.approach_clearance
        )
        result = self.move_to_pose(x, y, approach_z)
        if result != "SUCCESS":
            return result

        if not self.remove_world_object(obj_name):
            return "EXECUTION_FAILED"
            
        self.get_logger().info(f"Lowering to grasp {obj_name}...")
        result = self.move_cartesian([(x, y, grasp_z)])
        if result != "SUCCESS":
            self.update_attached_object(
                obj_name,
                attached=False,
                world_position=(x, y, z),
            )
            return result
            
        self.get_logger().info(f"Closing gripper on {obj_name}...")
        if not self.command_gripper(0.043, require_object=True):
            self.update_attached_object(
                obj_name,
                attached=False,
                world_position=(x, y, z),
            )
            return "EXECUTION_FAILED"
        if getattr(self, 'preserve_failed_grasp', False):
            # Reserve the held state as soon as physical contact is accepted;
            # failures in plugin/scene acknowledgement must not enable a new pick.
            self.attached_object = obj_name
        # Finger contact alone is not proof of grasp. Create a real fixed joint
        # in Gazebo and require its state acknowledgement before MoveIt is told
        # that the object is attached or any lift begins.
        tool_position = self.get_tool0_position()
        contact_object_position = self.get_gazebo_model_pose(obj_name)
        if tool_position is None or contact_object_position is None:
            self.get_logger().error(
                f"Cannot attach {obj_name}: contact pose is unavailable"
            )
            self.update_attached_object(
                obj_name, attached=False, world_position=(x, y, z)
            )
            return "GRASP_FAILED"
        # DetachableJoint starts in the attached state. Loading it only here is
        # essential: loading one for every cube at launch mechanically ties
        # objects on the table to the wrist and makes the arm controller fail.
        if not self.load_gazebo_attachment_plugin(obj_name):
            self.update_attached_object(
                obj_name, attached=False, world_position=(x, y, z)
            )
            return "GRASP_FAILED"
        self.grasp_world_offset = tuple(
            obj_value - tool_value
            for obj_value, tool_value in zip(
                contact_object_position, tool_position
            )
        )
        self.gazebo_cube_name = obj_name
        self.gazebo_cube_attached = True
        self.attached_object = obj_name
        if getattr(self, 'preserve_failed_grasp', False) and not self.set_gazebo_attachment(obj_name, attached=True):
            return 'GAZEBO_ATTACH_UNCONFIRMED'
        if not self.update_attached_object(obj_name, attached=True):
            if not getattr(self, 'preserve_failed_grasp', False):
                self.restore_grasp_object(obj_name, (x, y, z))
            return "EXECUTION_FAILED"
        self.attached_object = obj_name
        self.get_logger().info(f"Lifting {obj_name}...")
        result = self.move_cartesian([(x, y, approach_z)])
        if result != "SUCCESS":
            if not getattr(self, 'preserve_failed_grasp', False):
                self.restore_grasp_object(obj_name, (x, y, z))
            return result

        # The fixed-joint acknowledgement is still not enough: verify the
        # simulator pose after lift so a wrong model/link cannot pass as grasp.
        tool_position = self.get_tool0_position()
        actual_object_position = self.get_gazebo_model_pose(obj_name)
        expected_object_position = (
            None
            if tool_position is None or self.grasp_world_offset is None
            else tuple(
                tool_value + offset
                for tool_value, offset in zip(
                    tool_position, self.grasp_world_offset
                )
            )
        )
        if (
            actual_object_position is None
            or expected_object_position is None
            or max(
                abs(actual - expected)
                for actual, expected in zip(
                    actual_object_position, expected_object_position
                )
            ) > 0.015
        ):
            self.get_logger().error(
                f"Grasp verification failed for {obj_name}: "
                f"actual={actual_object_position}, expected={expected_object_position}"
            )
            if not getattr(self, 'preserve_failed_grasp', False):
                self.restore_grasp_object(obj_name, (x, y, z))
            return "GRASP_FAILED"

        return "SUCCESS"

    def place(self, obj_name, zone_name):
        if not self.simulator_is_ready():
            return "SIMULATOR_NOT_READY"
        if obj_name not in self.scene.get('objects', {}):
            return "INVALID_OBJECT"
        if zone_name not in self.scene.get('zones', {}):
            return "INVALID_ZONE"
            
        if self.attached_object != obj_name:
            self.get_logger().error(f"Cannot place {obj_name}: it is not attached")
            return "NOT_ATTACHED"
        occupant = self.get_occupied_zones().get(zone_name)
        if occupant is not None and occupant != obj_name:
            self.get_logger().error(
                f"Cannot place in {zone_name}: it already contains {occupant}"
            )
            return "ZONE_OCCUPIED"
            
        pos = self.scene['zones'][zone_name]['position']
        center = (pos[0], pos[1], pos[2] + self.cube_height / 2.0)
        return self.place_at(obj_name, center, zone_name,
                             offsets=self.bounded_place_offsets(self.zone_usable_size[1], self.cube_size[1]))

    def verify_tool_pose(self, x, y, z):
        """Check TF arrival before releasing a held object, even after Cartesian SUCCESS."""
        actual = self.get_tool0_pose()
        if actual is None or not all(math.isfinite(v) for part in actual for v in part):
            return False
        error = math.dist(actual[0], (x, y, z - self.robot_base_height))
        dot = min(1.0, abs(sum(a*b for a, b in zip(actual[1],
                        (self.down_orientation.x, self.down_orientation.y,
                         self.down_orientation.z, self.down_orientation.w)))))
        return error <= 0.008 and 2*math.acos(dot) <= 0.08

    def place_at(self, obj_name, cube_center, zone_name, offsets=None,
                 approach_clearance=0.05, release_clearance=0.005):
        """Shared physical release implementation; coordinates are cube centre in world."""
        if self.attached_object != obj_name:
            return 'NOT_ATTACHED'
        x, y, z = cube_center

        # Keep the attached cube 5 mm above the support during the final
        # collision-checked descent. At the old exact-contact height MoveIt
        # stopped at 90% of the Cartesian path.
        release_z = z + self.grasp_object_offset + release_clearance
        approach_z = release_z + approach_clearance
        table = self.scene['table']
        table_top = table['center'][2] + table['size'][2] / 2.0
        tool_pose = self.get_tool0_pose()
        if tool_pose is None:
            return "FRAME_TRANSFORM_FAILURE"
        offsets = [0.0] if offsets is None else offsets
        self.get_logger().info(
            f"Place geometry: frame={self.frame_id}, link={self.ee_link}, "
            f"current_tool=({tool_pose[0][0]:.3f}, {tool_pose[0][1]:.3f}, "
            f"{tool_pose[0][2]:.3f}), current_orientation="
            f"({tool_pose[1][0]:.3f}, {tool_pose[1][1]:.3f}, "
            f"{tool_pose[1][2]:.3f}, {tool_pose[1][3]:.3f}), "
            f"pre_place=({x:.3f}, {y:.3f}, {approach_z:.3f}), "
            f"final_place=({x:.3f}, {y:.3f}, {release_z:.3f}), "
            f"zone_marker={self.zone_marker_size[0]:.3f}x"
            f"{self.zone_marker_size[1]:.3f} m, usable_zone="
            f"{self.zone_usable_size[0]:.3f}x{self.zone_usable_size[1]:.3f} m, "
            f"cube={self.cube_size[0]:.3f}x{self.cube_size[1]:.3f}x"
            f"{self.cube_size[2]:.3f} m, table_top={table_top:.3f}, "
            f"tool0_to_cube_center={self.grasp_object_offset:.3f}, "
            f"finger_reach={self.finger_reach:.3f}"
        )

        selected_y = None
        last_result = "PLANNING_FAILED"
        for offset in offsets:
            candidate_y = y + offset
            label = f"{zone_name} candidate y_offset={offset:+.3f} m"
            half_zone = self.zone_usable_size[1] / 2.0
            half_cube = self.cube_size[1] / 2.0
            self.get_logger().info(
                f"Checking {label}; cube_edge={abs(offset) + half_cube:.3f} "
                f"<= usable_half_width={half_zone:.3f}"
            )
            if not self.diagnose_place_pose(x, candidate_y, release_z, label):
                continue

            self.get_logger().info(f"Moving above {label}...")
            last_result = self.move_cartesian(
                [(x, candidate_y, approach_z)],
                diagnostic_label=f"Approach to {label}",
            )
            if last_result == "PLANNING_FAILED":
                self.get_logger().warning(
                    f"Cartesian approach failed for {label}; trying a strict "
                    "collision-aware pose goal and verifying the pre-place pose"
                )
                last_result = self.move_to_verified_place_pose(
                    x, candidate_y, approach_z, f"pre-place {label}"
                )
            if last_result != "SUCCESS":
                if last_result != "PLANNING_FAILED":
                    self.get_logger().error(
                        f"Approach execution state is uncertain for {label}; "
                        f"{obj_name} remains attached and no other candidate will be tried"
                    )
                    return last_result
                continue

            self.get_logger().info(f"Lowering to {label}...")
            last_result = self.move_cartesian(
                [(x, candidate_y, release_z)],
                diagnostic_label=f"Descent to {label}",
            )
            if last_result == "PLANNING_FAILED":
                self.get_logger().warning(
                    f"Cartesian descent failed for {label}; trying a strict "
                    "collision-aware pose goal and verifying the reached pose"
                )
                last_result = self.move_to_verified_place_pose(
                    x, candidate_y, release_z, label
                )
            if last_result not in ("SUCCESS", "PLANNING_FAILED"):
                self.get_logger().error(
                    f"Descent execution state is uncertain for {label}; "
                    f"{obj_name} remains attached and no other candidate will be tried"
                )
                return last_result
            if last_result == "SUCCESS":
                selected_y = candidate_y
                if offset != 0.0:
                    self.get_logger().warning(
                        f"Selected in-zone placement offset for {zone_name}: "
                        f"y_offset={offset:+.3f} m"
                    )
                break

        if selected_y is None:
            self.get_logger().error(
                f"No complete collision-aware descent found for {zone_name}; "
                f"{obj_name} remains closed, attached in Gazebo and attached "
                "in MoveIt. New picks remain blocked."
            )
            return last_result
        if not self.verify_tool_pose(x, selected_y, release_z):
            self.get_logger().error('Final place pose not reached; gripper stays closed and object attached')
            return 'PLACE_POSE_NOT_REACHED'
            
        self.get_logger().info(f"Opening gripper to release {obj_name}...")
        placed_position = [x, selected_y, z]

        # Release the physical simulator joint first. The cube remains between
        # the closed fingers at the support pose while the gripper opens.
        if not self.set_gazebo_attachment(obj_name, attached=False):
            self.get_logger().error(
                f"Release aborted: Gazebo did not detach {obj_name}; "
                "object remains attached"
            )
            return "EXECUTION_FAILED"
        self.gazebo_cube_attached = False

        if not self.command_gripper(0.0):
            # Attempt to restore the physical joint and retain conservative
            # internal / MoveIt attachment state, blocking subsequent picks.
            self.gazebo_cube_attached = self.set_gazebo_attachment(
                obj_name, attached=True
            )
            self.get_logger().error(
                f"Release failed for {obj_name}; object remains attached and "
                "new pick commands are blocked"
            )
            return "GRIPPER_RELEASE_FAILED"

        time.sleep(0.3)
        actual_placed_position = self.get_gazebo_model_pose(obj_name)
        if actual_placed_position is None:
            self.get_logger().error(
                f"Release verification failed for {obj_name}: pose unavailable"
            )
            return "EXECUTION_FAILED"
        placed_position = list(actual_placed_position)

        if not self.update_attached_object(
            obj_name, attached=False, world_position=placed_position
        ):
            self.get_logger().error(
                f"Gripper opened but MoveIt could not detach {obj_name}; "
                "retaining conservative attached state"
            )
            return "EXECUTION_FAILED"
        self.attached_object = None
        self.gazebo_cube_name = None
        self.grasp_world_offset = None
        self.scene['objects'][obj_name]['position'] = placed_position

        self.get_logger().info("Retreating...")
        retreat_result = self.move_cartesian(
            [(x, selected_y, approach_z)],
            diagnostic_label=f"Retreat from {zone_name}",
        )
        return retreat_result

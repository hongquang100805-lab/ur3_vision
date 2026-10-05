import os

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    CollisionObject,
    ObjectColor,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
import rclpy
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from .scene_config import load_scene


class ScenePublisher(Node):
    def __init__(self):
        super().__init__('scene_publisher')
        self.publisher = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.get_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
        self.scene_request_pending = False
        
        pkg_share = get_package_share_directory('ur3_vision_planning')
        scene_file = os.path.join(pkg_share, 'config', 'scene.yaml')
        
        self.declare_parameter('scenario', 'default')
        self.scene = load_scene(scene_file, self.get_parameter('scenario').value)
        self.robot_base_height = self.scene.get('robot_base_height', 0.0)
        self.cube_size = self.scene.get('cube_size', [0.04, 0.04, 0.04])
            
        self.timer = self.create_timer(2.0, self.publish_scene)
        self.get_logger().info('Scene publisher started')

    def publish_scene(self):
        if (
            self.scene_request_pending
            or not self.scene_client.service_is_ready()
            or not self.get_scene_client.service_is_ready()
        ):
            return
        scene_msg = PlanningScene()
        scene_msg.is_diff = True
        scene_msg.robot_state.is_diff = True
        
        # 1. Add Table
        table = CollisionObject()
        table.id = 'table'
        table.header.frame_id = 'base_link'
        table.operation = CollisionObject.ADD
        
        table_box = SolidPrimitive()
        table_box.type = SolidPrimitive.BOX
        table_config = self.scene['table']
        table_box.dimensions = table_config['size']
        
        table_pose = Pose()
        table_pose.position.x = table_config['center'][0]
        table_pose.position.y = table_config['center'][1]
        table_pose.position.z = table_config['center'][2] - self.robot_base_height
        
        table.primitives.append(table_box)
        table.primitive_poses.append(table_pose)
        scene_msg.world.collision_objects.append(table)
        
        table_color = ObjectColor()
        table_color.id = 'table'
        table_color.color.r = 0.35
        table_color.color.g = 0.35
        table_color.color.b = 0.38
        table_color.color.a = 1.0
        scene_msg.object_colors.append(table_color)

        # 2. Add Cubes
        cube_colors = {
            'red_cube': (1.0, 0.1, 0.1),
            'yellow_cube': (1.0, 0.9, 0.0),
            'blue_cube': (0.1, 0.4, 1.0)
        }
        
        for obj_name, obj_data in self.scene.get('objects', {}).items():
            obj = CollisionObject()
            obj.id = obj_name
            obj.header.frame_id = 'base_link'
            obj.operation = CollisionObject.ADD
            
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = self.cube_size
            
            pose = Pose()
            pos = obj_data['position']
            pose.position.x = pos[0]
            pose.position.y = pos[1]
            pose.position.z = pos[2] - self.robot_base_height
            
            obj.primitives.append(box)
            obj.primitive_poses.append(pose)
            scene_msg.world.collision_objects.append(obj)
            
            color = ObjectColor()
            color.id = obj_name
            c = obj_data.get('color', cube_colors.get(obj_name, (1.0, 1.0, 1.0)))
            color.color.r = c[0]
            color.color.g = c[1]
            color.color.b = c[2]
            color.color.a = 1.0
            scene_msg.object_colors.append(color)

        self.allowed_collision_pairs = {
            frozenset(("wrist_3_link", "simple_gripper_base_link")),
            frozenset(("simple_gripper_base_link", "simple_gripper_left_finger_link")),
            frozenset(("simple_gripper_base_link", "simple_gripper_right_finger_link")),
        }
        self.scene_request_pending = True
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        future = self.get_scene_client.call_async(request)
        future.add_done_callback(lambda result: self.apply_initial_scene(result, scene_msg))

    def apply_initial_scene(self, future, scene_msg):
        try:
            matrix = future.result().scene.allowed_collision_matrix
            original_names = list(matrix.entry_names)
            original_count = len(original_names)
            rows = [list(entry.enabled) for entry in matrix.entry_values]
            new_names = sorted(
                {link for pair in self.allowed_collision_pairs for link in pair}
                - set(original_names)
            )
            for row in rows:
                row.extend([False] * len(new_names))
            for _ in new_names:
                rows.append([False] * (original_count + len(new_names)))

            names = original_names + new_names
            indices = {name: index for index, name in enumerate(names)}
            for pair in self.allowed_collision_pairs:
                first, second = (indices[name] for name in pair)
                rows[first][second] = True
                rows[second][first] = True

            matrix.entry_names = names
            matrix.entry_values = [AllowedCollisionEntry(enabled=row) for row in rows]
            scene_msg.allowed_collision_matrix = matrix
            request = ApplyPlanningScene.Request(scene=scene_msg)
            apply_future = self.scene_client.call_async(request)
            apply_future.add_done_callback(self.scene_applied)
        except Exception as error:
            self.scene_request_pending = False
            self.get_logger().warning(f'Failed to extend MoveIt collision matrix: {error}')

    def scene_applied(self, future):
        self.scene_request_pending = False
        try:
            if future.result().success:
                self.timer.cancel()
                self.get_logger().info('Initial planning scene applied')
            else:
                self.get_logger().warning('MoveIt rejected the initial planning scene; retrying')
        except Exception as error:
            self.get_logger().warning(f'Failed to apply initial planning scene: {error}')


def main(args=None):
    rclpy.init(args=args)
    node = ScenePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

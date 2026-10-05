import pytest

from ur3_vision_planning.robot_skills import RobotSkills


def test_collision_free_approach_clears_cube_and_fingers():
    approach_z = RobotSkills.collision_free_approach_z(0.22)

    assert approach_z == pytest.approx(0.40)
    assert approach_z - 0.15 == pytest.approx(0.22 + 0.02 + 0.01)


def test_parse_gazebo_model_pose():
    output = """Requesting state...
Pose [ XYZ (m) ] [ RPY (rad) ]:
  [0.380000 -0.150000 0.260000]
  [0.000000 0.000000 0.000000]
"""

    assert RobotSkills.parse_gazebo_model_pose(output) == pytest.approx(
        (0.38, -0.15, 0.26)
    )


def test_parse_gazebo_model_pose_rejects_non_pose_output():
    assert RobotSkills.parse_gazebo_model_pose("service unavailable") is None


def test_parse_detachable_joint_state_uses_latest_state():
    output = 'data: "detached"\n---\ndata: "attached"\n'
    assert RobotSkills.parse_detachable_joint_state(output) is True


def test_parse_detachable_joint_state_accepts_boolean_compatibility():
    output = "data: false\n---\ndata: true\n"
    assert RobotSkills.parse_detachable_joint_state(output) is True


def test_parse_detachable_joint_state_rejects_missing_boolean():
    assert RobotSkills.parse_detachable_joint_state("no publisher") is None


def test_place_offsets_keep_entire_cube_inside_usable_zone():
    offsets = RobotSkills.bounded_place_offsets(0.075, 0.04)

    assert offsets == pytest.approx([0.0, 0.015, -0.015])
    assert all(abs(offset) + 0.04 / 2.0 <= 0.075 / 2.0 for offset in offsets)


def test_first_unreached_cartesian_point_uses_fraction_and_step():
    point = RobotSkills.first_unreached_cartesian_point(
        (0.28, 0.0, 0.436), (0.28, 0.0, 0.386), 0.1, 0.005
    )

    assert point == pytest.approx((0.28, 0.0, 0.426))

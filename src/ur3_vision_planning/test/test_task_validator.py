import json

import pytest

from ur3_vision_planning.task_validator import TaskValidator


@pytest.fixture
def validator():
    return TaskValidator()


def validate(validator, steps, occupied=None):
    return validator.validate(json.dumps({"plan": steps}), occupied_zones=occupied)


def test_valid_single_object_plan(validator):
    valid, message = validate(
        validator,
        [
            {"skill": "pick", "object": "red_cube"},
            {"skill": "place", "object": "red_cube", "zone": "zone_b"},
            {"skill": "home"},
        ],
    )
    assert valid
    assert message == "PLAN VALID"


def test_valid_personalized_multi_object_plan(validator):
    steps = []
    for obj, zone in (
        ("yellow_cube", "zone_a"),
        ("blue_cube", "zone_b"),
        ("red_cube", "zone_c"),
    ):
        steps.extend(
            (
                {"skill": "pick", "object": obj},
                {"skill": "place", "object": obj, "zone": zone},
            )
        )
    steps.append({"skill": "home"})
    assert validate(validator, steps)[0]


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ("not json", "JSON_ERROR"),
        ({"plan": "not a list"}, "INVALID_PLAN"),
        ({"plan": [{"skill": "dance"}]}, "INVALID_SKILL"),
        ({"plan": [{"skill": "pick", "object": "green_cube"}]}, "INVALID_OBJECT"),
        (
            {
                "plan": [
                    {"skill": "pick", "object": "red_cube"},
                    {"skill": "place", "object": "red_cube", "zone": "zone_d"},
                ]
            },
            "INVALID_ZONE",
        ),
    ],
)
def test_rejects_invalid_json_schema_and_values(validator, payload, error):
    valid, message = validator.validate(payload)
    assert not valid
    assert message.startswith(error)


@pytest.mark.parametrize(
    "steps",
    [
        [{"skill": "place", "object": "red_cube", "zone": "zone_a"}, {"skill": "home"}],
        [
            {"skill": "pick", "object": "red_cube"},
            {"skill": "pick", "object": "blue_cube"},
            {"skill": "home"},
        ],
        [
            {"skill": "pick", "object": "red_cube"},
            {"skill": "place", "object": "blue_cube", "zone": "zone_a"},
            {"skill": "home"},
        ],
        [{"skill": "pick", "object": "red_cube"}, {"skill": "home"}],
        [{"skill": "home"}, {"skill": "pick", "object": "red_cube"}],
    ],
)
def test_rejects_invalid_order(validator, steps):
    valid, message = validate(validator, steps)
    assert not valid
    assert message.startswith("INVALID_ORDER")


def test_rejects_extra_parameters(validator):
    valid, message = validate(validator, [{"skill": "home", "object": "red_cube"}])
    assert not valid
    assert message.startswith("INVALID_PLAN")


def test_rejects_occupied_zone(validator):
    valid, message = validate(
        validator,
        [
            {"skill": "pick", "object": "red_cube"},
            {"skill": "place", "object": "red_cube", "zone": "zone_b"},
            {"skill": "home"},
        ],
        occupied={"zone_b": "blue_cube"},
    )
    assert not valid
    assert message == "ZONE_OCCUPIED: zone_b already contains blue_cube"


def test_can_free_an_occupied_zone_earlier_in_plan(validator):
    valid, _ = validate(
        validator,
        [
            {"skill": "pick", "object": "blue_cube"},
            {"skill": "place", "object": "blue_cube", "zone": "zone_c"},
            {"skill": "pick", "object": "red_cube"},
            {"skill": "place", "object": "red_cube", "zone": "zone_b"},
            {"skill": "home"},
        ],
        occupied={"zone_b": "blue_cube"},
    )
    assert valid

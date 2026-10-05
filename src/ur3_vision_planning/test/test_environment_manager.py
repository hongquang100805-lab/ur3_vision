from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ur3_vision_planning.environment_manager import (
    EnvironmentError, EnvironmentManager, EnvironmentPlanner, parse_json,
)


@pytest.fixture
def planner():
    config = Path(__file__).resolve().parents[1] / 'config'
    def load(name):
        return yaml.safe_load((config / name).read_text())
    return EnvironmentPlanner(load('scene.yaml'), load('temporary_positions.yaml'),
                              load('camera.yaml')['camera'])


@pytest.fixture
def state():
    # Independent camera-shaped observations; not generated from spawn YAML.
    positions = {'red_cube': (0.38, -0.15), 'yellow_cube': (0.38, 0),
                 'blue_cube': (0.28, 0), 'green_cube': (0.38, -0.075),
                 'purple_cube': (0.38, 0.075)}
    return {
        'stamp': 10.0, 'published_at': 10.1, 'frame_id': 'base_link',
        'status': 'ok', 'stale': False,
        'objects': {name: {'detected': True, 'stale': False, 'age_s': 0.1,
                           'stamp': 10.0, 'position': {'x': x, 'y': y, 'z': 0.22},
                           'zone': 'zone_b' if name == 'blue_cube' else None}
                    for name, (x, y) in positions.items()},
        'zones': {zone: {'occupied': zone == 'zone_b',
                        'object': 'blue_cube' if zone == 'zone_b' else None,
                        'objects': ['blue_cube'] if zone == 'zone_b' else [],
                        'stale': False, 'status': 'occupied' if zone == 'zone_b' else 'empty'}
                  for zone in ('zone_a', 'zone_b', 'zone_c')},
    }


def test_occupied_zone_moves_occupant_before_target(planner, state):
    plan = planner.resolve_goal('red_cube', 'zone_b', state)
    assert plan['plan'] == [
        {'skill': 'pick', 'object': 'blue_cube'},
        {'skill': 'place', 'object': 'blue_cube', 'destination': 'temporary_1'},
        {'skill': 'pick', 'object': 'red_cube'},
        {'skill': 'place', 'object': 'red_cube', 'destination': 'zone_b'},
        {'skill': 'home'},
    ]
    assert plan['temporary_positions']['temporary_1']['x'] == 0.43


def test_already_satisfied_and_empty_zone(planner, state):
    assert planner.resolve_goal('blue_cube', 'zone_b', state) == {
        'plan': [{'skill': 'home'}], 'temporary_positions': {}}
    assert planner.resolve_goal('green_cube', 'zone_a', state)['plan'] == [
        {'skill': 'pick', 'object': 'green_cube'},
        {'skill': 'place', 'object': 'green_cube', 'destination': 'zone_a'},
        {'skill': 'home'},
    ]


@pytest.mark.parametrize('change', [
    lambda s: s.update(status='partial'),
    lambda s: s.update(stale=True),
    lambda s: s.update(stamp=float('nan')),
    lambda s: s.update(published_at=99),
    lambda s: s['objects']['red_cube'].update(detected=False),
    lambda s: s['objects']['red_cube'].update(stale=True),
    lambda s: s['objects']['red_cube'].update(age_s=3),
    lambda s: s['objects']['red_cube']['position'].update(x=float('inf')),
    lambda s: s['zones']['zone_b'].update(objects=['blue_cube', 'yellow_cube']),
    lambda s: s['zones']['zone_b'].update(object='red_cube'),
])
def test_invalid_environment_refuses_plan(planner, state, change):
    change(state)
    with pytest.raises(EnvironmentError):
        planner.resolve_goal('red_cube', 'zone_b', state)


def test_missing_environment_and_invalid_destination(planner, state):
    with pytest.raises(EnvironmentError, match='NOT_RECEIVED'):
        planner.resolve_goal('red_cube', 'zone_b', None)
    with pytest.raises(EnvironmentError, match='INVALID_TARGET_ZONE'):
        planner.resolve_goal('red_cube', 'other_zone', state)


def test_candidate_selection_uses_observed_positions(planner, state):
    state['objects']['green_cube']['position'] = {'x': 0.43, 'y': -0.23, 'z': 0.22}
    name, _ = planner.find_free_temporary_position(state)
    assert name == 'temporary_2'
    assert 'green_cube' in planner.last_rejections['temporary_1']


def test_no_free_slot_refuses_plan(planner, state):
    # This fixture tests exhaustion of its three slots, not the production grid.
    planner.config['temporary_positions'] = dict(list(planner.config['temporary_positions'].items())[:3])
    for obj, slot in zip(('green_cube', 'purple_cube', 'yellow_cube'),
                         planner.config['temporary_positions'].values()):
        state['objects'][obj]['position'] = {axis: slot[axis] for axis in ('x', 'y', 'z')}
    with pytest.raises(EnvironmentError, match='NO_FREE_TEMPORARY_POSITION'):
        planner.resolve_goal('red_cube', 'zone_b', state)


def test_invalid_candidate_geometry(planner, state):
    point = {'frame_id': 'base_link', 'x': 0.28, 'y': -0.15, 'z': 0.22}
    assert 'zone' in planner.candidate_rejection(point, state)
    point['x'] = 0.60
    assert 'boundary' in planner.candidate_rejection(point, state)


@pytest.mark.parametrize('change', [
    lambda p: p['plan'][0].update(skill='teleport'),
    lambda p: p['plan'][0].update(object='unknown_cube'),
    lambda p: p['plan'][1].update(destination='temporary_99'),
    lambda p: p['plan'][1].update(object='red_cube'),
    lambda p: p['plan'][1].update(x=0.1),
    lambda p: p['temporary_positions']['temporary_1'].update(x=0.1),
    lambda p: p['plan'].pop(0),
    lambda p: p['plan'].pop(),
])
def test_dry_run_validator_rejects_tampered_plan(planner, state, change):
    plan = planner.resolve_goal('red_cube', 'zone_b', state)
    change(plan)
    valid, _ = planner.validator.validate(plan, planner.selected_temporary_positions, state)
    assert not valid


@pytest.mark.parametrize('payload', ['{bad', '{"status":"ok","status":"stale"}', '{"x":NaN}'])
def test_bad_json_is_rejected(payload):
    with pytest.raises(EnvironmentError):
        parse_json(payload)


@pytest.mark.parametrize('failure', ['receive_timeout', 'frozen_stamp', 'bad_latest',
                                   'wrong_clock', 'expired_object'])
def test_snapshot_rejects_stale_cached_state(planner, state, monkeypatch, failure):
    monkeypatch.setattr('ur3_vision_planning.environment_manager.time.monotonic', lambda: 100.0)
    clock = SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_200_000_000))
    manager = SimpleNamespace(
        latest_error=None, last_received=99.9, last_progress=99.9,
        latest_state=state, config=planner.config, planner=planner,
        get_clock=lambda: clock)
    if failure == 'receive_timeout':
        manager.last_received = 98.0
    elif failure == 'frozen_stamp':
        manager.last_progress = 98.0
    elif failure == 'bad_latest':
        manager.latest_error = 'INVALID_JSON'
    elif failure == 'wrong_clock':
        clock.now = lambda: SimpleNamespace(nanoseconds=100_000_000_000)
    else:
        state['objects']['red_cube']['age_s'] = 0.99
    with pytest.raises(EnvironmentError):
        EnvironmentManager.snapshot(manager)


def test_all_configured_candidates_fit_default_camera_scene(planner, state):
    # Include blue at its default observed location as well as occupied-zone scenario.
    state['objects']['blue_cube']['position'] = {'x': 0.38, 'y': 0.15, 'z': 0.22}
    for candidate in planner.config['temporary_positions'].values():
        assert planner.candidate_rejection(candidate, state) is None

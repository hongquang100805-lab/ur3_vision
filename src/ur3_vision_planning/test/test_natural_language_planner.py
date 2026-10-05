"""Unit-level transport doubles only; production always calls real 9Router."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_environment_manager import planner, state  # reuse camera-shaped fixtures
from ur3_vision_planning.environment_manager import EnvironmentError, validate_structured_goal
from ur3_vision_planning.llm_planner import LLMPlanningError, _configured_model
from ur3_vision_planning.natural_language_planner import (
    StructuredGoalPlanner, advisory_catalog, goal_system_prompt, resolve_command,
)


STUDENT = {'student_name': 'Lê Hồng Quang', 'student_id': '23020757'}


@pytest.mark.parametrize('obj', ['red_cube', 'blue_cube', 'yellow_cube', 'green_cube', 'purple_cube'])
def test_goal_accepts_all_five_objects(obj):
    goal = {'object': obj, 'target_zone': 'zone_b'}
    assert validate_structured_goal(json.dumps(goal)) == goal


@pytest.mark.parametrize('payload', [
    '{}', 'null', '[]', '{broken', '```json\n{}\n```',
    '{"object":"red_cube","object":"blue_cube","target_zone":"zone_b"}',
    {'object': 'unknown_cube', 'target_zone': 'zone_b'},
    {'object': 'red_cube', 'target_zone': 'zone_d'},
    {'object': ['red_cube'], 'target_zone': 'zone_b'},
    {'object': 'red_cube', 'target_zone': False},
    {'object': 'red_cube', 'target_zone': 'zone_b', 'x': 0.43},
    {'object': 'red_cube', 'target_zone': 'zone_b', 'temporary_positions': {}},
    {'object': 'red_cube', 'target_zone': 'zone_b', 'skill': 'pick'},
    {'object': 'red_cube', 'target_zone': 'zone_b', 'trajectory': []},
])
def test_goal_rejects_untrusted_structure(payload):
    with pytest.raises(EnvironmentError):
        validate_structured_goal(payload)


def test_model_environment_precedence(monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'legacy/model')
    monkeypatch.setenv('OPENAI_MODEL', 'oc/muse-spark-1.2-contributor-free')
    assert _configured_model() == 'oc/muse-spark-1.2-contributor-free'
    monkeypatch.delenv('OPENAI_MODEL')
    assert _configured_model() == 'legacy/model'
    monkeypatch.delenv('LLM_MODEL')
    assert _configured_model(default='') == ''  # new pipeline must not choose Gemini


@pytest.mark.parametrize('content', [
    '{"object":"red_cube","target_zone":"zone_b"}', '{invalid',
])
def test_structured_goal_uses_exactly_one_real_transport_call(content):
    sent = []
    fake = SimpleNamespace(system_prompt=goal_system_prompt(STUDENT), _check_configuration=lambda: None)
    def chat(messages, operation, json_mode=False):
        sent.append((messages, operation, json_mode))
        return content
    fake._chat_completion = chat
    if content == '{invalid':
        with pytest.raises(EnvironmentError):
            StructuredGoalPlanner.structured_goal(fake, 'Put the red cube in Zone B.')
    else:
        assert StructuredGoalPlanner.structured_goal(fake, 'Đưa vật màu đỏ vào vùng B.') == {
            'object': 'red_cube', 'target_zone': 'zone_b'}
    assert len(sent) == 1
    assert sent[0][2] is True
    assert sent[0][0][0]['content'] == goal_system_prompt(STUDENT)
    assert 'purple_cube' in sent[0][0][0]['content']
    assert 'Lê Hồng Quang' in sent[0][0][0]['content']


def test_http_error_is_not_retried_or_replaced():
    calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        raise LLMPlanningError('HTTP 429')
    fake = SimpleNamespace(system_prompt='goal only', _check_configuration=lambda: None,
                           _chat_completion=failed)
    with pytest.raises(LLMPlanningError, match='429'):
        StructuredGoalPlanner.structured_goal(fake, 'Move red to B')
    assert len(calls) == 1


def test_dynamic_catalog_is_warning_only(capsys):
    advisory_catalog(SimpleNamespace(model='oc/not-listed', list_models=lambda: set()))
    assert 'WARNING' in capsys.readouterr().out


def test_response_preview_redacts_key_before_truncation():
    key = 'private-api-key'
    fake = SimpleNamespace(api_key=key)
    preview = StructuredGoalPlanner._body_preview(fake, f'error Authorization: {key}'.encode())
    assert key not in preview and '<redacted>' in preview


@pytest.mark.parametrize('tamper', ['wrong_zone', 'wrong_object', 'skipped_occupant', 'extra_moves'])
def test_final_plan_must_satisfy_structured_goal(planner, state, tamper):
    goal = {'object': 'red_cube', 'target_zone': 'zone_b'}
    plan = planner.resolve_goal(**{'object_name': 'red_cube', 'target_zone': 'zone_b', 'environment_state': state})
    if tamper == 'wrong_zone':
        plan['plan'][3]['destination'] = 'zone_c'
    elif tamper == 'wrong_object':
        plan['plan'][2]['object'] = plan['plan'][3]['object'] = 'green_cube'
    elif tamper == 'skipped_occupant':
        del plan['plan'][:2]
    else:
        plan['plan'][-1:-1] = [{'skill': 'pick', 'object': 'green_cube'},
                               {'skill': 'place', 'object': 'green_cube', 'destination': 'zone_a'}]
    valid, _ = planner.validate_plan(plan, goal, state)
    assert not valid


def test_validator_rechecks_camera_occupancy_at_temporary_slot(planner, state):
    goal = {'object': 'red_cube', 'target_zone': 'zone_b'}
    plan = planner.resolve_goal('red_cube', 'zone_b', state)
    point = plan['temporary_positions']['temporary_1']
    state['objects']['green_cube']['position'] = {axis: point[axis] for axis in ('x', 'y', 'z')}
    valid, reason = planner.validate_plan(plan, goal, state)
    assert not valid and 'UNSAFE_TEMPORARY_POSITION' in reason


@pytest.mark.parametrize('steps', [
    [{'skill': 'pick', 'object': 'blue_cube'}, {'skill': 'pick', 'object': 'red_cube'}, {'skill': 'home'}],
    [{'skill': 'place', 'object': 'red_cube', 'destination': 'zone_a'}, {'skill': 'home'}],
    [{'skill': 'home'}, {'skill': 'pick', 'object': 'red_cube'}],
])
def test_invalid_held_object_order_is_rejected(planner, state, steps):
    goal = {'object': 'red_cube', 'target_zone': 'zone_b'}
    plan = planner.resolve_goal('red_cube', 'zone_b', state)
    plan['plan'] = steps
    assert not planner.validate_plan(plan, goal, state)[0]


def test_pipeline_resolves_llm_goal_with_camera_not_llm_coordinates(planner, state, monkeypatch):
    monkeypatch.setattr('ur3_vision_planning.natural_language_planner.rclpy.ok', lambda: True)
    monkeypatch.setattr('ur3_vision_planning.natural_language_planner.rclpy.spin_once', lambda *a, **k: None)
    goal = {'object': 'red_cube', 'target_zone': 'zone_b'}
    manager = SimpleNamespace(planner=planner, fresh_snapshot=lambda: copy.deepcopy(state))
    llm = SimpleNamespace(structured_goal=lambda command: goal)
    parsed, observed, expanded = resolve_command(manager, llm, 'red to B')
    assert parsed == goal and observed == state
    assert expanded['plan'][0] == {'skill': 'pick', 'object': 'blue_cube'}
    assert expanded['temporary_positions']['temporary_1']['x'] == 0.43


@pytest.mark.parametrize('failure', ['llm_added_coordinates', 'camera_stale'])
def test_pipeline_refuses_before_success_output(planner, state, monkeypatch, capsys, failure):
    monkeypatch.setattr('ur3_vision_planning.natural_language_planner.rclpy.ok', lambda: True)
    monkeypatch.setattr('ur3_vision_planning.natural_language_planner.rclpy.spin_once', lambda *a, **k: None)
    goal = {'object': 'red_cube', 'target_zone': 'zone_b'}
    camera_reads = []
    def snapshot():
        camera_reads.append(1)
        raise EnvironmentError('ENVIRONMENT_NOT_READY: status=stale')
    manager = SimpleNamespace(planner=planner, fresh_snapshot=snapshot)
    if failure == 'llm_added_coordinates':
        goal['x'] = 0.43
    llm = SimpleNamespace(structured_goal=lambda command: goal)
    with pytest.raises(EnvironmentError):
        resolve_command(manager, llm, 'red to B')
    assert len(camera_reads) == (0 if failure == 'llm_added_coordinates' else 1)
    assert 'DRY-RUN COMPLETE' not in capsys.readouterr().out


def test_dry_run_module_has_no_robot_motion_dependencies():
    path = Path(__file__).resolve().parents[1] / 'ur3_vision_planning' / 'natural_language_planner.py'
    tree = ast.parse(path.read_text())
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(module and ('robot_skills' in module or 'moveit' in module) for module in modules)
    names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    assert not set(names) & {'RobotSkills', 'ActionClient', 'MoveGroup', 'ExecuteTrajectory'}

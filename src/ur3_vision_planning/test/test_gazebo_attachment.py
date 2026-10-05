"""Attachment protocol regression tests; no simulator commands or ROS init."""

import subprocess
from types import SimpleNamespace

import pytest

from ur3_vision_planning import robot_skills as module
from ur3_vision_planning.robot_skills import RobotSkills


GENERIC_ATTACH = '''No publishers on topic [/ur3_vision_planning/grasp/blue_cube/attach]
Subscribers [Address, Message Type]:
  tcp://172.17.0.1:43423, google.protobuf.Message
'''
TYPED_DETACH = '''No publishers on topic [/ur3_vision_planning/grasp/blue_cube/detach]
Subscribers [Address, Message Type]:
  tcp://172.17.0.1:43423, gz.msgs.Empty
'''


@pytest.mark.parametrize('output,action,expected', [
    (GENERIC_ATTACH, 'attach', True),
    (GENERIC_ATTACH, 'detach', False),
    (TYPED_DETACH, 'attach', True),
    (TYPED_DETACH, 'detach', True),
    ('Publishers [Address, Message Type]:\n  tcp://x:1, gz.msgs.Empty\n'
     'Subscribers [Address, Message Type]:\n', 'attach', False),
    ('Subscribers [Address, Message Type]:\n'
     'Publishers [Address, Message Type]:\n  tcp://x:1, gz.msgs.Empty\n', 'attach', False),
    ('Subscribers [Address, Message Type]:\n  tcp://x:1, gz.msgs.StringMsg\n', 'attach', False),
    ('No subscribers on topic [/blue/attach]\n', 'attach', False),
    ('', 'attach', False),
])
def test_subscriber_metadata(output, action, expected):
    assert RobotSkills.attachment_subscriber_present(output, action) is expected


def test_discovery_uses_correct_action_topic(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=TYPED_DETACH)
    monkeypatch.setattr(module.subprocess, 'run', run)
    fake = SimpleNamespace(attachment_subscriber_present=RobotSkills.attachment_subscriber_present)
    assert RobotSkills.gazebo_attachment_topic_has_subscriber(fake, 'blue_cube', 'detach')
    assert calls[0][-1] == '/ur3_vision_planning/grasp/blue_cube/detach'


@pytest.fixture
def skills():
    logs, checks = [], []
    def subscriber(name, action='attach'):
        checks.append((name, action))
        return True
    logger = SimpleNamespace(info=logs.append, error=logs.append)
    fake = SimpleNamespace(
        scene={'objects': {'blue_cube': {}}},
        gazebo_attachment_plugins=set(), gazebo_attachment_states={},
        get_logger=lambda: logger,
        gazebo_attachment_topic_has_subscriber=subscriber,
        get_gazebo_model_entity_id=lambda name: 42,
        parse_detachable_joint_state=RobotSkills.parse_detachable_joint_state,
    )
    fake.logs, fake.checks = logs, checks
    return fake


class Listener:
    def __init__(self, output):
        self.output = output

    def communicate(self, **kwargs):
        return self.output, ''

    def poll(self):
        return 0


def protocol(monkeypatch, output, code=0, response=''):
    """Fixtures intercept every publisher/system-add invocation."""
    calls = []
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: Listener(output))
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=code, stdout=response, stderr='')
    monkeypatch.setattr(module.subprocess, 'run', run)
    return calls


def test_initial_dynamic_attach_reuses_ack_without_second_publish(skills, monkeypatch):
    # Loading initially attaches and emits one transition. The old code sent
    # attach a second time, got no state, and misread the generic subscriber.
    ready = iter((False, True))
    skills.gazebo_attachment_topic_has_subscriber = lambda *a: next(ready)
    calls = protocol(monkeypatch, 'data: "attached"\n', response='data: true\n')
    assert RobotSkills.load_gazebo_attachment_plugin(skills, 'blue_cube')
    assert skills.gazebo_attachment_states['blue_cube'] is True
    assert RobotSkills.set_gazebo_attachment(skills, 'blue_cube', True)
    assert len(calls) == 1 and calls[0][1] == 'service'
    assert any('no duplicate command' in line for line in skills.logs)


@pytest.mark.parametrize('state', [True, False])
def test_only_known_matching_state_is_idempotent(skills, monkeypatch, state):
    skills.gazebo_attachment_states['blue_cube'] = state
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: pytest.fail('Duplicate command'))
    assert RobotSkills.set_gazebo_attachment(skills, 'blue_cube', state)
    assert skills.checks == [('blue_cube', 'attach' if state else 'detach')]


def test_subscriber_alone_with_no_state_never_passes(skills, monkeypatch):
    calls = protocol(monkeypatch, '')
    assert not RobotSkills.set_gazebo_attachment(skills, 'blue_cube', True)
    assert len(calls) == 1
    assert skills.gazebo_attachment_states['blue_cube'] is None
    assert any('no state acknowledgement' in line for line in skills.logs)


def test_detach_requires_fresh_ack_and_checks_detach_subscriber(skills, monkeypatch):
    skills.gazebo_attachment_states['blue_cube'] = True
    calls = protocol(monkeypatch, 'data: "detached"\n')
    assert RobotSkills.set_gazebo_attachment(skills, 'blue_cube', False)
    assert skills.gazebo_attachment_states['blue_cube'] is False
    assert skills.checks == [('blue_cube', 'detach')]
    assert calls[0][3].endswith('/detach')


@pytest.mark.parametrize('failure', ['silent', 'wrong', 'publish', 'timeout'])
def test_failed_detach_does_not_release_known_held_state(skills, monkeypatch, failure):
    skills.attached_object, skills.gazebo_cube_attached = 'blue_cube', True
    skills.gazebo_attachment_states['blue_cube'] = True
    output = 'data: "attached"\n' if failure == 'wrong' else ''
    protocol(monkeypatch, output, code=1 if failure == 'publish' else 0)
    if failure == 'timeout':
        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired('gz topic', 2.)
        monkeypatch.setattr(module.subprocess, 'run', timeout)
    assert not RobotSkills.set_gazebo_attachment(skills, 'blue_cube', False)
    assert skills.attached_object == 'blue_cube' and skills.gazebo_cube_attached
    assert skills.gazebo_attachment_states['blue_cube'] is (True if failure == 'wrong' else None)


def test_discovery_loss_invalidates_previous_ack_without_publishing(skills, monkeypatch):
    skills.gazebo_attachment_states['blue_cube'] = True
    skills.gazebo_attachment_topic_has_subscriber = lambda *a: False
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: pytest.fail('Must stop'))
    assert not RobotSkills.set_gazebo_attachment(skills, 'blue_cube', True)
    assert skills.gazebo_attachment_states['blue_cube'] is None


def test_system_add_rejected_cannot_cache_ack(skills, monkeypatch):
    skills.gazebo_attachment_topic_has_subscriber = lambda *a: False
    protocol(monkeypatch, 'data: "attached"\n', code=1, response='data: true\n')
    assert not RobotSkills.load_gazebo_attachment_plugin(skills, 'blue_cube')
    assert 'blue_cube' not in skills.gazebo_attachment_states


def test_unknown_object_never_publishes(skills, monkeypatch):
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: pytest.fail('Unknown object'))
    assert not RobotSkills.set_gazebo_attachment(skills, 'unknown_cube', True)


def test_existing_plugin_discovery_does_not_invent_ack(skills, monkeypatch):
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: pytest.fail('Duplicate plugin'))
    assert RobotSkills.load_gazebo_attachment_plugin(skills, 'blue_cube')
    assert 'blue_cube' in skills.gazebo_attachment_plugins
    assert 'blue_cube' not in skills.gazebo_attachment_states

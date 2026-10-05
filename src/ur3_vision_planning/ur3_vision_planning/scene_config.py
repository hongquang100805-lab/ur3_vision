"""Resolve initial spawn scenarios shared by Gazebo and MoveIt."""

import yaml


def load_scene(path, scenario='default'):
    with open(path, encoding='utf-8') as config_file:
        scene = yaml.safe_load(config_file)
    scenarios = scene.get('scenarios', {'default': {}})
    if scenario not in scenarios:
        raise ValueError(f'Unknown scene scenario {scenario!r}; choose {list(scenarios)}')
    for name, position in scenarios[scenario].get('object_positions', {}).items():
        if name not in scene['objects']:
            raise ValueError(f'Scenario references unknown object {name!r}')
        scene['objects'][name]['position'] = list(position)
    return scene

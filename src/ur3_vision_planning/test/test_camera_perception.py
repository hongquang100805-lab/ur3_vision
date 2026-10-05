import numpy as np
import pytest
import yaml
from pathlib import Path

from ur3_vision_planning.camera_perception import median_depth, object_zone


def configs():
    directory = Path(__file__).resolve().parents[1] / 'config'
    return (yaml.safe_load((directory / 'perception.yaml').read_text()),
            yaml.safe_load((directory / 'scene.yaml').read_text()))


def test_depth_median_rejects_invalid_samples():
    config, _ = configs()
    depth = np.full((5, 5), 0.91, dtype=np.float32)
    depth[0] = [0, np.nan, np.inf, -1, 10]
    depth[1, 0] = 0.5
    assert median_depth(depth, 2, 2, config['depth']) == pytest.approx(0.91)
    assert median_depth(np.zeros((5, 5)), 2, 2, config['depth']) is None


def test_zone_assignment_requires_supported_whole_cube():
    config, scene = configs()
    assert object_zone([0.28, 0, 0.22], scene, config['zones'], config['block_size']) == 'zone_b'
    assert object_zone([0.38, 0, 0.22], scene, config['zones'], config['block_size']) is None
    assert object_zone([0.28, 0.03, 0.22], scene, config['zones'], config['block_size']) is None
    assert object_zone([0.28, 0, 0.30], scene, config['zones'], config['block_size']) is None

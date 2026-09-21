"""Opt-in OpenGL checks: VISTA_TEST_RENDER=1 xvfb-run -a python -m pytest."""
import os

import numpy as np
import pytest

from vista.config import load
from vista.runtime import components

pytestmark = pytest.mark.skipif(
    os.environ.get("VISTA_TEST_RENDER") != "1", reason="requires an OpenGL display"
)


@pytest.mark.parametrize("phase", ["phase2", "phase3"])
def test_rgb_capture(phase, monkeypatch):
    monkeypatch.setenv("VISTA_RENDER_HIDDEN", "1")
    cfg = load(phase + "_vista")
    env_type, binding = components(phase)
    env = env_type(**dict(cfg["env"], num_envs=1, render_fps=240))
    try:
        env.reset(seed=42)
        frame = binding.vec_render_rgb(env.c_envs, 0)
        assert frame.shape == (1080, 1920, 3)
        assert frame.dtype == np.uint8
        assert frame.max() > 200
        assert frame[:, :570].std() > 5
        for index in (-1, 1):
            with pytest.raises(IndexError):
                binding.vec_render_rgb(env.c_envs, index)
        env.step(np.full(env.num_agents, env.single_action_space.n - 1, dtype=np.int32))
        next_frame = binding.vec_render_rgb(env.c_envs, 0)
        assert not np.array_equal(frame, next_frame)
        env.reset(seed=42)
        reset_frame = binding.vec_render_rgb(env.c_envs, 0)
        np.testing.assert_array_equal(frame, reset_frame)
    finally:
        env.close()

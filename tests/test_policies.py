"""Regression checks against the released, immutable checkpoint artifacts."""
import hashlib
import json

import numpy as np
import pytest
import torch

from vista.config import PRESETS, ROOT, load, phase_of
from vista.runtime import components, make_policy, register
from vista.evaluate import evaluate


@pytest.mark.parametrize("preset", PRESETS)
def test_released_checkpoint(preset):
    torch.set_num_threads(1)
    cfg = load(preset)
    name = "LSTM" if preset.endswith("lstm") else "VISTA"
    assert cfg["policy_name"] == name
    assert cfg["rnn_name"] == name + "Recurrent"
    manifest = json.loads((ROOT / "checkpoints/manifest.json").read_text())
    checkpoint = ROOT / manifest[preset]["checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest[preset]["sha256"]
    env = components(phase_of(cfg))[0](**dict(cfg["env"], num_envs=1))
    try:
        policy = make_policy(cfg, env, checkpoint).eval()
        assert type(policy).__name__ == name + "Recurrent"
        assert type(policy.policy).__name__ == name
        obs, _ = env.reset(seed=42)
        state = {"lstm_h": None, "lstm_c": None}
        with torch.no_grad():
            for _ in range(3):
                logits, value = policy.forward_eval(torch.as_tensor(obs), state)
                assert logits.shape == (env.num_agents, env.single_action_space.n)
                assert torch.isfinite(logits).all()
                assert torch.isfinite(value).all()
                assert state["lstm_h"] is not None
                obs, *_ = env.step(logits.argmax(-1).numpy().astype(np.int32))
    finally:
        env.close()


@pytest.mark.parametrize("preset", PRESETS)
def test_evaluation_outputs(preset, tmp_path):
    out = tmp_path / preset
    rows = evaluate(load(preset), steps=2, output=out, deterministic=True)
    assert len(rows) == 1
    assert {p.name for p in out.iterdir()} == {
        "trajectory.csv", "final_rso.csv", "episodes.csv", "manifest.json"
    }
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["policy_name"] in ("VISTA", "LSTM")
    assert manifest["checkpoint_sha256"]
    assert manifest["horizon_steps"] == 2


def test_plugin_registration_switches_phase():
    import sys
    from vista import vista_policy, cooperative_policy
    for phase, module in [("phase3", cooperative_policy), ("phase1", vista_policy),
                          ("phase2", vista_policy)]:
        register(phase)
        plugin = sys.modules["pufferlib.environments.vista"].torch
        assert plugin.VISTA is module.VISTA
        assert plugin.LSTM is module.LSTM
    with pytest.raises(ValueError, match="Unknown phase"):
        register("phase4")

import pytest
from vista.config import load, phase_of


def test_scenario_override_updates_policy_dimensions():
    cfg = load("phase2_vista", ["env.num_agents=48", "env.rso_top_k=30"])
    assert cfg["policy"]["num_agents"] == 48
    assert cfg["policy"]["rso_top_k"] == 30
    assert phase_of(cfg) == "phase2"


@pytest.mark.parametrize("override", ["env.unknown=1", "num_agents=48"])
def test_invalid_override(override):
    with pytest.raises(ValueError):
        load("phase1_vista", [override])

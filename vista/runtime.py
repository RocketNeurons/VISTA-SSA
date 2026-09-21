"""Factory and external PufferLib plugin integration."""
import importlib
import sys
import types
from . import vista_policy, cooperative_policy
from .config import phase_of


def components(phase):
    module = importlib.import_module(f'vista.envs.{phase}.environment')
    binding = importlib.import_module(f'vista.envs.{phase}.binding')
    name = {'phase1': 'OrbitalEyesSandbox', 'phase2': 'OrbitalEyes',
            'phase3': 'OrbitalEyesCooperative'}[phase]
    return getattr(module, name), binding


def env_creator(name):
    return components(name.split('_')[0])[0]


def register(phase='phase1'):
    """Register this external package for the lifetime of this Python process."""
    if phase not in ('phase1', 'phase2', 'phase3'):
        raise ValueError(f'Unknown phase: {phase}')
    policy_module = cooperative_policy if phase == 'phase3' else vista_policy
    plugin = types.ModuleType('pufferlib.environments.vista')
    plugin.env_creator = env_creator
    plugin.torch = types.SimpleNamespace(**{
        name: getattr(policy_module, name)
        for name in ('VISTA', 'VISTARecurrent', 'LSTM', 'LSTMRecurrent')
    })
    sys.modules[plugin.__name__] = plugin


def make_policy(cfg, env, checkpoint=None, device='cpu'):
    import torch
    register(phase_of(cfg))
    module = sys.modules['pufferlib.environments.vista'].torch
    policy = getattr(module, cfg['policy_name'])(env, **cfg['policy'])
    if cfg['rnn_name']:
        policy = getattr(module, cfg['rnn_name'])(env, policy, **cfg['rnn'])
    if checkpoint:
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        weights = {k.removeprefix('module.'): v for k, v in weights.items()}
        policy.load_state_dict(weights, strict=True)
    return policy.to(device)

"""Factory and external PufferLib plugin integration."""
import importlib
import sys
import types
from . import policies, cooperative_policy


def components(phase):
    module = importlib.import_module(f'vista.envs.{phase}.environment')
    binding = importlib.import_module(f'vista.envs.{phase}.binding')
    name = {'phase1': 'OrbitalEyesSandbox', 'phase2': 'OrbitalEyes',
            'phase3': 'OrbitalEyesCooperative'}[phase]
    return getattr(module, name), binding


def env_creator(name):
    return components(name.split('_')[0])[0]


def register():
    """Register this external package for the lifetime of this Python process."""
    plugin = types.ModuleType('pufferlib.environments.vista')
    plugin.env_creator = env_creator
    plugin.torch = types.SimpleNamespace(**{
        name: getattr(module, name)
        for module in (policies, cooperative_policy)
        for name in dir(module) if name.startswith('OrbitalEyes')
    })
    sys.modules[plugin.__name__] = plugin


def make_policy(cfg, env, checkpoint=None, device='cpu'):
    import torch
    register()
    module = sys.modules['pufferlib.environments.vista'].torch
    policy = getattr(module, cfg['policy_name'])(env, **cfg['policy'])
    if cfg['rnn_name']:
        policy = getattr(module, cfg['rnn_name'])(env, policy, **cfg['rnn'])
    if checkpoint:
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        weights = {k.removeprefix('module.'): v for k, v in weights.items()}
        policy.load_state_dict(weights, strict=True)
    return policy.to(device)

"""Explicit INI loading, without modifying PufferLib's installed config tree."""
import ast
import configparser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ('phase1_vista', 'phase1_lstm', 'phase2_vista', 'phase2_lstm', 'phase3_vista')


def cast(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load(path, overrides=()):
    candidate = Path(path)
    if str(path) in PRESETS:
        candidate = ROOT / 'configs' / f'{path}.ini'
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    cp = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    cp.read([ROOT / 'configs/default.ini', candidate])
    cfg = dict(load_model_path=None, load_id=None, render_mode='raylib',
               save_frames=0, gif_path='eval.gif', fps=15, max_runs=20,
               wandb=False, wandb_project='vista', wandb_group='paper',
               neptune=False, neptune_name='pufferai', neptune_project='ablations',
               local_rank=0, tag=None, no_model_upload=True)
    for section in cp.sections():
        if section == 'base':
            cfg.update({k: cast(v) for k, v in cp[section].items()})
            continue
        dest = cfg
        for key in section.split('.'):
            dest = dest.setdefault(key, {})
        dest.update({k: cast(v) for k, v in cp[section].items()})
    for item in overrides:
        if '=' not in item or '.' not in item.split('=', 1)[0]:
            raise ValueError('Overrides must be section.key=value, e.g. env.num_rso=5000')
        key, value = item.split('=', 1)
        parts = key.split('.')
        dest = cfg
        for part in parts[:-1]:
            dest = dest.setdefault(part, {})
        if parts[-1] not in dest:
            raise ValueError(f'Unknown configuration key: {key}')
        dest[parts[-1]] = cast(value)
    for key in ('num_agents', 'rso_top_k', 'action_mode'):
        cfg['policy'][key] = cfg['env'][key]
    cfg['train']['use_rnn'] = cfg['rnn_name'] is not None
    cfg['package'] = 'vista'
    return cfg


def phase_of(cfg):
    name = cfg['env_name']
    phase = name.split('_')[0]
    if phase not in ('phase1', 'phase2', 'phase3'):
        raise ValueError('Keep env_name as phase1_*, phase2_* or phase3_* in custom INIs.')
    return phase

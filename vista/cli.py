"""Small public interface around the environments and external PuffeRL."""
import argparse
import json
from .config import ROOT, PRESETS, load


def main(argv=None):
    parser = argparse.ArgumentParser(description='VISTA sensor tasking: paper scenarios, pretrained policies, and classical references.')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List the five pretrained presets')
    for name in ('eval', 'train', 'sweep'):
        p = sub.add_parser(name)
        p.add_argument('config', help='Preset name or path to a custom INI')
        p.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE')
        p.add_argument('--device', default=None, choices=('cpu','cuda'))
        p.add_argument('--checkpoint', help='Optional weights; training starts fresh unless supplied')
        if name == 'eval':
            p.add_argument('--classical', choices=('random','oldest','greedy','maxu','eig','beam'))
            p.add_argument('--render', action='store_true')
            p.add_argument('--episodes', type=int, default=1)
            p.add_argument('--steps', type=int)
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--output', default='runs/eval')
            p.add_argument('--deterministic', action='store_true', help='Use argmax rather than the default stochastic policy')
        else:
            p.add_argument('--wandb', action='store_true')
            p.add_argument('--wandb-project', default='vista')
            p.add_argument('--max-runs', type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == 'list':
        print('\n'.join(PRESETS)); return
    cfg = load(args.config, args.set)
    if args.device:
        cfg['train']['device'] = args.device
    if args.command == 'eval':
        if args.classical and args.checkpoint:
            parser.error('--classical and --checkpoint are mutually exclusive')
        from .evaluate import evaluate
        evaluate(cfg, checkpoint=args.checkpoint, method=args.classical, episodes=args.episodes,
                 steps=args.steps, render=args.render, output=args.output, seed=args.seed,
                 device=args.device or 'cpu', deterministic=args.deterministic)
        return
    from .runtime import register
    register()
    from pufferlib import pufferl
    cfg.update(wandb=args.wandb, wandb_project=args.wandb_project,
               max_runs=args.max_runs, load_model_path=args.checkpoint)
    if args.command == 'sweep' and not args.wandb:
        parser.error('Sweeps require --wandb. Run wandb login first.')
    # Both public 3.0 and the development trainer support explicit args and
    # resolve the external environment module registered above.
    getattr(pufferl, args.command)(env_name=cfg['env_name'], args=cfg)

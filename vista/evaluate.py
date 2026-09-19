"""Rendered or headless frozen-policy evaluation with auditable CSV outputs."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import classical
from .config import ROOT, phase_of
from .runtime import components, make_policy


def evaluate(cfg, *, checkpoint=None, method=None, episodes=1, steps=None,
             render=False, output='runs/eval', seed=42, device='cpu', deterministic=False):
    if episodes < 1 or (steps is not None and steps < 1):
        raise ValueError('episodes and steps must be positive')
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; choose a fresh --output directory')
    out.mkdir(parents=True, exist_ok=True)
    phase = phase_of(cfg)
    env_cls, binding = components(phase)
    kw = dict(cfg['env'])
    horizon = int(steps if steps is not None else (4000 if phase == 'phase1' else 3600))
    # Read the terminal state before the native environment auto-resets.
    kw.update(num_envs=1, max_steps=horizon+1, report_interval=horizon+1)
    env = env_cls(**kw)
    torch.set_num_threads(1)
    policy = None
    if not method:
        checkpoint = Path(checkpoint or ROOT / 'checkpoints' / f'{cfg["env_name"]}.pt')
        policy = make_policy(cfg, env, checkpoint, device).eval()
    u = np.empty((1, int(kw['num_rso'])), np.float32)
    age = np.empty_like(u)
    rows = []
    started = time.perf_counter()
    try:
        with (out/'trajectory.csv').open('w', newline='') as f, (out/'final_rso.csv').open('w', newline='') as g:
            trajectory = csv.writer(f)
            trajectory.writerow(['episode','seed','step','time_s','mean_u_km','max_u_km','p99_u_km'])
            final = csv.writer(g)
            final.writerow(['episode','seed','rso','u_km','age_s'])
            for ep in range(episodes):
                ep_seed = seed+ep
                torch.manual_seed(ep_seed)
                rng = np.random.default_rng(ep_seed)
                obs, _ = env.reset(seed=ep_seed)
                state = {'lstm_h': None, 'lstm_c': None}
                for step in range(horizon+1):
                    if step % 30 == 0 or step == horizon:
                        binding.vec_rso_state(env.c_envs, u, age)
                        trajectory.writerow([ep, ep_seed, step, step*kw['dt'], float(u.mean()),
                                             float(u.max()), float(np.percentile(u,99))])
                    if step == horizon:
                        break
                    if policy is not None:
                        with torch.no_grad():
                            logits, _ = policy.forward_eval(torch.as_tensor(obs, device=device), state)
                            chosen = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
                            action = chosen.cpu().numpy().astype(np.int32)
                    else:
                        action = classical.actions(method, obs, kw, phase, rng, env.single_action_space.n)
                    obs, _, _, _, _ = env.step(action)
                    if render:
                        env.render()
                for rso in range(u.shape[1]):
                    final.writerow([ep, ep_seed, rso, float(u[0,rso]), float(age[0,rso])])
                rows.append({'episode':ep, 'seed':ep_seed, 'mean_u_km':float(u.mean()),
                             'max_u_km':float(u.max()), 'p99_u_km':float(np.percentile(u,99))})
                print(f'Episode {ep+1}/{episodes}: mean U={u.mean():.6f} km; max U={u.max():.6f} km', flush=True)
    finally:
        env.close()
    with (out/'episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    manifest = dict(config=cfg, seed=seed, episodes=episodes, horizon_steps=horizon,
                    method=method or 'policy', action_selection='argmax' if deterministic else 'sample',
                    checkpoint=str(checkpoint) if checkpoint else None,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
                    elapsed_seconds=time.perf_counter()-started)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return rows

"""Top-K classical references; reachability is independent of current FOV."""
import numpy as np

ALIASES = {'greedy': 'maxu', 'max_u': 'maxu', 'max_uncertainty_greedy': 'maxu',
           'expected_info_gain': 'eig', 'random_feasible': 'random', 'oldest_first': 'oldest',
           'beam_a_star': 'beam'}


def decode(obs, cfg, phase):
    n, k = int(cfg['num_agents']), int(cfg['rso_top_k'])
    off = (18 + 11*(n-1)) if phase == 'phase3' else (12 + 7*(n-1))
    tokens = obs[:, off:off+k*19].reshape(-1, k, 19)
    valid = ~np.all(tokens == -1, axis=-1)
    bounded = phase != 'phase2'
    raw = np.arctanh(np.clip(tokens, -.999999, .999999)) if bounded else tokens
    sigma = np.expm1(raw[:, :, 6:9]) * cfg['obs_u_norm']
    age = np.expm1(raw[:, :, 10]) * cfg['obs_age_norm']
    rel = raw[:, :, :3] * cfg['obs_dist_norm']
    distance = np.linalg.norm(rel, axis=-1)
    if phase == 'phase2':
        pos = obs[:, None, :3] * cfg['obs_dist_norm']
        d2 = np.sum(rel*rel, axis=-1)
        t = -np.sum(pos*rel, axis=-1)/np.maximum(d2, 1e-12)
        closest = pos + np.clip(t, 0, 1)[:, :, None]*rel
        occulted = (t > 0) & (t < 1) & (np.sum(closest*closest, axis=-1) < 6371.0**2)
        feasible = valid.copy()
        if cfg.get('enable_earth_mask', 1):
            feasible &= ~occulted
        if cfg.get('enable_range_mask', 0):
            feasible &= distance <= cfg['max_obs_range']
    else:
        feasible = valid & (tokens[:, :, 12] > .5)
    mode = cfg.get('quality_mode', 0)
    if mode == 0:
        quality = np.ones_like(age)
    elif phase == 'phase3' and mode == 2:
        # A scalar expected-reduction proxy, not the full anisotropic EKF.
        modality = obs[:, 12].astype(int)
        crossover = cfg.get('quality_range_crossover_km', 1900.0)
        width = cfg.get('quality_range_transition_km', 250.0)
        floor = cfg.get('quality_range_floor', .1)
        far = 1/(1+np.exp(-np.clip((distance-crossover)/width, -30, 30)))
        quality = floor + (1-floor)*np.where(modality[:, None] == 1, far, 1-far)
    else:
        quality = tokens[:, :, 13]
    return sigma, age, feasible, quality, rel


def _eig(sigma, quality, cfg):
    noise = cfg['ekf_meas_sigma'] / np.maximum(quality, .05)
    posterior = sigma * noise[:, :, None] / np.sqrt(sigma * sigma + noise[:, :, None]**2 + 1e-12)
    return (sigma - posterior).sum(axis=-1)


def _beam(obs, cfg, sigma, age, feasible, quality, rel, horizon=3, beam=4, pool_cap=60):
    """Receding-horizon coordinated beam search (Beam A* baseline from the paper).

    Builds the joint first-step assignment agent by agent under a no-double-
    observation constraint; leaves are scored by immediate EIG minus a
    surrogate roll-out of catalogue growth over `horizon` future steps.
    Replanned every step.
    """
    n, k = sigma.shape[:2]
    value = np.where(feasible, _eig(sigma, quality, cfg), -np.inf)
    act = np.where(feasible.any(-1), np.where(feasible, value, -np.inf).argmax(-1), 0)

    # Deduplicate candidate slots into unique RSO ids via absolute position
    sat = obs[:, None, :3] * cfg['obs_dist_norm']
    abs_pos = sat + rel                                  # (n, k, 3)
    gid = np.full((n, k), -1, np.int64)
    seen, u_now, age_now = {}, [], []
    for a in range(n):
        for s in range(k):
            if not feasible[a, s]:
                continue
            key = tuple(np.round(abs_pos[a, s] / 2.0).astype(int))
            g = seen.get(key)
            if g is None:
                g = len(u_now)
                seen[key] = g
                u_now.append(float(sigma[a, s].sum()))
                age_now.append(float(age[a, s]))
            gid[a, s] = g
    n_uni = len(u_now)
    if n_uni == 0:
        return act.astype(np.int32)
    u_now, age_now = np.array(u_now), np.array(age_now)

    dt = cfg.get('dt', 5.0)
    grow = ((cfg.get('growth_a_r', 0) + cfg.get('growth_b_r', 0) * age_now)
            + (cfg.get('growth_a_t', 0) + cfg.get('growth_b_t', 0) * age_now)
            + (cfg.get('growth_a_n', 0) + cfg.get('growth_b_n', 0) * age_now)) * dt
    grow = np.maximum(grow, 0.0)
    reset_floor = cfg.get('u_min_r', .02) + cfg.get('u_min_t', .02) + cfg.get('u_min_n', .02)

    keep = None
    if n_uni > pool_cap:
        keep = set(np.argpartition(-u_now, pool_cap)[:pool_cap].tolist())

    feas = []
    for a in range(n):
        opts = [(s, int(gid[a, s]), float(value[a, s])) for s in range(k)
                if gid[a, s] >= 0 and (keep is None or int(gid[a, s]) in keep)]
        opts.sort(key=lambda t: -t[2])
        feas.append(opts[:beam])

    def cost_to_go(observed):
        u = u_now.copy()
        if observed:
            u[list(observed)] = reset_floor
        tot = 0.0
        for _ in range(horizon):
            u = u + grow
            idx = np.argpartition(-u, n)[:n] if n < u.size else np.arange(u.size)
            u[idx] = reset_floor
            tot += u.sum()
        return tot

    nodes = [({}, frozenset(), 0.0)]
    for a in range(n):
        nxt = []
        for slots, obs_set, imv in nodes:
            for (s, g, v) in feas[a]:
                if g in obs_set:
                    continue
                ns = dict(slots); ns[a] = s
                nxt.append((ns, obs_set | {g}, imv + v))
            nxt.append((slots, obs_set, imv))  # allow skip to avoid dead-ends
        nxt.sort(key=lambda t: -t[2])
        nodes = nxt[:beam]

    best_score, best_slots = -1e30, None
    for slots, obs_set, imv in nodes:
        score = imv - cost_to_go(obs_set)
        if score > best_score:
            best_score, best_slots = score, slots
    if best_slots:
        for a, s in best_slots.items():
            act[a] = s
    return act.astype(np.int32)


def actions(method, obs, cfg, phase, rng, num_actions):
    method = ALIASES.get(method, method)
    sigma, age, feasible, quality, rel = decode(obs, cfg, phase)
    if method == 'beam':
        result = _beam(obs, cfg, sigma, age, feasible, quality, rel)
        k = int(cfg['rso_top_k'])
        result[~feasible.any(axis=-1)] = k if num_actions == k + 1 else k - 1
        return result
    if method == 'random':
        score = rng.random(age.shape)
    elif method == 'oldest':
        score = age
    elif method == 'maxu':
        score = sigma.sum(axis=-1)
    elif method == 'eig':
        score = _eig(sigma, quality, cfg)
    else:
        raise ValueError(f'Unknown classical method: {method}')
    result = np.where(feasible, score, -np.inf).argmax(axis=-1).astype(np.int32)
    k = int(cfg['rso_top_k'])
    result[~feasible.any(axis=-1)] = k if num_actions == k+1 else k-1
    return result

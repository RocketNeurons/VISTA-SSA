import numpy as np
import gymnasium

import pufferlib
from vista.envs.phase3 import binding


# ── Uncertainty model presets ────────────────────────────────────────────────
# Each preset overrides only the uncertainty-related defaults.
# Usage:  OrbitalEyes(**UNCERTAINTY_PRESETS["pessimistic"])
UNCERTAINTY_PRESETS = {
    "nominal": dict(
        growth_a_r=2e-5,  growth_a_t=8e-5,  growth_a_n=2e-5,
        growth_b_r=2e-8,  growth_b_t=1e-7,  growth_b_n=2e-8,
        k_v=1e-6,
        alpha_r=0.05,     alpha_t=0.03,     alpha_n=0.05,
        alpha_vel=2e-4,
        u_min_r=0.02,     u_min_t=0.02,     u_min_n=0.02,
        vel_u_min=2e-5,
        u_init_r=0.3,     u_init_t=0.8,     u_init_n=0.3,
        sigma_vel_init=0.001,
        uncertainty_threshold=3.0,
        obs_u_norm=1.0,
        alpha_u=750.0,
        alpha_v=75.0,
    ),
    "pessimistic": dict(
        growth_a_r=6e-5,  growth_a_t=2.4e-4, growth_a_n=6e-5,   # 3× nominal
        growth_b_r=6e-8,  growth_b_t=3e-7,   growth_b_n=6e-8,   # 3× nominal
        k_v=3e-6,
        alpha_r=0.0375,   alpha_t=0.0225,    alpha_n=0.0375,    # 0.75× nominal
        alpha_vel=1.5e-4,
        u_min_r=0.03,     u_min_t=0.03,      u_min_n=0.03,
        vel_u_min=3e-5,
        u_init_r=0.3,     u_init_t=0.8,      u_init_n=0.3,
        sigma_vel_init=0.001,
        uncertainty_threshold=3.0,
        obs_u_norm=1.0,
        alpha_u=750.0,
        alpha_v=75.0,
    ),
    "worst_case_plausible": dict(
        growth_a_r=1e-4,  growth_a_t=4e-4,  growth_a_n=1e-4,   # 5× nominal
        growth_b_r=1e-7,  growth_b_t=5e-7,  growth_b_n=1e-7,   # 5× nominal
        k_v=5e-6,
        alpha_r=0.025,    alpha_t=0.015,    alpha_n=0.025,     # 0.5× nominal
        alpha_vel=1e-4,
        u_min_r=0.03,     u_min_t=0.03,     u_min_n=0.03,
        vel_u_min=3e-5,
        u_init_r=0.5,     u_init_t=1.2,     u_init_n=0.5,
        sigma_vel_init=0.002,
        uncertainty_threshold=4.0,
        obs_u_norm=1.0,
        alpha_u=750.0,
        alpha_v=75.0,
    ),
}


# ── Orbit presets ────────────────────────────────────────────────────────────
# Each preset defines orbital elements for a common orbit type.
# Usage:  orbit_configs=[dict(**ORBIT_PRESETS["sso_700"], num_sats=5)]
ORBIT_PRESETS = {
    "sso_700": dict(a=7078.0, e=0.001, inc=98.0, raan=0.0, omega=0.0),
    "sso_500": dict(a=6878.0, e=0.001, inc=97.4, raan=0.0, omega=0.0),
    "sso_800": dict(a=7178.0, e=0.001, inc=98.6, raan=0.0, omega=0.0),
    "leo_equatorial": dict(a=6771.0, e=0.001, inc=0.0, raan=0.0, omega=0.0),
    "polar_700": dict(a=7078.0, e=0.001, inc=90.0, raan=0.0, omega=0.0),
    "iss_like": dict(a=6778.0, e=0.001, inc=51.6, raan=0.0, omega=0.0),
    "meo": dict(a=26560.0, e=0.001, inc=55.0, raan=0.0, omega=0.0),
    "molniya": dict(a=26600.0, e=0.74, inc=63.4, raan=0.0, omega=270.0),
}


class OrbitalEyesCooperative(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        num_agents=4,
        num_rso=240,
        max_steps=4000,
        dt=5.0,
        action_mode=0,
        rso_top_k=60,
        propagation_mode=0,
        fixed_rso_slots=0,
        ground_sensor=1,
        ground_lat_deg=0.0,
        ground_lon_deg=0.0,
        ground_alt_km=0.0,
        sensor_configs=None,
        # Type-level sensor capabilities. Legacy optical/radar names map to
        # Type A/B so older configs remain loadable. Per-sensor modality and
        # location are supplied by sensor_configs or flat sensor_N_* INI keys.
        fov_deg=6.0,
        max_slew_rate_deg=0.5,
        optical_fov_deg=6.0,
        optical_max_slew_rate_deg=0.5,
        optical_max_obs_range=3200.0,
        optical_sigma_los=0.08,
        optical_sigma_cross=0.08,
        radar_fov_deg=6.0,
        radar_max_slew_rate_deg=0.5,
        radar_max_obs_range=3200.0,
        radar_sigma_los=0.08,
        radar_sigma_cross=0.08,
        radar_sigma_rate=0.0,
        # Satellite orbit constellation configuration
        # orbit_configs: list of dicts, each with keys:
        #   a (km), e, inc (deg), raan (deg), omega (deg), num_sats (int)
        # If None, creates a single SSO orbit with all agents.
        # Use ORBIT_PRESETS for common orbit types, e.g.:
        #   orbit_configs=[
        #       dict(**ORBIT_PRESETS["sso_700"], num_sats=5),
        #       dict(**ORBIT_PRESETS["polar_700"], num_sats=5, raan=90.0),
        #   ]
        orbit_configs=None,
        # Legacy single-orbit params (used only when orbit_configs is None)
        sat_a=7078.0,
        sat_e=0.001,
        sat_inc=98.0,
        sat_raan=0.0,
        sat_omega=0.0,
        # RSO population bounds
        rso_a_min=6800.0,
        rso_a_max=7200.0,
        rso_e_min=0.0,
        rso_e_max=0.02,
        rso_inc_min=0.0,
        rso_inc_max=100.0,
        rso_raan_min=0.0,
        rso_raan_max=360.0,
        rso_omega_min=0.0,
        rso_omega_max=360.0,
        rso_size_min=0.1,
        rso_size_max=5.0,
        # When enabled, each RSO plane crosses a random point inside this
        # spherical ground cap with a uniformly random local heading.
        rso_common_dome_mode=0,
        rso_dome_center_lat_deg=0.0,
        rso_dome_center_lon_deg=0.0,
        rso_dome_radius_deg=8.0,
        # Uncertainty model — LEO optical SSA regime
        # growth_a: base position uncertainty growth (km/s)
        # growth_b: age-dependent growth acceleration (km/s²)
        # alpha: observation reduction per quality=1 obs (km or km/s)
        growth_mode=0,
        growth_a_r=2e-5,       # radial base growth (km/s)
        growth_a_t=8e-5,       # along-track base growth (km/s)
        growth_a_n=2e-5,       # cross-track base growth (km/s)
        growth_b_r=2e-8,       # radial age-dependent accel (km/s²)
        growth_b_t=1e-7,       # along-track age-dependent accel (km/s²)
        growth_b_n=2e-8,       # cross-track age-dependent accel (km/s²)
        k_v=1e-6,              # velocity uncertainty growth (km/s per s)
        reduction_mode=0,
        alpha_r=0.05,          # radial reduction per obs (km)
        alpha_t=0.03,          # along-track reduction per obs (km)
        alpha_n=0.05,          # cross-track reduction per obs (km)
        alpha_vel=2e-4,        # velocity reduction per obs (km/s)
        u_min_r=0.02,          # radial uncertainty floor (km)
        u_min_t=0.02,          # along-track uncertainty floor (km)
        u_min_n=0.02,          # cross-track uncertainty floor (km)
        vel_u_min=2e-5,        # velocity uncertainty floor (km/s)
        # EKF backend
        #   uncertainty_mode = 0 : surrogate only (cheapest)
        #   uncertainty_mode = 1 : EKF + surrogate hybrid (EKF runs only on
        #                          observation events; surrogate carries
        #                          uncertainty forward between obs)
        #   uncertainty_mode = 2 : full-catalogue fixed-dt EKF predictor
        #                          with measurement updates on observed RSOs
        uncertainty_mode=0,
        ekf_sigma_a=1e-7,      # process-noise accel std (km/s²)
        ekf_meas_sigma=0.05,   # position measurement std at q_eff=1 (km)
        ekf_max_substep_dt=60.0,  # cap (s) on a single covariance propagation
                                  # jump; long gaps are split into sub-steps so
                                  # the fp32 expm stays in range. 0 = single jump
        ekf_sigma_max=50.0,    # optional hard clamp (km) on propagated RTN
                               # position sigma. 0 = disabled
        # Initial uncertainty ranges — each RSO sampled in [min, max] at reset.
        # Set max == min (or max = 0) to keep a fixed value (backwards-compat).
        u_init_r=0.3,          # radial min (km)
        u_init_t=0.8,          # along-track min (km)
        u_init_n=0.3,          # cross-track min (km)
        u_init_r_max=0.3,      # radial max (km)
        u_init_t_max=0.8,      # along-track max (km)
        u_init_n_max=0.3,      # cross-track max (km)
        sigma_vel_init=0.001,  # velocity min (km/s)
        sigma_vel_init_max=0.001,  # velocity max (km/s)
        age_init=100.0,        # time-since-last-obs min (s)
        age_init_max=100.0,    # time-since-last-obs max (s)
        # Observation quality
        quality_mode=2,
        quality_dist_weight=0.4,
        quality_angle_weight=0.4,
        quality_size_weight=0.2,
        range_crossover_km=1900.0,
        range_transition_km=300.0,
        range_quality_floor=0.15,
        max_obs_range=10000.0,
        # Masking
        enable_earth_mask=1,
        enable_range_mask=1,
        enable_fov_obs=0,      # 0=discrete observes only selected RSO, 1=all RSOs in FOV
        # Reward
        reward_mode=1,
        alpha_u=750.0,
        alpha_v=75.0,
        alpha_local=0.0,      # weight on per-sensor fractional covariance gain
        alpha_ctrl=0.1,        # control cost: penalty for sensor slew per step
        reward_scale=0.0,      # divisor for r_direct; 0 = auto-compute from problem size
        reward_baseline=0.0,   # per-step growth cost to baseline-subtract; 0 = auto
        reward_u_ref=0.0,      # fixed U_ref for level reward; 0 = initial mean uncertainty
        reward_u_target=1.0,    # reward_mode=4 scale; separate from custody threshold
        reward_mean_target=0.25, # reward_mode=5 fine mean target (km)
        reward_tail_target=0.35, # reward_mode=5 fine top-tail target (km)
        reward_max_target=0.42,  # reward_mode=5 fine max target (km)
        reward_fine_max_start=1.0, # max-U bonus onset (km)
        reward_fine_max_target=0.75, # smooth max-U shaping transition (km)
        reward_fine_max_temperature=0.10, # transition width (km)
        reward_fine_max_weight=0.0, # additive shaping; disabled by default
        reward_fine_bonus_scale=0.0, # bounded fine max-U bonus; disabled by default
        reward_fine_mean_start=0.40, # mean-U bonus onset (km)
        reward_fine_mean_bonus_scale=0.0, # bounded fine mean-U bonus; disabled by default
        reward_clip_min=-1.0,  # lower clamp for reward; 0 = disabled
        reward_clip_max=1.0,   # upper clamp for reward; 0 = disabled
        reward_mean_weight=0.50,
        reward_tail_weight=0.35,
        reward_max_weight=0.25,
        reward_threshold_weight=0.15,
        reward_tail_frac=0.20,
        reward_delta_scale=2.0,
        reward_level_scale=0.05,
        reward_terminal_bonus=2.0,
        uncertainty_threshold=3.0,
        team_spirit=1.0,      # 0=individual local gain, 1=team-mean local gain
        redundancy_penalty=0.02,
        # Priority weights
        priority_alpha=1.0,
        priority_beta=0.5,
        priority_gamma=0.1,
        # Normalization
        obs_dist_norm=10000.0,
        obs_vel_norm=10.0,
        obs_age_norm=1000.0,
        obs_u_norm=1.0,
        obs_u_vel_norm=0.001,
        # Rendering
        render_scale=1.0,
        render_rso_scale=1.0,  # RSO visual size multiplier (< 1 to shrink, > 1 to enlarge)
        render_fps=15,
        render_mode=None,
        report_interval=256,
        buf=None,
        seed=0,
        **kwargs,
    ):
        if not ground_sensor:
            raise ValueError(
                "OrbitalEyesCooperative is intentionally a fixed-ground scenario"
            )

        # Resolve per-sensor modality and location. Flat INI syntax is:
        # sensor_0_type, sensor_0_lat_deg, sensor_0_lon_deg, sensor_0_alt_km.
        if sensor_configs is None and "num_sensors" in kwargs:
            num_sensors = int(kwargs.pop("num_sensors"))
            if num_sensors != num_agents:
                raise ValueError(
                    f"num_sensors={num_sensors} must equal num_agents={num_agents}"
                )
            sensor_configs = []
            for sensor_id in range(num_sensors):
                config = {}
                for field in ("type", "lat_deg", "lon_deg", "alt_km"):
                    key = f"sensor_{sensor_id}_{field}"
                    if key not in kwargs:
                        raise ValueError(f"Missing required sensor config key: {key}")
                    value = kwargs.pop(key)
                    config[field] = int(value) if field == "type" else float(value)
                sensor_configs.append(config)
        if sensor_configs is None:
            sensor_configs = [
                dict(type=0, lat_deg=-4.0, lon_deg=-4.0, alt_km=0.0),
                dict(type=0, lat_deg=4.0, lon_deg=4.0, alt_km=0.0),
                dict(type=1, lat_deg=-4.0, lon_deg=4.0, alt_km=0.0),
                dict(type=1, lat_deg=4.0, lon_deg=-4.0, alt_km=0.0),
            ]
        if len(sensor_configs) != num_agents:
            raise ValueError(
                f"Expected {num_agents} sensor configurations, got {len(sensor_configs)}"
            )
        for sensor_id, config in enumerate(sensor_configs):
            if int(config["type"]) not in (0, 1):
                raise ValueError(
                    f"sensor {sensor_id} type must be 0 (near-range Type A) or 1 (far-range Type B)"
                )

        # ── Resolve orbit configuration ──
        # Priority: orbit_configs > flat orbit_N_* kwargs > legacy sat_* params
        if orbit_configs is None and "num_orbits" in kwargs:
            # Flat config syntax (e.g., from .ini files):
            #   num_orbits = 2
            #   orbit_0_a = 7078.0, orbit_0_e = 0.001, ...
            n_orb = int(kwargs.pop("num_orbits"))
            orbit_configs = []
            for o in range(n_orb):
                oc = {}
                for field in ("a", "e", "inc", "raan", "omega", "num_sats"):
                    key = f"orbit_{o}_{field}"
                    assert key in kwargs, (
                        f"num_orbits={n_orb} but missing '{key}' in config"
                    )
                    val = kwargs.pop(key)
                    oc[field] = int(val) if field == "num_sats" else float(val)
                orbit_configs.append(oc)

        if orbit_configs is None:
            # Backward compatible: single orbit from legacy sat_* params
            orbit_configs = [dict(
                a=sat_a, e=sat_e, inc=sat_inc,
                raan=sat_raan, omega=sat_omega,
                num_sats=num_agents,
            )]

        # Validate orbit_configs
        assert len(orbit_configs) >= 1, "At least one orbit must be defined"
        assert len(orbit_configs) <= 8, f"Maximum 8 orbits supported, got {len(orbit_configs)}"
        total_sats = sum(oc["num_sats"] for oc in orbit_configs)
        assert total_sats == num_agents, (
            f"Total satellites across orbits ({total_sats}) must equal "
            f"num_agents ({num_agents})"
        )
        for i, oc in enumerate(orbit_configs):
            for key in ("a", "e", "inc", "raan", "omega", "num_sats"):
                assert key in oc, f"orbit_configs[{i}] missing required key '{key}'"

        assert rso_common_dome_mode in (0, 1), (
            "rso_common_dome_mode must be 0 or 1"
        )
        assert -90.0 <= rso_dome_center_lat_deg <= 90.0
        assert 0.0 <= rso_dome_radius_deg < 90.0

        num_orbits = len(orbit_configs)

        # Observation: self(18) + teammates(N-1)*11 + RSO tokens(K)*19.
        obs_size = 18 + 11 * (num_agents - 1) + 19 * rso_top_k

        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32,
        )

        if action_mode == 0:
            # Discrete: select from top-K RSOs, plus one hold/no-op action
            self.single_action_space = gymnasium.spaces.Discrete(rso_top_k + 1)
        else:
            # Continuous: delta azimuth, delta elevation
            self.single_action_space = gymnasium.spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32,
            )

        self.num_agents = num_envs * num_agents
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.tick = 0
        self.action_mode = action_mode
        self._agents_per_env = num_agents

        super().__init__(buf)

        # Ensure correct dtypes for C binding
        if action_mode == 0:
            self.actions = self.actions.astype(np.int32)
        else:
            self.actions = self.actions.astype(np.float32)

        # Flatten orbit and sensor configs to indexed kwargs for C binding.
        orbit_kwargs = {"num_orbits": num_orbits}
        for o, oc in enumerate(orbit_configs):
            orbit_kwargs[f"orbit_{o}_a"] = oc["a"]
            orbit_kwargs[f"orbit_{o}_e"] = oc["e"]
            orbit_kwargs[f"orbit_{o}_inc"] = oc["inc"]
            orbit_kwargs[f"orbit_{o}_raan"] = oc["raan"]
            orbit_kwargs[f"orbit_{o}_omega"] = oc["omega"]
            orbit_kwargs[f"orbit_{o}_num_sats"] = oc["num_sats"]

        sensor_kwargs = {}
        for sensor_id, config in enumerate(sensor_configs):
            sensor_kwargs[f"sensor_{sensor_id}_type"] = int(config["type"])
            sensor_kwargs[f"sensor_{sensor_id}_lat_deg"] = float(config["lat_deg"])
            sensor_kwargs[f"sensor_{sensor_id}_lon_deg"] = float(config["lon_deg"])
            sensor_kwargs[f"sensor_{sensor_id}_alt_km"] = float(config["alt_km"])
        self.sensor_configs = tuple(dict(config) for config in sensor_configs)

        c_envs = []
        for i in range(num_envs):
            lo = i * num_agents
            hi = (i + 1) * num_agents
            c_envs.append(binding.env_init(
                self.observations[lo:hi],
                self.actions[lo:hi],
                self.rewards[lo:hi],
                self.terminals[lo:hi],
                self.truncations[lo:hi],
                i,
                num_agents=num_agents,
                num_rso=num_rso,
                max_steps=max_steps,
                dt=dt,
                action_mode=action_mode,
                rso_top_k=rso_top_k,
                propagation_mode=propagation_mode,
                fixed_rso_slots=fixed_rso_slots,
                ground_sensor=ground_sensor,
                ground_lat_deg=ground_lat_deg,
                ground_lon_deg=ground_lon_deg,
                ground_alt_km=ground_alt_km,
                fov_deg=fov_deg,
                max_slew_rate_deg=max_slew_rate_deg,
                optical_fov_deg=optical_fov_deg,
                optical_max_slew_rate_deg=optical_max_slew_rate_deg,
                optical_max_obs_range=optical_max_obs_range,
                optical_sigma_los=optical_sigma_los,
                optical_sigma_cross=optical_sigma_cross,
                radar_fov_deg=radar_fov_deg,
                radar_max_slew_rate_deg=radar_max_slew_rate_deg,
                radar_max_obs_range=radar_max_obs_range,
                radar_sigma_los=radar_sigma_los,
                radar_sigma_cross=radar_sigma_cross,
                radar_sigma_rate=radar_sigma_rate,
                **orbit_kwargs,
                **sensor_kwargs,
                rso_a_min=rso_a_min,
                rso_a_max=rso_a_max,
                rso_e_min=rso_e_min,
                rso_e_max=rso_e_max,
                rso_inc_min=rso_inc_min,
                rso_inc_max=rso_inc_max,
                rso_raan_min=rso_raan_min,
                rso_raan_max=rso_raan_max,
                rso_omega_min=rso_omega_min,
                rso_omega_max=rso_omega_max,
                rso_size_min=rso_size_min,
                rso_size_max=rso_size_max,
                rso_common_dome_mode=rso_common_dome_mode,
                rso_dome_center_lat_deg=rso_dome_center_lat_deg,
                rso_dome_center_lon_deg=rso_dome_center_lon_deg,
                rso_dome_radius_deg=rso_dome_radius_deg,
                growth_mode=growth_mode,
                growth_a_r=growth_a_r,
                growth_a_t=growth_a_t,
                growth_a_n=growth_a_n,
                growth_b_r=growth_b_r,
                growth_b_t=growth_b_t,
                growth_b_n=growth_b_n,
                k_v=k_v,
                reduction_mode=reduction_mode,
                alpha_r=alpha_r,
                alpha_t=alpha_t,
                alpha_n=alpha_n,
                alpha_vel=alpha_vel,
                u_min_r=u_min_r,
                u_min_t=u_min_t,
                u_min_n=u_min_n,
                vel_u_min=vel_u_min,
                uncertainty_mode=uncertainty_mode,
                ekf_sigma_a=ekf_sigma_a,
                ekf_meas_sigma=ekf_meas_sigma,
                ekf_max_substep_dt=ekf_max_substep_dt,
                ekf_sigma_max=ekf_sigma_max,
                u_init_r=u_init_r,
                u_init_t=u_init_t,
                u_init_n=u_init_n,
                u_init_r_max=u_init_r_max,
                u_init_t_max=u_init_t_max,
                u_init_n_max=u_init_n_max,
                sigma_vel_init=sigma_vel_init,
                sigma_vel_init_max=sigma_vel_init_max,
                age_init=age_init,
                age_init_max=age_init_max,
                quality_mode=quality_mode,
                quality_dist_weight=quality_dist_weight,
                quality_angle_weight=quality_angle_weight,
                quality_size_weight=quality_size_weight,
                range_crossover_km=range_crossover_km,
                range_transition_km=range_transition_km,
                range_quality_floor=range_quality_floor,
                max_obs_range=max_obs_range,
                enable_earth_mask=enable_earth_mask,
                enable_range_mask=enable_range_mask,
                enable_fov_obs=enable_fov_obs,
                reward_mode=reward_mode,
                alpha_u=alpha_u,
                alpha_v=alpha_v,
                alpha_local=alpha_local,
                alpha_ctrl=alpha_ctrl,
                reward_scale=reward_scale,
                reward_baseline=reward_baseline,
                reward_u_ref=reward_u_ref,
                reward_u_target=reward_u_target,
                reward_mean_target=reward_mean_target,
                reward_tail_target=reward_tail_target,
                reward_max_target=reward_max_target,
                reward_fine_max_start=reward_fine_max_start,
                reward_fine_max_target=reward_fine_max_target,
                reward_fine_max_temperature=reward_fine_max_temperature,
                reward_fine_max_weight=reward_fine_max_weight,
                reward_fine_bonus_scale=reward_fine_bonus_scale,
                reward_fine_mean_start=reward_fine_mean_start,
                reward_fine_mean_bonus_scale=reward_fine_mean_bonus_scale,
                reward_clip_min=reward_clip_min,
                reward_clip_max=reward_clip_max,
                reward_mean_weight=reward_mean_weight,
                reward_tail_weight=reward_tail_weight,
                reward_max_weight=reward_max_weight,
                reward_threshold_weight=reward_threshold_weight,
                reward_tail_frac=reward_tail_frac,
                reward_delta_scale=reward_delta_scale,
                reward_level_scale=reward_level_scale,
                reward_terminal_bonus=reward_terminal_bonus,
                uncertainty_threshold=uncertainty_threshold,
                team_spirit=team_spirit,
                redundancy_penalty=redundancy_penalty,
                priority_alpha=priority_alpha,
                priority_beta=priority_beta,
                priority_gamma=priority_gamma,
                obs_dist_norm=obs_dist_norm,
                obs_vel_norm=obs_vel_norm,
                obs_age_norm=obs_age_norm,
                obs_u_norm=obs_u_norm,
                obs_u_vel_norm=obs_u_vel_norm,
                render_scale=render_scale,
                render_rso_scale=render_rso_scale,
                render_fps=render_fps,
            ))

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        self.tick = 0
        if seed is None:
            seed = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                info.append(log_data)

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


class OrbitalEyesCooperativeAttention(OrbitalEyesCooperative):
    """Training alias for the modality-aware attention-pointer policy."""


class OrbitalEyesCooperativeLSTM(OrbitalEyesCooperative):
    """Training alias for the flat recurrent baseline."""


def test_performance(timeout=10, atn_cache=1024):
    env = OrbitalEyesCooperative(num_envs=10)
    env.reset()
    tick = 0
    actions = [env.action_space.sample() for _ in range(atn_cache)]

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {env.num_agents * tick / (time.time() - start)}")


if __name__ == "__main__":
    test_performance()

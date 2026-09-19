/*
 * Orbital Eyes: Cooperative SSA Sensor Scheduling
 *
 * Multiple observing satellites must cooperatively track a large population of
 * Resident Space Objects (RSOs) by deciding where to point their sensors in
 * order to minimise global uncertainty.
 *
 * Action modes:
 *   0 = Discrete (select RSO index from top-K priority set)
 *   1 = Continuous (delta azimuth, delta elevation)
 *
 * Observation layout per agent (flat):
 *   [self(18) | other_agents(N-1)*11 | rso_tokens(K)*19]
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include "raylib.h"
#include "raymath.h"

// ─── Profiling infrastructure ───
#ifdef PROFILE_ENV
static inline double _prof_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#define PROF_START(var) double _prof_##var = _prof_now()
#define PROF_END(env, var) (env)->prof_##var += _prof_now() - _prof_##var
#else
#define PROF_START(var) ((void)0)
#define PROF_END(env, var) ((void)0)
#endif

// ─── Constants ───
#ifndef PI
#define PI 3.14159265358979323846f
#endif

#define EARTH_RADIUS_KM  6371.0f
#define EARTH_MU         398600.4418f  // km^3/s^2
#define J2_COEFF         1.08263e-3f
#ifndef DEG2RAD
#define DEG2RAD          (PI / 180.0f)
#endif
#ifndef RAD2DEG
#define RAD2DEG          (180.0f / PI)
#endif

// Multi-orbit support
#define MAX_ORBITS 8
#define MAX_SENSORS 16
#define SENSOR_TYPE_A 0  // near-range specialist
#define SENSOR_TYPE_B 1  // far-range specialist
// Backwards-compatible aliases: modality IDs remain 0 and 1.
#define SENSOR_OPTICAL SENSOR_TYPE_A
#define SENSOR_RADAR   SENSOR_TYPE_B

// Observation feature sizes
#define SELF_OBS_SIZE     18
#define OTHER_AGENT_SIZE  11
#define RSO_TOKEN_SIZE    19  // 17 base + 2 angular direction (delta_az, delta_el)

// ─── Utility ───
static inline float randf(float lo, float hi) {
    return lo + (hi - lo) * (float)rand() / (float)RAND_MAX;
}

// Compute sin and cos in one call when the platform provides sincosf
// (glibc under _GNU_SOURCE, which Python.h enables). Falls back to two calls.
static inline void sincos_f(float x, float* s, float* c) {
#if defined(_GNU_SOURCE)
    sincosf(x, s, c);
#else
    *s = sinf(x);
    *c = cosf(x);
#endif
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float vec3_dot(float ax, float ay, float az,
                             float bx, float by, float bz) {
    return ax * bx + ay * by + az * bz;
}

static inline float vec3_len(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z);
}

static inline float vec3_dist(float ax, float ay, float az,
                              float bx, float by, float bz) {
    float dx = ax - bx, dy = ay - by, dz = az - bz;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

static inline void vec3_normalize(float* x, float* y, float* z) {
    float len = vec3_len(*x, *y, *z);
    if (len > 1e-12f) {
        *x /= len; *y /= len; *z /= len;
    }
}

static inline void vec3_cross(float ax, float ay, float az,
                               float bx, float by, float bz,
                               float* cx, float* cy, float* cz) {
    *cx = ay * bz - az * by;
    *cy = az * bx - ax * bz;
    *cz = ax * by - ay * bx;
}

// ─── Kepler equation solver (Newton-Raphson) ───
// Solve M = E - e*sin(E) for E given mean anomaly M and eccentricity e
static float solve_kepler(float M, float e) {
    // Normalize M to [0, 2π)
    M = fmodf(M, 2.0f * PI);
    if (M < 0.0f) M += 2.0f * PI;

    float E = M;  // Initial guess
    for (int iter = 0; iter < 20; iter++) {
        float dE = (E - e * sinf(E) - M) / (1.0f - e * cosf(E));
        E -= dE;
        if (fabsf(dE) < 1e-8f) break;
    }
    return E;
}

// ─── Keplerian orbital elements to ECI position and velocity ───
// elements: [a(km), e, i(rad), RAAN(rad), omega(rad), M0(rad)]
// t: time since epoch (seconds)
// propagation_mode: 0=keplerian, 1=keplerian+J2
static void orbital_elements_to_eci(
    float a, float e, float inc, float raan, float omega, float M0,
    float t, int propagation_mode,
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz)
{
    // Mean motion
    float n = sqrtf(EARTH_MU / (a * a * a));

    // J2 secular perturbations (on RAAN and omega)
    float raan_dot = 0.0f, omega_dot = 0.0f;
    if (propagation_mode >= 1) {
        float p = a * (1.0f - e * e);
        float cos_i = cosf(inc);
        float sin_i = sinf(inc);
        float factor = -1.5f * n * J2_COEFF * (EARTH_RADIUS_KM * EARTH_RADIUS_KM) / (p * p);
        raan_dot = factor * cos_i;
        omega_dot = factor * (2.5f * sin_i * sin_i - 2.0f);
    }

    // Perturbed RAAN and omega
    float raan_t = raan + raan_dot * t;
    float omega_t = omega + omega_dot * t;

    // Mean anomaly at time t
    float M = M0 + n * t;

    // Eccentric anomaly
    float E = solve_kepler(M, e);

    // True anomaly
    float cos_E = cosf(E);
    float sin_E = sinf(E);
    float nu = atan2f(sqrtf(1.0f - e * e) * sin_E, cos_E - e);

    // Distance
    float r = a * (1.0f - e * cos_E);

    // Position in orbital plane
    float x_orb = r * cosf(nu);
    float y_orb = r * sinf(nu);

    // Velocity in orbital plane
    float v_factor = sqrtf(EARTH_MU * a) / r;
    float vx_orb = -v_factor * sin_E;
    float vy_orb = v_factor * sqrtf(1.0f - e * e) * cos_E;

    // Rotation from orbital plane to ECI
    float cos_O = cosf(raan_t), sin_O = sinf(raan_t);
    float cos_w = cosf(omega_t), sin_w = sinf(omega_t);
    float cos_i = cosf(inc),     sin_i = sinf(inc);

    // Rotation matrix columns (perifocal → ECI)
    float Px = cos_O * cos_w - sin_O * sin_w * cos_i;
    float Py = sin_O * cos_w + cos_O * sin_w * cos_i;
    float Pz = sin_w * sin_i;

    float Qx = -cos_O * sin_w - sin_O * cos_w * cos_i;
    float Qy = -sin_O * sin_w + cos_O * cos_w * cos_i;
    float Qz = cos_w * sin_i;

    // ECI position
    *px = Px * x_orb + Qx * y_orb;
    *py = Py * x_orb + Qy * y_orb;
    *pz = Pz * x_orb + Qz * y_orb;

    // ECI velocity
    *vx = Px * vx_orb + Qx * vy_orb;
    *vy = Py * vx_orb + Qy * vy_orb;
    *vz = Pz * vx_orb + Qz * vy_orb;
}

// ─── Precompute orbital invariants for fast propagation ───
// Call once per object after setting orbital elements.
static inline void precompute_orbital_cache(
    float a, float e, float inc, int propagation_mode,
    float* out_n, float* out_sqrt_mu_a, float* out_sqrt_1_e2,
    float* out_cos_inc, float* out_sin_inc,
    float* out_raan_dot, float* out_omega_dot)
{
    float n = sqrtf(EARTH_MU / (a * a * a));
    *out_n = n;
    *out_sqrt_mu_a = sqrtf(EARTH_MU * a);
    *out_sqrt_1_e2 = sqrtf(1.0f - e * e);
    *out_cos_inc = cosf(inc);
    *out_sin_inc = sinf(inc);

    *out_raan_dot = 0.0f;
    *out_omega_dot = 0.0f;
    if (propagation_mode >= 1) {
        float p = a * (1.0f - e * e);
        float factor = -1.5f * n * J2_COEFF * (EARTH_RADIUS_KM * EARTH_RADIUS_KM) / (p * p);
        *out_raan_dot = factor * (*out_cos_inc);
        *out_omega_dot = factor * (2.5f * (*out_sin_inc) * (*out_sin_inc) - 2.0f);
    }
}

// ─── Fast Kepler solver for low-eccentricity orbits ───
// For e < 0.01: 2 Newton iterations are guaranteed to converge to < 1e-12.
// Falls back to standard iterative solver for higher eccentricities.
static inline float solve_kepler_fast(float M, float e) {
    // Normalize M to [0, 2π)
    M = fmodf(M, 2.0f * PI);
    if (M < 0.0f) M += 2.0f * PI;

    float E = M;
    if (e < 0.01f) {
        // Two Newton-Raphson steps (sufficient for low-e)
        float dE = (E - e * sinf(E) - M) / (1.0f - e * cosf(E));
        E -= dE;
        dE = (E - e * sinf(E) - M) / (1.0f - e * cosf(E));
        E -= dE;
    } else {
        for (int iter = 0; iter < 20; iter++) {
            float dE = (E - e * sinf(E) - M) / (1.0f - e * cosf(E));
            E -= dE;
            if (fabsf(dE) < 1e-8f) break;
        }
    }
    return E;
}

// ─── Fast orbital propagation using cached invariants ───
// Eliminates: sqrt(mu/a³), sqrt(mu*a), sqrt(1-e²), cos(inc), sin(inc),
//             atan2(nu), cos(nu), sin(nu), and J2 rate recomputation.
static inline void propagate_cached(
    float a, float e, float raan, float omega, float M0,
    float t,
    // Cached invariants:
    float n, float sqrt_mu_a, float sqrt_1_e2,
    float cos_inc, float sin_inc,
    float raan_dot, float omega_dot,
    // Output:
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz)
{
    // Perturbed RAAN and omega (raan_dot/omega_dot are 0 if no J2)
    float raan_t = raan + raan_dot * t;
    float omega_t = omega + omega_dot * t;

    // Mean anomaly at time t
    float M = M0 + n * t;

    // Eccentric anomaly (fast path for low-e)
    float E = solve_kepler_fast(M, e);

    float cos_E = cosf(E);
    float sin_E = sinf(E);

    // Distance
    float r = a * (1.0f - e * cos_E);

    // Position in orbital plane (direct from E — no true anomaly needed)
    // x_orb = a*(cos_E - e),  y_orb = a*sqrt(1-e²)*sin_E
    float x_orb = a * (cos_E - e);
    float y_orb = a * sqrt_1_e2 * sin_E;

    // Velocity in orbital plane
    float v_factor = sqrt_mu_a / r;
    float vx_orb = -v_factor * sin_E;
    float vy_orb =  v_factor * sqrt_1_e2 * cos_E;

    // Rotation from orbital plane to ECI
    float cos_O = cosf(raan_t), sin_O = sinf(raan_t);
    float cos_w = cosf(omega_t), sin_w = sinf(omega_t);

    // Rotation matrix columns (perifocal → ECI)
    float Px = cos_O * cos_w - sin_O * sin_w * cos_inc;
    float Py = sin_O * cos_w + cos_O * sin_w * cos_inc;
    float Pz = sin_w * sin_inc;

    float Qx = -cos_O * sin_w - sin_O * cos_w * cos_inc;
    float Qy = -sin_O * sin_w + cos_O * cos_w * cos_inc;
    float Qz = cos_w * sin_inc;

    // ECI position
    *px = Px * x_orb + Qx * y_orb;
    *py = Py * x_orb + Qy * y_orb;
    *pz = Pz * x_orb + Qz * y_orb;

    // ECI velocity
    *vx = Px * vx_orb + Qx * vy_orb;
    *vy = Py * vx_orb + Qy * vy_orb;
    *vz = Pz * vx_orb + Qz * vy_orb;
}

// ─── Fast Kepler solver returning cos/sin of the eccentric anomaly ───
// Same iteration as solve_kepler_fast but reuses the (sin, cos) pair produced
// by sincosf at each Newton step and returns the values at the converged E,
// so the caller does not recompute cosf(E)/sinf(E). Numerically equivalent to
// solve_kepler_fast followed by cosf(E)/sinf(E).
static inline void solve_kepler_fast_sc(float M, float e,
                                        float* out_cosE, float* out_sinE) {
    M = fmodf(M, 2.0f * PI);
    if (M < 0.0f) M += 2.0f * PI;

    float E = M;
    float s, c;
    if (e < 0.01f) {
        sincos_f(E, &s, &c);
        E -= (E - e * s - M) / (1.0f - e * c);
        sincos_f(E, &s, &c);
        E -= (E - e * s - M) / (1.0f - e * c);
    } else {
        for (int iter = 0; iter < 20; iter++) {
            sincos_f(E, &s, &c);
            float dE = (E - e * s - M) / (1.0f - e * c);
            E -= dE;
            if (fabsf(dE) < 1e-8f) break;
        }
    }
    sincos_f(E, out_sinE, out_cosE);
}

// Compute the (constant) perifocal→ECI rotation columns for a fixed orbit.
static inline void precompute_rotation_columns(
    float raan, float omega, float cos_inc, float sin_inc,
    float* Px, float* Py, float* Pz, float* Qx, float* Qy, float* Qz)
{
    float cos_O, sin_O, cos_w, sin_w;
    sincos_f(raan, &sin_O, &cos_O);
    sincos_f(omega, &sin_w, &cos_w);
    *Px = cos_O * cos_w - sin_O * sin_w * cos_inc;
    *Py = sin_O * cos_w + cos_O * sin_w * cos_inc;
    *Pz = sin_w * sin_inc;
    *Qx = -cos_O * sin_w - sin_O * cos_w * cos_inc;
    *Qy = -sin_O * sin_w + cos_O * cos_w * cos_inc;
    *Qz = cos_w * sin_inc;
}

// ─── Fast propagation with a pre-rotated perifocal→ECI basis ───
// When RAAN and argument-of-perigee are constant over the episode
// (propagation_mode == 0, no J2 secular drift), the perifocal→ECI rotation
// columns (Px,Py,Pz,Qx,Qy,Qz) are episode-invariants computed once at reset.
// This routine then only solves Kepler and applies the cached basis,
// eliminating four transcendental evaluations (cos/sin of RAAN and omega) plus
// the rotation-matrix assembly on every step.
static inline void propagate_cached_rot(
    float a, float e, float M0, float t,
    float n, float sqrt_mu_a, float sqrt_1_e2,
    float Px, float Py, float Pz, float Qx, float Qy, float Qz,
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz)
{
    float M = M0 + n * t;
    float cos_E, sin_E;
    solve_kepler_fast_sc(M, e, &cos_E, &sin_E);

    float r = a * (1.0f - e * cos_E);
    float x_orb = a * (cos_E - e);
    float y_orb = a * sqrt_1_e2 * sin_E;

    float v_factor = sqrt_mu_a / r;
    float vx_orb = -v_factor * sin_E;
    float vy_orb =  v_factor * sqrt_1_e2 * cos_E;

    *px = Px * x_orb + Qx * y_orb;
    *py = Py * x_orb + Qy * y_orb;
    *pz = Pz * x_orb + Qz * y_orb;

    *vx = Px * vx_orb + Qx * vy_orb;
    *vy = Py * vx_orb + Qy * vy_orb;
    *vz = Pz * vx_orb + Qz * vy_orb;
}

// ─── Orbit configuration ───
typedef struct {
    float a;          // semi-major axis (km)
    float e;          // eccentricity
    float inc;        // inclination (degrees — converted to radians in c_reset)
    float raan;       // RAAN (degrees)
    float omega;      // argument of perigee (degrees)
    int num_sats;     // number of satellites in this orbit
} OrbitConfig;

typedef struct {
    int sensor_type;
    float lat_deg;
    float lon_deg;
    float alt_km;
} SensorConfig;

// ─── Per-satellite agent ───
typedef struct {
    float pos[3];           // ECI position (km)
    float vel[3];           // ECI velocity (km/s)
    float sensor_dir[3];    // unit vector, current pointing direction
    float prev_sensor_dir[3]; // sensor_dir before this step's action
    float fov;              // half-cone angle (radians)
    float max_slew_rate;    // rad/s
    float max_obs_range;    // km
    float meas_sigma_los;   // position sigma along line of sight (km)
    float meas_sigma_cross; // position sigma transverse to line of sight (km)
    float meas_sigma_rate;  // line-of-sight velocity sigma (km/s), <=0 disables
    int sensor_type;        // SENSOR_TYPE_A or SENSOR_TYPE_B
    float ground_lat_deg;
    float ground_lon_deg;
    float ground_alt_km;
    int last_action;        // discrete: global RSO index of selected target
    int last_action_idx;    // discrete: RSO slot 0 to K-1, hold action K
    float last_delta[2];    // continuous: last delta az/el
    int observed_any;       // 1 if this agent observed at least one RSO last step
    int orbit_id;           // which orbit this satellite belongs to
    // Orbital elements
    float oe_a, oe_e, oe_inc, oe_raan, oe_omega, oe_M0;
    // Cached orbital invariants (computed once in c_reset)
    float oe_n;              // mean motion sqrt(mu/a³)
    float oe_sqrt_mu_a;      // sqrt(mu*a)  for velocity
    float oe_sqrt_1_e2;      // sqrt(1 - e²)
    float oe_cos_inc, oe_sin_inc;
    float oe_raan_dot, oe_omega_dot;  // J2 secular rates (0 if mode 0)
    // Cached perifocal→ECI rotation columns (valid when RAAN/omega constant,
    // i.e. propagation_mode == 0); computed once in c_reset.
    float oe_Px, oe_Py, oe_Pz, oe_Qx, oe_Qy, oe_Qz;
} Satellite;

// ─── Per-RSO tracked object ───
//
// Uncertainty scheme:
//   Modes 0/1 are visibility-gated and lazy: u_r, u_t, u_n, sigma_vel are
//   anchors set at the last observation event or reset. The current value is
//   reconstructed by surrogate_u_*_now from anchor + age.
//   Mode 2 keeps those fields current every step from the propagated EKF
//   covariance; age remains time since last observation for policy/logs.
//
//   On a step where the RSO IS observed: catch up current value, apply
//   reduction, save back as new anchor, age = 0.
//   On a step where the RSO is NOT observed: only `age += dt`.
//
//   `u_anchor_sum = u_r + u_t + u_n` is cached so consumers that need the
//   summed RTN anchor (priority / reward) avoid 2 adds per read.
typedef struct {
    float pos[3];           // ECI position (km)
    float vel[3];           // ECI velocity (km/s)
    float u_r, u_t, u_n;   // RTN uncertainty ANCHORS (value at last obs)
    float sigma_vel;        // velocity uncertainty ANCHOR (value at last obs)
    float u_anchor_sum;     // cached u_r + u_t + u_n (recomputed on obs)
    float age;              // time since last observation (seconds)
    float size;             // object cross-section (affects obs quality)
    float priority;         // computed priority score (cached)
    int num_agents_observing; // how many agents currently pointing at this RSO
    // ─ EKF state (only used when env->uncertainty_mode == 1) ─
    // 6x6 ECI covariance for [r, v] stored row-major (fp32, 144 B/RSO).
    // Updated only on observation events; between obs the surrogate helpers
    // provide the current value from the u_*/sigma_vel anchors + age.
    float P[36];
    // 0 at episode reset; set to 1 after the FIRST real observation this episode.
    // Used to separate the "catalog prior gap" (age_init, large P, measurement
    // dominates → skip propagation) from intra-episode gaps (small post-update P
    // → safe to substep-propagate).
    unsigned char ekf_anchor_fresh;
    // Orbital elements
    float oe_a, oe_e, oe_inc, oe_raan, oe_omega, oe_M0;
    // Cached orbital invariants (computed once in c_reset)
    float oe_n;              // mean motion sqrt(mu/a³)
    float oe_sqrt_mu_a;      // sqrt(mu*a)  for velocity
    float oe_sqrt_1_e2;      // sqrt(1 - e²)
    float oe_cos_inc, oe_sin_inc;
    float oe_raan_dot, oe_omega_dot;  // J2 secular rates (0 if mode 0)
    // Cached perifocal→ECI rotation columns (valid when RAAN/omega constant,
    // i.e. propagation_mode == 0); computed once in c_reset.
    float oe_Px, oe_Py, oe_Pz, oe_Qx, oe_Qy, oe_Qz;
} RSO;

// ─── Episode logging (all floats, last field must be n) ───
typedef struct {
    float episode_return;
    float episode_length;
    float mean_uncertainty;
    float max_uncertainty;
    float fraction_above_threshold;
    float total_observations;
    float mean_age;
    float effective_tracked;
    float mean_step_reward;
    float final_u_ratio;
    float u_ref;
    float catalogue_badness;
    float tail_badness;
    float initial_mean_uncertainty;
    float initial_max_uncertainty;
    float initial_fraction_above_threshold;
    float initial_catalogue_badness;
    float duplicate_assignment_rate;
    float same_modality_duplicate_rate;
    float complementary_fusion_rate;
    float optical_observations;
    float radar_observations;
    float mean_local_information_gain;
    float optical_information_gain;
    float radar_information_gain;
    float mean_type_a_observation_range;
    float mean_type_b_observation_range;
    float type_a_preferred_fraction;
    float type_b_preferred_fraction;
    float crossover_observation_fraction;
    float n;
} Log;

// ─── Rendering client ───
typedef struct {
    Camera3D camera;
    float cam_yaw;
    float cam_pitch;
    float cam_dist;
} Client;

// ─── Main environment struct ───
typedef struct {
    Log log;                    // REQUIRED: must be first
    float* observations;        // REQUIRED
    int* actions;               // REQUIRED (int for discrete mode)
    float* rewards;             // REQUIRED
    unsigned char* terminals;   // REQUIRED

    // Entities
    Satellite* satellites;
    RSO* rsos;

    // Per-agent tracking: steps_since_observed[agent * num_rso + rso] = step count
    int* steps_since_observed;

    // Top-K index mapping: top_k_indices[agent * rso_top_k + k] = global RSO index
    int* top_k_indices;

    // Priority scratch buffer for sorting (avoids per-step allocation)
    int* priority_sort_indices;
    float* priority_sort_values;

    // Top-K selection output buffers (size rso_top_k each)
    int* topk_out_indices;
    float* topk_out_values;

    // Pre-computed agent-independent base priority per RSO
    float* base_priority;

    // Per-RSO reachability flag (reused per agent, size num_rso)
    int* reachable_flags;

    // Reduction accumulator: uncertainty reduction per RSO per axis
    float* reduction_buffer;  // [num_rso * 4] — r, t, n, vel
    // Per-agent/RSO measurement quality. Zero means no successful measurement.
    float* measurement_quality; // [num_agents * num_rso]
    // Independent fractional covariance reduction earned by each sensor during
    // the current step. Reward conversion clamps each value to [0, 1].
    float local_info_gain[MAX_SENSORS];

    // External-target override (benchmark only). When external_target_active!=0
    // the discrete-action branch slews each agent toward external_targets[a]
    // (a global RSO index, or <0 = no-op) instead of decoding the top-K action.
    // This grants classical baselines reachable-set access. It is NEVER enabled
    // for the neural-network policy, which always commands via the top-K head.
    int* external_targets;       // [num_agents]
    int external_target_active;  // 0 = normal top-K decode, 1 = override

    // Config: core
    int num_agents;
    int num_rso;
    int max_steps;
    float dt;
    int action_mode;        // 0=discrete, 1=continuous
    int rso_top_k;
    int propagation_mode;   // 0=keplerian, 1=keplerian+J2
    int fixed_rso_slots;    // 1 = token/action slot k always maps to global RSO k
    int ground_sensor;      // 1 = agents are fixed ground sensors, not orbiting sats
    float ground_lat_deg;
    float ground_lon_deg;
    float ground_alt_km;
    SensorConfig sensor_configs[MAX_SENSORS];

    // Config: sensor
    float fov_deg;          // sensor FOV half-angle (degrees)
    float max_slew_rate_deg; // max slew rate (deg/s)
    float optical_fov_deg;
    float optical_max_slew_rate_deg;
    float optical_max_obs_range;
    float optical_sigma_los;
    float optical_sigma_cross;
    float radar_fov_deg;
    float radar_max_slew_rate_deg;
    float radar_max_obs_range;
    float radar_sigma_los;
    float radar_sigma_cross;
    float radar_sigma_rate;

    // Config: satellite orbits (multi-orbit constellation)
    int num_orbits;
    OrbitConfig orbits[MAX_ORBITS];

    // Config: RSO population orbital element bounds
    float rso_a_min, rso_a_max;
    float rso_e_min, rso_e_max;
    float rso_inc_min, rso_inc_max;
    float rso_raan_min, rso_raan_max;
    float rso_omega_min, rso_omega_max;
    float rso_size_min, rso_size_max;
    int rso_common_dome_mode;
    float rso_dome_center_lat_deg;
    float rso_dome_center_lon_deg;
    float rso_dome_radius_deg;

    // Config: uncertainty model (LEO optical SSA regime)
    //   growth_a_{r,t,n} : base position uncertainty growth rate (km/s)
    //   growth_b_{r,t,n} : age-dependent growth acceleration  (km/s²)
    //   k_v              : velocity uncertainty growth rate    (km/s per second)
    //   alpha_{r,t,n}    : position reduction per obs * quality (km)
    //   alpha_vel        : velocity reduction per obs * quality (km/s)
    //   u_min_{r,t,n}    : position uncertainty floor          (km)
    //   vel_u_min        : velocity uncertainty floor          (km/s)
    int growth_mode;        // 0=age, 1=age_velocity
    float growth_a_r, growth_a_t, growth_a_n;
    float growth_b_r, growth_b_t, growth_b_n;
    float k_v;              // velocity uncertainty growth rate (km/s²)
    int reduction_mode;     // 0=additive, 1=multiplicative
    float alpha_r, alpha_t, alpha_n;   // observation reduction (km)
    float alpha_vel;        // velocity uncertainty reduction (km/s)
    float u_min_r, u_min_t, u_min_n;  // uncertainty floors (km)
    float vel_u_min;                   // velocity uncertainty floor (km/s)

    // Config: uncertainty backend (Step 2: hybrid EKF + surrogate)
    //   uncertainty_mode = 0 : surrogate only (closed-form anchor + additive/
    //                          multiplicative reduction; cheapest, no EKF).
    //   uncertainty_mode = 1 : EKF + surrogate hybrid. On each observation
    //                          event we propagate the per-RSO 6x6 ECI
    //                          covariance via Phi = expm(F*dt) (two-body F,
    //                          scaled-and-squared truncated Taylor), apply
    //                          a Joseph-form position-only measurement
    //                          update with R = ekf_meas_sigma^2 / q_eff^2,
    //                          then sync the surrogate anchors from
    //                          sqrt(diag(P_rtn)). Between obs events the
    //                          surrogate carries forward as in mode 0.
    int uncertainty_mode;
    float ekf_sigma_a;      // continuous-time accel process-noise std (km/s²)
    float ekf_meas_sigma;   // position measurement std at q_eff=1 (km)
    // Numerical safety for the event-driven covariance propagation:
    //   ekf_max_substep_dt : cap (s) on the dt of a single expm(F*dt) +
    //                        Phi·P·Phiᵀ propagation. Long observation gaps are
    //                        split into ceil(gap / cap) equal sub-steps so the
    //                        scaled-and-squared fp32 Taylor expm stays inside its
    //                        validated ‖F·dt‖≲3 range and the propagated
    //                        covariance does not blow up numerically. 0 = single
    //                        jump (legacy behaviour, unsafe over multi-orbit gaps).
    //   ekf_sigma_max      : optional hard clamp (km) on each propagated RTN
    //                        position sigma (and 1e3× on velocity sigma) to bound
    //                        worst-case spikes. 0 = disabled.
    float ekf_max_substep_dt;
    float ekf_sigma_max;

    // Config: initial uncertainty (km, km/s, s)
    // Each RSO's initial state is sampled uniformly in [min, max].
    // Set max == min (or max = 0) to keep the fixed legacy behaviour.
    float u_init_r, u_init_t, u_init_n;       // position uncertainty min (km)
    float u_init_r_max, u_init_t_max, u_init_n_max; // position uncertainty max (km)
    float sigma_vel_init, sigma_vel_init_max;  // velocity uncertainty range (km/s)
    float age_init, age_init_max;              // time-since-last-obs range (s)

    // Config: observation quality
    int quality_mode;       // 0=binary, 1=generic geometry, 2=range-specialized
    float quality_dist_weight;
    float quality_angle_weight;
    float quality_size_weight;
    float range_crossover_km;
    float range_transition_km;
    float range_quality_floor;
    float max_obs_range;    // km

    // Config: masking
    int enable_earth_mask;
    int enable_range_mask;
    int enable_fov_obs;     // 0=discrete observes only selected RSO, 1=all RSOs in FOV

    // Config: reward
    int reward_mode;        // 0=delta_J, 1=mean level, 2=risk potential, 3=risk level+potential, 4=tail-sensitive risk, 5=precision-target
    float alpha_u;
    float alpha_v;
    float alpha_local;
    float alpha_ctrl;       // control cost: penalty per unit of slew
    float reward_scale;     // divisor for r_direct; 0 = auto-compute from problem size
    float reward_baseline;  // per-step growth cost to add before scaling; 0 = auto
    float reward_u_ref;     // fixed reference mean uncertainty for level reward; 0 = initial mean
    float reward_u_target;  // uncertainty scale used by reward_mode=4
    float reward_mean_target; // target mean uncertainty for reward_mode=5
    float reward_tail_target; // target top-tail uncertainty for reward_mode=5
    float reward_max_target;  // target maximum uncertainty for reward_mode=5
    float reward_fine_max_start; // max-U bonus onset (km)
    float reward_fine_max_target; // transition center for fine max-U shaping
    float reward_fine_max_temperature; // smooth transition width (km)
    float reward_fine_max_weight; // additive fine-regime shaping weight
    float reward_fine_bonus_scale; // bounded per-step fine max-U bonus
    float reward_fine_mean_start; // mean-U bonus onset (km)
    float reward_fine_mean_bonus_scale; // bounded per-step fine mean-U bonus
    float reward_clip_min;  // lower clamp for per-step r_direct. 0 = disabled.
    float reward_clip_max;  // upper clamp for per-step r_direct. 0 = disabled.
    float reward_mean_weight;
    float reward_tail_weight;
    float reward_max_weight;
    float reward_threshold_weight;
    float reward_tail_frac;
    float reward_delta_scale;
    float reward_level_scale;
    float reward_terminal_bonus;
    float uncertainty_threshold;
    float team_spirit;
    float redundancy_penalty;

    // Scratch values populated by compute_precision_badness so the optional
    // fine-regime bonuses do not rescan the full RSO catalogue.
    float precision_mean_uncertainty;
    float precision_max_uncertainty;

    // Config: priority weights (for top-K selection)
    float priority_alpha;   // uncertainty weight
    float priority_beta;    // age weight
    float priority_gamma;   // visibility quality weight

    // Config: normalization
    float obs_dist_norm;    // normalization factor for distances in observations (km)
    float obs_vel_norm;     // normalization factor for velocities
    float obs_age_norm;     // normalization factor for age
    float obs_u_norm;       // normalization factor for uncertainty values
    float obs_u_vel_norm;   // normalization factor for velocity uncertainty (km/s)

    // Derived
    int obs_size;
    int self_obs_offset;
    int agents_obs_offset;
    int rso_obs_offset;

    // State
    int tick;
    float epoch_time;       // current simulation time (seconds since epoch)
    float J_prev;           // previous population objective value
    float U_ref;            // reference mean uncertainty for level-based reward
    float episode_return_accum;       // current episode return, not cleared by vec_log
    float total_observations_accum;   // current episode observations, not cleared by vec_log
    float badness_prev;              // previous risk-aware catalogue badness
    float badness_initial;           // initial risk-aware catalogue badness
    float initial_mean_uncertainty;
    float initial_max_uncertainty;
    float initial_fraction_above_threshold;
    float duplicate_assignment_accum;
    float same_modality_duplicate_accum;
    float complementary_fusion_accum;
    float optical_observations_accum;
    float radar_observations_accum;
    float local_information_gain_accum;
    float optical_information_gain_accum;
    float radar_information_gain_accum;
    float type_a_observation_range_accum;
    float type_b_observation_range_accum;
    float type_a_preferred_observations_accum;
    float type_b_preferred_observations_accum;
    float crossover_observations_accum;

    // Rendering
    Client* client;
    float render_scale;     // km to render units conversion
    float render_rso_scale; // RSO visual size multiplier (1.0 = default)
    int render_fps;         // target FPS for rendering (adjustable at runtime with +/-)

    // Profiling accumulators (only active when compiled with -DPROFILE_ENV)
    double prof_propagation;      // orbital propagation (sat + RSO)
    double prof_actions;          // action processing / sensor slew
    double prof_observation_det;  // observation determination (FOV/visibility checks)
    double prof_uncertainty;      // uncertainty growth + reduction
    double prof_reward;           // reward computation
    double prof_build_obs;        // compute_observations (priority sort + obs assembly)
    double prof_total_step;       // total c_step time
    int    prof_step_count;       // number of steps profiled
} OrbitalEyesCooperative;

// ─── Surrogate uncertainty: lazy current-value helpers ─────────────────────
// All math lives here so swapping the surrogate (or splitting short/long-Δt
// regimes for the EKF in step 2) only touches these functions.
//
// Closed form for the current per-axis position uncertainty given the anchor:
//     u_now = u_anchor + a*age + 0.5*b*age^2
// where (a, b) = (growth_a_*, growth_b_*).
// Velocity uncertainty is linear in age: sigma_vel_now = anchor + k_v*age.
// In uncertainty_mode=2 the EKF predictor refreshes these values every step,
// so the helpers return the already-current covariance-derived sigmas.
static inline float surrogate_u_r_now(const RSO* r, const OrbitalEyesCooperative* env) {
    if (env->uncertainty_mode == 2) return r->u_r;
    float age = r->age;
    return r->u_r + (env->growth_a_r + 0.5f * env->growth_b_r * age) * age;
}
static inline float surrogate_u_t_now(const RSO* r, const OrbitalEyesCooperative* env) {
    if (env->uncertainty_mode == 2) return r->u_t;
    float age = r->age;
    return r->u_t + (env->growth_a_t + 0.5f * env->growth_b_t * age) * age;
}
static inline float surrogate_u_n_now(const RSO* r, const OrbitalEyesCooperative* env) {
    if (env->uncertainty_mode == 2) return r->u_n;
    float age = r->age;
    return r->u_n + (env->growth_a_n + 0.5f * env->growth_b_n * age) * age;
}
static inline float surrogate_sigma_vel_now(const RSO* r, const OrbitalEyesCooperative* env) {
    if (env->uncertainty_mode == 2) return r->sigma_vel;
    if (env->growth_mode >= 1) {
        return r->sigma_vel + env->k_v * r->age;
    }
    return r->sigma_vel;
}
// Sum form used by priority / reward / log aggregates. Uses cached u_anchor_sum
// so the per-axis anchors are not summed every call.
static inline float surrogate_u_sum_now(const RSO* r, const OrbitalEyesCooperative* env) {
    if (env->uncertainty_mode == 2) return r->u_anchor_sum;
    float age = r->age;
    float linear = (env->growth_a_r + env->growth_a_t + env->growth_a_n) * age;
    float quad   = 0.5f * (env->growth_b_r + env->growth_b_t + env->growth_b_n)
                   * age * age;
    return r->u_anchor_sum + linear + quad;
}

// ─── EKF helpers (Step 2: hybrid EKF + surrogate) ─────────────────────────
//
// Mathematical model (same EKF used in resources/orbital_eyes scripts):
//   x = [r_eci, v_eci]  (6-state, but we run covariance-only)
//   F = df/dx with f = two-body acceleration
//   Phi = expm(F * dt)  via scaled-and-squared truncated Taylor (fp32)
//   Q_d = discrete white-noise acceleration model
//          [dt^3/3 I3, dt^2/2 I3;  dt^2/2 I3, dt I3] * sigma_a^2
//   Measurement: position-only, H = [I3 | 0],  R = sigma_meas^2 * I3
//   Update: Joseph form.
//
// Efficiency tricks:
//   * Event-driven: P only touched on observation events (visibility edges).
//     Between obs the cheap surrogate provides the current uncertainty.
//   * fp32 throughout.
//   * 3x3 closed-form S^-1 for the position-only update (no full 6x6 solve).
//   * Scaled-and-squared Taylor for expm (no LAPACK dependency, ~5k flops).
//   * uncertainty_mode=2 uses a fixed-dt order-2 transition specialized to
//     F=[0 I; G 0], avoiding expm for per-step catalogue-wide prediction.
//   * Aggregate multiple observers into one effective measurement
//     (sigma_meas = ekf_meas_sigma / q_eff).

static inline void mat6_zero(float A[36]) { memset(A, 0, 36 * sizeof(float)); }
static inline void mat6_identity(float A[36]) {
    memset(A, 0, 36 * sizeof(float));
    for (int i = 0; i < 6; i++) A[i * 6 + i] = 1.0f;
}
static inline void mat6_copy(const float A[36], float B[36]) {
    memcpy(B, A, 36 * sizeof(float));
}
// C = A * B  (no aliasing of C with A or B)
static inline void mat6_mul(const float A[36], const float B[36], float C[36]) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            float s = 0.0f;
            for (int k = 0; k < 6; k++) s += A[i * 6 + k] * B[k * 6 + j];
            C[i * 6 + j] = s;
        }
    }
}
// C = A * B^T
static inline void mat6_mul_bt(const float A[36], const float B[36], float C[36]) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            float s = 0.0f;
            for (int k = 0; k < 6; k++) s += A[i * 6 + k] * B[j * 6 + k];
            C[i * 6 + j] = s;
        }
    }
}
static inline void mat6_symmetrize(float A[36]) {
    for (int i = 0; i < 6; i++) {
        for (int j = i + 1; j < 6; j++) {
            float s = 0.5f * (A[i * 6 + j] + A[j * 6 + i]);
            A[i * 6 + j] = s;
            A[j * 6 + i] = s;
        }
    }
}

// F = df/dx for two-body dynamics at ECI position r.
// F = [0 I; dA/dr 0]  with  dA_i/dr_j = -mu/|r|^3 δ_ij + 3 mu r_i r_j / |r|^5.
static inline void ekf_gravity_gradient(const float r[3], float G[9]) {
    float x = r[0], y = r[1], z = r[2];
    float rn2 = x * x + y * y + z * z;
    if (rn2 < 1e-18f) {
        memset(G, 0, 9 * sizeof(float));
        return;
    }
    float rn = sqrtf(rn2);
    float rn3 = rn2 * rn;
    float rn5 = rn3 * rn2;
    float k = EARTH_MU / rn3;
    float kk = 3.0f * EARTH_MU / rn5;
    G[0] = -k + kk * x * x;
    G[1] =       kk * x * y;
    G[2] =       kk * x * z;
    G[3] =       kk * y * x;
    G[4] = -k + kk * y * y;
    G[5] =       kk * y * z;
    G[6] =       kk * z * x;
    G[7] =       kk * z * y;
    G[8] = -k + kk * z * z;
}

static inline void ekf_dynamics_jacobian(const float r[3], float F[36]) {
    mat6_zero(F);
    F[0 * 6 + 3] = 1.0f;
    F[1 * 6 + 4] = 1.0f;
    F[2 * 6 + 5] = 1.0f;
    float G[9];
    ekf_gravity_gradient(r, G);
    F[3 * 6 + 0] = G[0]; F[3 * 6 + 1] = G[1]; F[3 * 6 + 2] = G[2];
    F[4 * 6 + 0] = G[3]; F[4 * 6 + 1] = G[4]; F[4 * 6 + 2] = G[5];
    F[5 * 6 + 0] = G[6]; F[5 * 6 + 1] = G[7]; F[5 * 6 + 2] = G[8];
}

// Phi = expm(F * dt) via scaled-and-squared 6-term Taylor.
// Robust for ||F dt||_inf up to ~3 (dt up to a few thousand s at LEO).
static inline void ekf_state_transition(const float F[36], float dt, float Phi[36]) {
    float A[36];
    for (int i = 0; i < 36; i++) A[i] = F[i] * dt;
    float norm = 0.0f;
    for (int i = 0; i < 6; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < 6; j++) row_sum += fabsf(A[i * 6 + j]);
        if (row_sum > norm) norm = row_sum;
    }
    int s = 0;
    while (norm > 0.5f && s < 10) { norm *= 0.5f; s++; }
    if (s > 0) {
        float scale = 1.0f / (float)(1 << s);
        for (int i = 0; i < 36; i++) A[i] *= scale;
    }
    float result[36], A2[36], A3[36], A4[36], A5[36], A6[36], tmp[36];
    mat6_identity(result);
    for (int i = 0; i < 36; i++) result[i] += A[i];
    mat6_mul(A, A, A2);
    for (int i = 0; i < 36; i++) result[i] += 0.5f * A2[i];
    mat6_mul(A2, A, A3);
    for (int i = 0; i < 36; i++) result[i] += (1.0f / 6.0f) * A3[i];
    mat6_mul(A2, A2, A4);
    for (int i = 0; i < 36; i++) result[i] += (1.0f / 24.0f) * A4[i];
    mat6_mul(A4, A, A5);
    for (int i = 0; i < 36; i++) result[i] += (1.0f / 120.0f) * A5[i];
    mat6_mul(A4, A2, A6);
    for (int i = 0; i < 36; i++) result[i] += (1.0f / 720.0f) * A6[i];
    for (int k = 0; k < s; k++) {
        mat6_mul(result, result, tmp);
        mat6_copy(tmp, result);
    }
    mat6_copy(result, Phi);
}

// In place: P <- Phi P Phi^T + Q_d  (discrete white-noise accel)
static inline void ekf_propagate_P(float P[36], const float Phi[36],
                                   float sigma_a, float dt) {
    float PhiP[36], newP[36];
    mat6_mul(Phi, P, PhiP);
    mat6_mul_bt(PhiP, Phi, newP);
    float sa2 = sigma_a * sigma_a;
    float dt2 = dt * dt;
    float q_rr = sa2 * dt2 * dt / 3.0f;
    float q_rv = sa2 * dt2 * 0.5f;
    float q_vv = sa2 * dt;
    for (int i = 0; i < 3; i++) {
        newP[i * 6 + i]             += q_rr;
        newP[(i + 3) * 6 + (i + 3)] += q_vv;
        newP[i * 6 + (i + 3)]       += q_rv;
        newP[(i + 3) * 6 + i]       += q_rv;
    }
    mat6_copy(newP, P);
    mat6_symmetrize(P);
}

// Joseph-form covariance update for position-only measurement.
//   H = [I3 | 0],  R = sigma_meas^2 * I3,  K = P[:,0:3] (P[0:3,0:3] + R)^-1.
static inline void ekf_update_position(float P[36], float sigma_meas) {
    float sm2 = sigma_meas * sigma_meas;
    float S0 = P[0 * 6 + 0] + sm2, S1 = P[0 * 6 + 1],       S2 = P[0 * 6 + 2];
    float S3 = P[1 * 6 + 0],       S4 = P[1 * 6 + 1] + sm2, S5 = P[1 * 6 + 2];
    float S6 = P[2 * 6 + 0],       S7 = P[2 * 6 + 1],       S8 = P[2 * 6 + 2] + sm2;
    float det = S0 * (S4 * S8 - S5 * S7)
              - S1 * (S3 * S8 - S5 * S6)
              + S2 * (S3 * S7 - S4 * S6);
    if (fabsf(det) < 1e-20f) return;
    float idet = 1.0f / det;
    float Si[9];
    Si[0] = (S4 * S8 - S5 * S7) * idet;
    Si[1] = (S2 * S7 - S1 * S8) * idet;
    Si[2] = (S1 * S5 - S2 * S4) * idet;
    Si[3] = (S5 * S6 - S3 * S8) * idet;
    Si[4] = (S0 * S8 - S2 * S6) * idet;
    Si[5] = (S2 * S3 - S0 * S5) * idet;
    Si[6] = (S3 * S7 - S4 * S6) * idet;
    Si[7] = (S1 * S6 - S0 * S7) * idet;
    Si[8] = (S0 * S4 - S1 * S3) * idet;
    float K[18];
    for (int i = 0; i < 6; i++) {
        float p0 = P[i * 6 + 0], p1 = P[i * 6 + 1], p2 = P[i * 6 + 2];
        K[i * 3 + 0] = p0 * Si[0] + p1 * Si[3] + p2 * Si[6];
        K[i * 3 + 1] = p0 * Si[1] + p1 * Si[4] + p2 * Si[7];
        K[i * 3 + 2] = p0 * Si[2] + p1 * Si[5] + p2 * Si[8];
    }
    float M[36];
    mat6_identity(M);
    for (int i = 0; i < 6; i++) {
        M[i * 6 + 0] -= K[i * 3 + 0];
        M[i * 6 + 1] -= K[i * 3 + 1];
        M[i * 6 + 2] -= K[i * 3 + 2];
    }
    float MP[36], newP[36];
    mat6_mul(M, P, MP);
    mat6_mul_bt(MP, M, newP);
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            float s = K[i * 3 + 0] * K[j * 3 + 0]
                    + K[i * 3 + 1] * K[j * 3 + 1]
                    + K[i * 3 + 2] * K[j * 3 + 2];
            newP[i * 6 + j] += sm2 * s;
        }
    }
    mat6_copy(newP, P);
    mat6_symmetrize(P);
}

// Joseph-form scalar covariance update for one directional measurement h*x.
static inline void ekf_update_scalar_direction(
    float P[36], const float h[6], float sigma)
{
    if (sigma <= 0.0f || !isfinite(sigma)) return;
    float ph[6] = {0};
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) ph[i] += P[i * 6 + j] * h[j];
    }
    float innovation = sigma * sigma;
    for (int i = 0; i < 6; i++) innovation += h[i] * ph[i];
    if (innovation <= 1e-20f || !isfinite(innovation)) return;

    float K[6];
    for (int i = 0; i < 6; i++) K[i] = ph[i] / innovation;
    float M[36];
    mat6_identity(M);
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) M[i * 6 + j] -= K[i] * h[j];
    }
    float MP[36], newP[36];
    mat6_mul(M, P, MP);
    mat6_mul_bt(MP, M, newP);
    float variance = sigma * sigma;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            newP[i * 6 + j] += variance * K[i] * K[j];
        }
    }
    mat6_copy(newP, P);
    mat6_symmetrize(P);
}

// Apply one sensor's directional measurement model. In the range-specialized
// experiment both modalities use the same base covariance update; their only
// distinction is the hidden distance-response factor passed as `quality`.
static inline void ekf_update_sensor_measurement(
    float P[36], const Satellite* sensor, const RSO* rso, float quality)
{
    float q = fmaxf(quality, 0.05f);
    float los[3] = {
        rso->pos[0] - sensor->pos[0],
        rso->pos[1] - sensor->pos[1],
        rso->pos[2] - sensor->pos[2],
    };
    vec3_normalize(&los[0], &los[1], &los[2]);

    float ref[3] = {0.0f, 0.0f, 1.0f};
    if (fabsf(los[2]) > 0.9f) {
        ref[0] = 1.0f; ref[1] = 0.0f; ref[2] = 0.0f;
    }
    float cross1[3], cross2[3];
    vec3_cross(los[0], los[1], los[2], ref[0], ref[1], ref[2],
               &cross1[0], &cross1[1], &cross1[2]);
    vec3_normalize(&cross1[0], &cross1[1], &cross1[2]);
    vec3_cross(los[0], los[1], los[2],
               cross1[0], cross1[1], cross1[2],
               &cross2[0], &cross2[1], &cross2[2]);
    vec3_normalize(&cross2[0], &cross2[1], &cross2[2]);

    float h[6] = {0};
    h[0] = cross1[0]; h[1] = cross1[1]; h[2] = cross1[2];
    ekf_update_scalar_direction(P, h, sensor->meas_sigma_cross / q);
    h[0] = cross2[0]; h[1] = cross2[1]; h[2] = cross2[2];
    ekf_update_scalar_direction(P, h, sensor->meas_sigma_cross / q);
    h[0] = los[0]; h[1] = los[1]; h[2] = los[2];
    ekf_update_scalar_direction(P, h, sensor->meas_sigma_los / q);

    if (sensor->meas_sigma_rate > 0.0f) {
        memset(h, 0, sizeof(h));
        h[3] = los[0]; h[4] = los[1]; h[5] = los[2];
        ekf_update_scalar_direction(P, h, sensor->meas_sigma_rate / q);
    }
}

static inline void ekf_apply_buffered_measurements(
    OrbitalEyesCooperative* env, int rso_idx)
{
    RSO* rso = &env->rsos[rso_idx];
    for (int a = 0; a < env->num_agents; a++) {
        float q = env->measurement_quality[a * env->num_rso + rso_idx];
        if (q > 0.0f) {
            ekf_update_sensor_measurement(rso->P, &env->satellites[a], rso, q);
        }
    }
}

// Extract RTN-frame 1-sigma position uncertainty from ECI covariance and state.
static inline void ekf_rtn_sigmas(const float P[36], const float r[3], const float v[3],
                                  float* sig_r, float* sig_t, float* sig_n,
                                  float* sig_v) {
    float rn = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if (rn < 1e-12f) { *sig_r = *sig_t = *sig_n = *sig_v = 0.0f; return; }
    float rhx = r[0] / rn, rhy = r[1] / rn, rhz = r[2] / rn;
    float hx = r[1] * v[2] - r[2] * v[1];
    float hy = r[2] * v[0] - r[0] * v[2];
    float hz = r[0] * v[1] - r[1] * v[0];
    float hn = sqrtf(hx * hx + hy * hy + hz * hz);
    if (hn < 1e-12f) { *sig_r = *sig_t = *sig_n = *sig_v = 0.0f; return; }
    float nhx = hx / hn, nhy = hy / hn, nhz = hz / hn;
    float thx = nhy * rhz - nhz * rhy;
    float thy = nhz * rhx - nhx * rhz;
    float thz = nhx * rhy - nhy * rhx;
    #define QF(ux, uy, uz) ( \
          (ux) * ((ux) * P[0]         + (uy) * P[1]         + (uz) * P[2])         \
        + (uy) * ((ux) * P[1 * 6 + 0] + (uy) * P[1 * 6 + 1] + (uz) * P[1 * 6 + 2]) \
        + (uz) * ((ux) * P[2 * 6 + 0] + (uy) * P[2 * 6 + 1] + (uz) * P[2 * 6 + 2]) \
    )
    float vr2 = QF(rhx, rhy, rhz);
    float vt2 = QF(thx, thy, thz);
    float vn2 = QF(nhx, nhy, nhz);
    #undef QF
    *sig_r = sqrtf(fmaxf(0.0f, vr2));
    *sig_t = sqrtf(fmaxf(0.0f, vt2));
    *sig_n = sqrtf(fmaxf(0.0f, vn2));
    float vv = (P[3 * 6 + 3] + P[4 * 6 + 4] + P[5 * 6 + 5]) / 3.0f;
    *sig_v = sqrtf(fmaxf(0.0f, vv));
}

// Counterfactual one-sensor covariance gain from the common predicted prior.
// Sensors are evaluated independently, avoiding update-order bias when several
// modalities observe the same RSO.
static inline float ekf_independent_information_gain(
    const float prior[36], const Satellite* sensor, const RSO* rso,
    float quality)
{
    float before_r, before_t, before_n, before_v;
    ekf_rtn_sigmas(prior, rso->pos, rso->vel,
                   &before_r, &before_t, &before_n, &before_v);
    float before = before_r + before_t + before_n;
    if (before <= 1e-6f || !isfinite(before)) return 0.0f;

    float posterior[36];
    mat6_copy(prior, posterior);
    ekf_update_sensor_measurement(posterior, sensor, rso, quality);
    float after_r, after_t, after_n, after_v;
    ekf_rtn_sigmas(posterior, rso->pos, rso->vel,
                   &after_r, &after_t, &after_n, &after_v);
    float after = after_r + after_t + after_n;
    if (!isfinite(after)) return 0.0f;
    return clampf((before - after) / before, 0.0f, 1.0f);
}

// Initialize P from RTN-diagonal anchors at reset:
//   P_rr_eci = T^T diag(u_r^2, u_t^2, u_n^2) T
//   P_vv_eci = sigma_v^2 * I_3,  cross blocks zero.
static inline void ekf_init_P(float P[36], const float r[3], const float v[3],
                              float ur, float ut, float un, float sv) {
    mat6_zero(P);
    float rn = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if (rn < 1e-12f) return;
    float rhx = r[0] / rn, rhy = r[1] / rn, rhz = r[2] / rn;
    float hx = r[1] * v[2] - r[2] * v[1];
    float hy = r[2] * v[0] - r[0] * v[2];
    float hz = r[0] * v[1] - r[1] * v[0];
    float hn = sqrtf(hx * hx + hy * hy + hz * hz);
    if (hn < 1e-12f) return;
    float nhx = hx / hn, nhy = hy / hn, nhz = hz / hn;
    float thx = nhy * rhz - nhz * rhy;
    float thy = nhz * rhx - nhx * rhz;
    float thz = nhx * rhy - nhy * rhx;
    float ur2 = ur * ur, ut2 = ut * ut, un2 = un * un;
    float Rh[3] = {rhx, rhy, rhz};
    float Th[3] = {thx, thy, thz};
    float Nh[3] = {nhx, nhy, nhz};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            P[i * 6 + j] = ur2 * Rh[i] * Rh[j]
                         + ut2 * Th[i] * Th[j]
                         + un2 * Nh[i] * Nh[j];
        }
    }
    float sv2 = sv * sv;
    P[3 * 6 + 3] = sv2;
    P[4 * 6 + 4] = sv2;
    P[5 * 6 + 5] = sv2;
}


static inline void mat3_mul(const float A[9], const float B[9], float C[9]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[i * 3 + j] = A[i * 3 + 0] * B[j]
                         + A[i * 3 + 1] * B[3 + j]
                         + A[i * 3 + 2] * B[6 + j];
        }
    }
}

static inline void mat3_mul_bt(const float A[9], const float B[9], float C[9]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[i * 3 + j] = A[i * 3 + 0] * B[j * 3 + 0]
                         + A[i * 3 + 1] * B[j * 3 + 1]
                         + A[i * 3 + 2] * B[j * 3 + 2];
        }
    }
}

static inline void mat3_add_scaled(float A[9], const float B[9], float scale) {
    for (int i = 0; i < 9; i++) A[i] += scale * B[i];
}

static inline void mat3_add_scaled_transpose(float A[9], const float B[9], float scale) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) A[i * 3 + j] += scale * B[j * 3 + i];
    }
}

static inline void mat3_symmetrize(float A[9]) {
    float s01 = 0.5f * (A[1] + A[3]);
    float s02 = 0.5f * (A[2] + A[6]);
    float s12 = 0.5f * (A[5] + A[7]);
    A[1] = A[3] = s01;
    A[2] = A[6] = s02;
    A[5] = A[7] = s12;
}

// Fixed-dt catalogue predictor for uncertainty_mode=2.
// Uses Phi ~= [I + 0.5 G dt^2, I dt; G dt, I + 0.5 G dt^2]
// with G=d(a_gravity)/dr. This keeps the 3-hour dt=5s covariance drift close
// to the full EKF reference while avoiding per-RSO matrix exponentials.
static inline void ekf_predict_fixed_dt_order2(float P[36], const float r[3],
                                               float sigma_a, float dt) {
    float G[9];
    ekf_gravity_gradient(r, G);

    float A[9] = {0};
    float C[9];
    float dt2 = dt * dt;
    for (int i = 0; i < 9; i++) {
        A[i] = 0.5f * G[i] * dt2;
        C[i] = G[i] * dt;
    }
    A[0] += 1.0f; A[4] += 1.0f; A[8] += 1.0f;

    float Prr[9], Prv[9], Pvv[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Prr[i * 3 + j] = P[i * 6 + j];
            Prv[i * 3 + j] = P[i * 6 + (j + 3)];
            Pvv[i * 3 + j] = P[(i + 3) * 6 + (j + 3)];
        }
    }

    float APrr[9], APrv[9], tmp[9], term[9];
    float Prr_new[9], Prv_new[9], Pvv_new[9];

    // Prr' = A Prr A^T + dt(A Prv + (A Prv)^T) + dt^2 Pvv + Qrr.
    mat3_mul(A, Prr, APrr);
    mat3_mul_bt(APrr, A, Prr_new);
    mat3_mul(A, Prv, APrv);
    mat3_add_scaled(Prr_new, APrv, dt);
    mat3_add_scaled_transpose(Prr_new, APrv, dt);
    mat3_add_scaled(Prr_new, Pvv, dt2);

    // Prv' = A Prr C^T + A Prv A^T + dt Prv^T C^T + dt Pvv A^T + Qrv.
    mat3_mul_bt(APrr, C, Prv_new);
    mat3_mul_bt(APrv, A, term);
    mat3_add_scaled(Prv_new, term, 1.0f);
    mat3_mul(C, Prv, tmp);
    mat3_add_scaled_transpose(Prv_new, tmp, dt);
    mat3_mul_bt(Pvv, A, term);
    mat3_add_scaled(Prv_new, term, dt);

    // Pvv' = C Prr C^T + C Prv A^T + A Prv^T C^T + A Pvv A^T + Qvv.
    mat3_mul(C, Prr, tmp);
    mat3_mul_bt(tmp, C, Pvv_new);
    mat3_mul(C, Prv, tmp);
    mat3_mul_bt(tmp, A, term);
    mat3_add_scaled(Pvv_new, term, 1.0f);
    mat3_add_scaled_transpose(Pvv_new, term, 1.0f);
    mat3_mul(A, Pvv, tmp);
    mat3_mul_bt(tmp, A, term);
    mat3_add_scaled(Pvv_new, term, 1.0f);

    float sa2 = sigma_a * sigma_a;
    float q_rr = sa2 * dt2 * dt / 3.0f;
    float q_rv = sa2 * dt2 * 0.5f;
    float q_vv = sa2 * dt;
    for (int i = 0; i < 3; i++) {
        Prr_new[i * 3 + i] += q_rr;
        Prv_new[i * 3 + i] += q_rv;
        Pvv_new[i * 3 + i] += q_vv;
    }
    mat3_symmetrize(Prr_new);
    mat3_symmetrize(Pvv_new);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            P[i * 6 + j] = Prr_new[i * 3 + j];
            P[i * 6 + (j + 3)] = Prv_new[i * 3 + j];
            P[(j + 3) * 6 + i] = Prv_new[i * 3 + j];
            P[(i + 3) * 6 + (j + 3)] = Pvv_new[i * 3 + j];
        }
    }
}

static inline void ekf_sync_rso_uncertainty(RSO* r, const OrbitalEyesCooperative* env) {
    float sig_r, sig_t, sig_n, sig_v;
    ekf_rtn_sigmas(r->P, r->pos, r->vel, &sig_r, &sig_t, &sig_n, &sig_v);

    int rebuild_P = 0;
    if (!isfinite(sig_r)) { sig_r = env->u_min_r; rebuild_P = 1; }
    if (!isfinite(sig_t)) { sig_t = env->u_min_t; rebuild_P = 1; }
    if (!isfinite(sig_n)) { sig_n = env->u_min_n; rebuild_P = 1; }
    if (!isfinite(sig_v)) { sig_v = env->vel_u_min; rebuild_P = 1; }

    if (env->ekf_sigma_max > 0.0f) {
        float smax = env->ekf_sigma_max;
        float svmax = fmaxf(env->vel_u_min, smax * 1e-3f);
        if (sig_r > smax) { sig_r = smax; rebuild_P = 1; }
        if (sig_t > smax) { sig_t = smax; rebuild_P = 1; }
        if (sig_n > smax) { sig_n = smax; rebuild_P = 1; }
        if (sig_v > svmax) { sig_v = svmax; rebuild_P = 1; }
    }

    r->u_r = fmaxf(env->u_min_r, sig_r);
    r->u_t = fmaxf(env->u_min_t, sig_t);
    r->u_n = fmaxf(env->u_min_n, sig_n);
    r->sigma_vel = fmaxf(env->vel_u_min, sig_v);
    r->u_anchor_sum = r->u_r + r->u_t + r->u_n;

    // Keep the internal covariance consistent with the capped/logged state.
    if (rebuild_P) {
        ekf_init_P(r->P, r->pos, r->vel,
                   r->u_r, r->u_t, r->u_n, r->sigma_vel);
    }
}

static int calc_obs_size(int num_agents, int rso_top_k) {
    return SELF_OBS_SIZE + OTHER_AGENT_SIZE * (num_agents - 1)
           + RSO_TOKEN_SIZE * rso_top_k;
}

// ─── Init (allocate sub-arrays) ───
void init(OrbitalEyesCooperative* env) {
    env->satellites = (Satellite*)calloc(env->num_agents, sizeof(Satellite));
    env->rsos = (RSO*)calloc(env->num_rso, sizeof(RSO));

    // Per-agent per-RSO tracking: steps since this agent last observed each RSO
    env->steps_since_observed = (int*)calloc(
        env->num_agents * env->num_rso, sizeof(int));

    // Top-K index mapping
    env->top_k_indices = (int*)calloc(
        env->num_agents * env->rso_top_k, sizeof(int));

    // Sort scratch buffers
    env->priority_sort_indices = (int*)calloc(env->num_rso, sizeof(int));
    env->priority_sort_values = (float*)calloc(env->num_rso, sizeof(float));

    // Top-K output buffers
    env->topk_out_indices = (int*)calloc(env->rso_top_k, sizeof(int));
    env->topk_out_values  = (float*)calloc(env->rso_top_k, sizeof(float));

    // Agent-independent base priority + reachability cache
    env->base_priority   = (float*)calloc(env->num_rso, sizeof(float));
    env->reachable_flags = (int*)calloc(env->num_rso, sizeof(int));

    // Reduction and modality-specific measurement accumulators.
    env->reduction_buffer = (float*)calloc(env->num_rso * 4, sizeof(float));
    env->measurement_quality = (float*)calloc(
        env->num_agents * env->num_rso, sizeof(float));

    // External-target override buffer (benchmark reachable-set access; off by default)
    env->external_targets = (int*)calloc(env->num_agents, sizeof(int));
    env->external_target_active = 0;

    // Compute observation size
    env->obs_size = calc_obs_size(env->num_agents, env->rso_top_k);

    // Observation layout offsets
    env->self_obs_offset = 0;
    env->agents_obs_offset = SELF_OBS_SIZE;
    env->rso_obs_offset = SELF_OBS_SIZE + OTHER_AGENT_SIZE * (env->num_agents - 1);

    // Materialize per-sensor geometry and capabilities. The default Phase-III
    // config sets these equal across types so access, agility, and the base
    // measurement model cannot explain learned specialization.
    for (int i = 0; i < env->num_agents; i++) {
        Satellite* s = &env->satellites[i];
        SensorConfig* cfg = &env->sensor_configs[i];
        s->sensor_type = clampi(cfg->sensor_type, SENSOR_OPTICAL, SENSOR_RADAR);
        s->ground_lat_deg = cfg->lat_deg;
        s->ground_lon_deg = cfg->lon_deg;
        s->ground_alt_km = cfg->alt_km;
        if (s->sensor_type == SENSOR_RADAR) {
            s->fov = env->radar_fov_deg * DEG2RAD;
            s->max_slew_rate = env->radar_max_slew_rate_deg * DEG2RAD;
            s->max_obs_range = env->radar_max_obs_range;
            s->meas_sigma_los = env->radar_sigma_los;
            s->meas_sigma_cross = env->radar_sigma_cross;
            s->meas_sigma_rate = env->radar_sigma_rate;
        } else {
            s->fov = env->optical_fov_deg * DEG2RAD;
            s->max_slew_rate = env->optical_max_slew_rate_deg * DEG2RAD;
            s->max_obs_range = env->optical_max_obs_range;
            s->meas_sigma_los = env->optical_sigma_los;
            s->meas_sigma_cross = env->optical_sigma_cross;
            s->meas_sigma_rate = 0.0f;
        }
    }

    // Initialise steps_since_observed to -1 (never observed)
    for (int i = 0; i < env->num_agents * env->num_rso; i++) {
        env->steps_since_observed[i] = -1;
    }

    env->client = NULL;
}

// ─── Sensor slew: rotate sensor_dir toward target_dir, rate-limited ───
static void slew_sensor(float* dir, const float* target, float max_rate, float dt) {
    // Compute angle between current direction and target
    float dot = dir[0]*target[0] + dir[1]*target[1] + dir[2]*target[2];
    dot = clampf(dot, -1.0f, 1.0f);
    float angle = acosf(dot);

    if (angle < 1e-6f) return;  // Already pointing at target

    float max_angle = max_rate * dt;
    float t_interp = (angle <= max_angle) ? 1.0f : max_angle / angle;

    // Rotation axis
    float ax, ay, az;
    vec3_cross(dir[0], dir[1], dir[2], target[0], target[1], target[2],
               &ax, &ay, &az);
    float axis_len = vec3_len(ax, ay, az);

    if (axis_len < 1e-12f) {
        // Directions are (anti-)parallel — pick arbitrary perpendicular axis
        if (fabsf(dir[0]) < 0.9f) {
            ax = 0; ay = -dir[2]; az = dir[1];
        } else {
            ax = dir[2]; ay = 0; az = -dir[0];
        }
        axis_len = vec3_len(ax, ay, az);
    }
    ax /= axis_len; ay /= axis_len; az /= axis_len;

    // Rodrigues rotation formula: v' = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
    float theta = angle * t_interp;
    float cos_t = cosf(theta), sin_t = sinf(theta);
    float kdotv = ax * dir[0] + ay * dir[1] + az * dir[2];

    float kxv_x, kxv_y, kxv_z;
    vec3_cross(ax, ay, az, dir[0], dir[1], dir[2], &kxv_x, &kxv_y, &kxv_z);

    dir[0] = dir[0] * cos_t + kxv_x * sin_t + ax * kdotv * (1.0f - cos_t);
    dir[1] = dir[1] * cos_t + kxv_y * sin_t + ay * kdotv * (1.0f - cos_t);
    dir[2] = dir[2] * cos_t + kxv_z * sin_t + az * kdotv * (1.0f - cos_t);

    // Re-normalise
    vec3_normalize(&dir[0], &dir[1], &dir[2]);
}

// ─── Earth occlusion test ───
// Returns 1 if line of sight from observer to target is blocked by Earth sphere.
// Both endpoints (satellite and RSO) are always above the surface, so the
// segment can only be blocked when its closest approach to Earth centre falls
// strictly inside the segment AND below the surface. That lets us skip the
// square root taken on the previous formulation: we test the closest-approach
// parameter t* = -b/(2a) and the minimum value f(t*) = c - b^2/(4a) of the
// quadratic |obs + t d|^2 - R^2 directly. Mathematically identical to the
// two-root test for endpoints outside the sphere.
static int earth_occluded(const float* obs_pos, const float* tgt_pos) {
    float dx = tgt_pos[0] - obs_pos[0];
    float dy = tgt_pos[1] - obs_pos[1];
    float dz = tgt_pos[2] - obs_pos[2];

    float a = dx*dx + dy*dy + dz*dz;
    if (a < 1e-12f) return 0;  // coincident points

    float b = 2.0f * (obs_pos[0]*dx + obs_pos[1]*dy + obs_pos[2]*dz);

    // Closest-approach parameter along the segment. If it lies outside [0,1]
    // the nearest point is an endpoint, which is above the surface → no block.
    float inv_2a = 0.5f / a;
    float tstar = -b * inv_2a;
    if (tstar <= 0.0f || tstar >= 1.0f) return 0;

    float c = obs_pos[0]*obs_pos[0] + obs_pos[1]*obs_pos[1] + obs_pos[2]*obs_pos[2]
              - EARTH_RADIUS_KM * EARTH_RADIUS_KM;

    // Minimum of the quadratic at t*: f(t*) = c - b^2/(4a). Blocked iff < 0.
    float min_f = c - b * b * inv_2a * 0.5f;
    return min_f < 0.0f;
}

// ─── Observation quality ───
static float compute_obs_quality(
    OrbitalEyesCooperative* env, const Satellite* sat,
    const float* rso_pos, float rso_size)
{
    const float* sat_pos = sat->pos;
    const float* sat_dir = sat->sensor_dir;
    if (env->quality_mode == 0) return 1.0f;  // Simple binary

    float dx = rso_pos[0] - sat_pos[0];
    float dy = rso_pos[1] - sat_pos[1];
    float dz = rso_pos[2] - sat_pos[2];
    float dist = vec3_len(dx, dy, dz);
    if (dist < 1e-6f) return 1.0f;

    if (env->quality_mode == 2) {
        // Complementary smooth range responses. Type A is strongest nearby;
        // Type B is strongest far away. Both remain useful through a non-zero
        // floor and have identical response at the crossover. This response is
        // intentionally not exposed as an observation feature.
        float width = fmaxf(env->range_transition_km, 1.0f);
        float x = clampf((dist - env->range_crossover_km) / width,
                         -20.0f, 20.0f);
        float far_response = 1.0f / (1.0f + expf(-x));
        float preferred = sat->sensor_type == SENSOR_TYPE_B
                        ? far_response : 1.0f - far_response;
        float floor = clampf(env->range_quality_floor, 0.0f, 1.0f);
        return floor + (1.0f - floor) * preferred;
    }

    // Distance factor: linearly decreasing with distance
    float dist_factor = 1.0f - clampf(dist / sat->max_obs_range, 0.0f, 1.0f);

    // Angle factor: cosine of off-boresight angle
    float dot = (dx * sat_dir[0] + dy * sat_dir[1] + dz * sat_dir[2]) / dist;
    float angle_factor = clampf(dot, 0.0f, 1.0f);

    // Size factor: larger objects are easier to observe
    float size_factor = clampf(rso_size, 0.1f, 10.0f) / 10.0f;

    float quality = env->quality_dist_weight * dist_factor
                  + env->quality_angle_weight * angle_factor
                  + env->quality_size_weight * size_factor;

    float total_weight = env->quality_dist_weight + env->quality_angle_weight
                       + env->quality_size_weight;
    if (total_weight > 0.0f) quality /= total_weight;

    return clampf(quality, 0.0f, 1.0f);
}

// ─── Simple insertion sort by priority (descending) ───
static void sort_by_priority_desc(int* indices, float* values, int n) {
    for (int i = 1; i < n; i++) {
        float val = values[i];
        int idx = indices[i];
        int j = i - 1;
        while (j >= 0 && values[j] < val) {
            values[j + 1] = values[j];
            indices[j + 1] = indices[j];
            j--;
        }
        values[j + 1] = val;
        indices[j + 1] = idx;
    }
}

// ─── Min-heap helpers for top-K selection ───
static inline void _heap_sift_down(int* indices, float* values, int n, int i) {
    while (1) {
        int smallest = i;
        int l = 2 * i + 1;
        int r = 2 * i + 2;
        if (l < n && values[l] < values[smallest]) smallest = l;
        if (r < n && values[r] < values[smallest]) smallest = r;
        if (smallest == i) break;
        // swap
        float tv = values[i]; values[i] = values[smallest]; values[smallest] = tv;
        int   ti = indices[i]; indices[i] = indices[smallest]; indices[smallest] = ti;
        i = smallest;
    }
}

// Select the top-K elements by largest value using a min-heap of size K.
// Result is written to out_indices/out_values, sorted descending.
// all_values is READ-ONLY (not modified).
static void top_k_desc(const int* all_indices, const float* all_values, int n, int k,
                        int* out_indices, float* out_values) {
    if (k <= 0) return;
    if (k >= n) {
        // Need all elements – copy and sort in place
        memcpy(out_indices, all_indices, n * sizeof(int));
        memcpy(out_values, all_values, n * sizeof(float));
        sort_by_priority_desc(out_indices, out_values, n);
        return;
    }

    // Seed the min-heap with the first K elements
    memcpy(out_indices, all_indices, k * sizeof(int));
    memcpy(out_values, all_values, k * sizeof(float));

    // Build min-heap (smallest value at root)
    for (int i = k / 2 - 1; i >= 0; i--)
        _heap_sift_down(out_indices, out_values, k, i);

    // Scan remaining elements; replace root if larger
    for (int i = k; i < n; i++) {
        if (all_values[i] > out_values[0]) {
            out_values[0]  = all_values[i];
            out_indices[0] = all_indices[i];
            _heap_sift_down(out_indices, out_values, k, 0);
        }
    }

    // Sort the K winners descending (K is tiny – insertion sort)
    for (int i = 1; i < k; i++) {
        float val = out_values[i];
        int   idx = out_indices[i];
        int j = i - 1;
        while (j >= 0 && out_values[j] < val) {
            out_values[j + 1]  = out_values[j];
            out_indices[j + 1] = out_indices[j];
            j--;
        }
        out_values[j + 1]  = val;
        out_indices[j + 1] = idx;
    }
}

// ─── Compute population objective J ───
static float compute_population_objective(OrbitalEyesCooperative* env) {
    float sum_U = 0.0f;
    float max_U = 0.0f;
    int above_threshold = 0;

    for (int i = 0; i < env->num_rso; i++) {
        RSO* r = &env->rsos[i];
        float U = surrogate_u_sum_now(r, env);
        sum_U += U;
        if (U > max_U) max_U = U;
        if (U > env->uncertainty_threshold) above_threshold++;
    }

    float mean_U = sum_U / (float)env->num_rso;
    float frac_above = (float)above_threshold / (float)env->num_rso;

    return env->alpha_u * mean_U + env->alpha_v * frac_above;
}


// Risk-aware catalogue badness for reward_mode=2.
// Lower is better. The log transform keeps extreme outliers important without
// letting one RSO explode the reward scale.
static float compute_catalogue_badness(
    OrbitalEyesCooperative* env, float* out_tail_badness, float* out_frac_above) {
    float target = env->uncertainty_threshold;
    if (target <= 1e-6f) target = env->U_ref;
    if (target <= 1e-6f) target = 1.0f;

    float sum_badness = 0.0f;
    int above_threshold = 0;
    for (int i = 0; i < env->num_rso; i++) {
        float U = surrogate_u_sum_now(&env->rsos[i], env);
        float x = fmaxf(U / target, 0.0f);
        float b = log1pf(x);
        env->priority_sort_values[i] = b;
        sum_badness += b;
        if (x > 1.0f) above_threshold++;
    }

    // Sort descending for the worst-tail term. num_rso is small in these envs.
    for (int i = 1; i < env->num_rso; i++) {
        float key = env->priority_sort_values[i];
        int j = i - 1;
        while (j >= 0 && env->priority_sort_values[j] < key) {
            env->priority_sort_values[j + 1] = env->priority_sort_values[j];
            j--;
        }
        env->priority_sort_values[j + 1] = key;
    }

    int tail_n = (int)ceilf(env->reward_tail_frac * (float)env->num_rso);
    if (tail_n < 1) tail_n = 1;
    if (tail_n > env->num_rso) tail_n = env->num_rso;

    float tail_sum = 0.0f;
    for (int i = 0; i < tail_n; i++) tail_sum += env->priority_sort_values[i];

    float mean_badness = sum_badness / (float)env->num_rso;
    float tail_badness = tail_sum / (float)tail_n;
    float frac_above = (float)above_threshold / (float)env->num_rso;

    if (out_tail_badness) *out_tail_badness = tail_badness;
    if (out_frac_above) *out_frac_above = frac_above;

    return env->reward_mean_weight * mean_badness
         + env->reward_tail_weight * tail_badness
         + env->reward_threshold_weight * frac_above;
}

// Tail-sensitive reward badness for reward_mode=4. This uses a tighter
// uncertainty scale, the worst tail, and the single worst RSO.
static float compute_tail_sensitive_badness(OrbitalEyesCooperative* env) {
    float target = env->reward_u_target;
    if (target <= 1e-6f) target = 1.0f;

    float sum_badness = 0.0f;
    float max_badness = 0.0f;
    int above_threshold = 0;
    for (int i = 0; i < env->num_rso; i++) {
        float U = surrogate_u_sum_now(&env->rsos[i], env);
        float b = log1pf(fmaxf(U / target, 0.0f));
        env->priority_sort_values[i] = b;
        sum_badness += b;
        if (b > max_badness) max_badness = b;
        if (U > target) above_threshold++;
    }

    for (int i = 1; i < env->num_rso; i++) {
        float key = env->priority_sort_values[i];
        int j = i - 1;
        while (j >= 0 && env->priority_sort_values[j] < key) {
            env->priority_sort_values[j + 1] = env->priority_sort_values[j];
            j--;
        }
        env->priority_sort_values[j + 1] = key;
    }

    int tail_n = (int)ceilf(env->reward_tail_frac * (float)env->num_rso);
    if (tail_n < 1) tail_n = 1;
    if (tail_n > env->num_rso) tail_n = env->num_rso;

    float tail_sum = 0.0f;
    for (int i = 0; i < tail_n; i++) {
        tail_sum += env->priority_sort_values[i];
    }

    float mean_badness = sum_badness / (float)env->num_rso;
    float tail_badness = tail_sum / (float)tail_n;
    float frac_above = (float)above_threshold / (float)env->num_rso;

    return env->reward_mean_weight * mean_badness
         + env->reward_tail_weight * tail_badness
         + env->reward_max_weight * max_badness
         + env->reward_threshold_weight * frac_above;
}

// Precision-target badness for reward_mode=5. This keeps the reward focused
// on the fine-performance regime instead of the broad custody threshold.
static float compute_precision_badness(OrbitalEyesCooperative* env) {
    float sum_U = 0.0f;
    float max_U = 0.0f;

    for (int i = 0; i < env->num_rso; i++) {
        float U = fmaxf(surrogate_u_sum_now(&env->rsos[i], env), 0.0f);
        sum_U += U;
        if (U > max_U) max_U = U;
        env->priority_sort_values[i] = U;
    }

    // Sort descending so the top tail is a stable p90-style population term.
    for (int i = 1; i < env->num_rso; i++) {
        float key = env->priority_sort_values[i];
        int j = i - 1;
        while (j >= 0 && env->priority_sort_values[j] < key) {
            env->priority_sort_values[j + 1] = env->priority_sort_values[j];
            j--;
        }
        env->priority_sort_values[j + 1] = key;
    }

    int tail_n = (int)ceilf(env->reward_tail_frac * (float)env->num_rso);
    if (tail_n < 1) tail_n = 1;
    if (tail_n > env->num_rso) tail_n = env->num_rso;

    float tail_sum = 0.0f;
    for (int i = 0; i < tail_n; i++) {
        tail_sum += env->priority_sort_values[i];
    }

    float mean_U = sum_U / (float)env->num_rso;
    float tail_U = tail_sum / (float)tail_n;
    env->precision_mean_uncertainty = mean_U;
    env->precision_max_uncertainty = max_U;
    float mean_target = fmaxf(env->reward_mean_target, 1e-6f);
    float tail_target = fmaxf(env->reward_tail_target, 1e-6f);
    float max_target = fmaxf(env->reward_max_target, 1e-6f);

    float max_ratio = max_U / max_target;
    // Preserve the original log reward and add only a gentle excess penalty
    // above the target, so one neglected RSO gets extra attention without
    // changing the reward landscape in the normal operating range.
    float max_badness = log1pf(max_ratio)
                      + 0.25f * fmaxf(max_ratio - 1.0f, 0.0f);

    // Add a bounded, smooth signal around the difficult fine regime. This
    // term is saturated for very large max-U values, then becomes informative
    // as the worst RSO crosses the configured fine target and improves below
    // it. The existing log/max terms still provide the coarse-to-fine signal.
    float fine_target = fmaxf(env->reward_fine_max_target, 1e-6f);
    float fine_temperature = fmaxf(env->reward_fine_max_temperature, 1e-3f);
    float fine_x = clampf((max_U - fine_target) / fine_temperature, -20.0f, 20.0f);
    float fine_badness = 1.0f / (1.0f + expf(-fine_x));

    return env->reward_mean_weight * log1pf(mean_U / mean_target)
         + env->reward_tail_weight * log1pf(tail_U / tail_target)
         + env->reward_max_weight * max_badness
         + env->reward_fine_max_weight * fine_badness;
}

static float precision_goal_badness(OrbitalEyesCooperative* env) {
    // At each configured target, every log1p(U / target) term equals log(2).
    // The sigmoid fine-regime term is centered at 0.5 when max-U equals its
    // transition target, so the level reward is neutral at that boundary.
    float weight_sum = env->reward_mean_weight
                     + env->reward_tail_weight
                     + env->reward_max_weight;
    return fmaxf(weight_sum, 1e-6f) * logf(2.0f)
         + env->reward_fine_max_weight * 0.5f;
}

// Bounded staged score for the difficult max-U margin. It starts at a
// configured onset, then grows more strongly below the fine threshold.
static float compute_fine_max_score(OrbitalEyesCooperative* env) {
    float mid = fmaxf(env->reward_fine_max_target,
                      env->reward_max_target + 1e-3f);
    float start = fmaxf(env->reward_fine_max_start, mid + 1e-3f);
    float final_target = fmaxf(env->reward_max_target, 1e-6f);

    float max_U = env->precision_max_uncertainty;

    float x_coarse = clampf((start - max_U) / (start - mid), 0.0f, 1.0f);
    float x_fine = clampf((mid - max_U) / (mid - final_target), 0.0f, 1.0f);
    // The square-root ramp gives a useful signal immediately below the
    // plateau onset, while the smoothstep keeps the final region bounded.
    float coarse = sqrtf(x_coarse);
    float fine = x_fine * x_fine * (3.0f - 2.0f * x_fine);

    return 0.35f * coarse + 0.65f * fine;
}

static float compute_fine_mean_score(OrbitalEyesCooperative* env) {
    float target = fmaxf(env->reward_mean_target, 1e-6f);
    float start = fmaxf(env->reward_fine_mean_start, target + 1e-3f);
    float mean_U = env->precision_mean_uncertainty;
    float x = clampf((start - mean_U) / (start - target), 0.0f, 1.0f);
    return sqrtf(x);
}

// Forward declaration
static void compute_observations(OrbitalEyesCooperative* env);
static void set_was_my_target(OrbitalEyesCooperative* env);

// ─── Reset ───

static void set_ground_sensor_state(OrbitalEyesCooperative* env, Satellite* s, int reset_pointing) {
    (void)env;
    float lat = s->ground_lat_deg * DEG2RAD;
    float lon = s->ground_lon_deg * DEG2RAD;
    float radius = EARTH_RADIUS_KM + s->ground_alt_km;
    float cos_lat = cosf(lat);
    float ux = cos_lat * cosf(lon);
    float uy = cos_lat * sinf(lon);
    float uz = sinf(lat);

    s->pos[0] = radius * ux;
    s->pos[1] = radius * uy;
    s->pos[2] = radius * uz;
    s->vel[0] = 0.0f;
    s->vel[1] = 0.0f;
    s->vel[2] = 0.0f;

    if (reset_pointing) {
        s->sensor_dir[0] = ux;
        s->sensor_dir[1] = uy;
        s->sensor_dir[2] = uz;
        vec3_normalize(&s->sensor_dir[0], &s->sensor_dir[1], &s->sensor_dir[2]);
    }
}

// Sample an orbital plane that intersects a random point in the configured
// ground cap with a uniformly random local flight heading. This preserves
// array reachability without forcing the catalogue into a narrow RAAN band.
static void sample_common_dome_orbit_plane(
    OrbitalEyesCooperative* env, float* inclination, float* raan
) {
    float lat0 = env->rso_dome_center_lat_deg * DEG2RAD;
    float lon0 = env->rso_dome_center_lon_deg * DEG2RAD;
    float radius = fmaxf(env->rso_dome_radius_deg, 0.0f) * DEG2RAD;

    // Uniform area sampling on the spherical cap, followed by the direct
    // geodesic solution from its centre.
    float u = randf(0.0f, 1.0f);
    float cos_delta = 1.0f - u * (1.0f - cosf(radius));
    float delta = acosf(clampf(cos_delta, -1.0f, 1.0f));
    float bearing = randf(0.0f, 2.0f * PI);
    float sin_lat0 = sinf(lat0);
    float cos_lat0 = cosf(lat0);
    float sin_delta = sinf(delta);
    float cos_delta_exact = cosf(delta);
    float sin_lat = sin_lat0 * cos_delta_exact
        + cos_lat0 * sin_delta * cosf(bearing);
    sin_lat = clampf(sin_lat, -1.0f, 1.0f);
    float lat = asinf(sin_lat);
    float lon = lon0 + atan2f(
        sinf(bearing) * sin_delta * cos_lat0,
        cos_delta_exact - sin_lat0 * sin_lat);

    float sin_lat_i = sinf(lat);
    float cos_lat_i = cosf(lat);
    float sin_lon_i = sinf(lon);
    float cos_lon_i = cosf(lon);

    // Radial vector at the sampled crossing point and local tangent basis.
    float px = cos_lat_i * cos_lon_i;
    float py = cos_lat_i * sin_lon_i;
    float pz = sin_lat_i;
    float ex = -sin_lon_i;
    float ey = cos_lon_i;
    float ez = 0.0f;
    float nx = -sin_lat_i * cos_lon_i;
    float ny = -sin_lat_i * sin_lon_i;
    float nz = cos_lat_i;

    float heading = randf(0.0f, 2.0f * PI);
    float qx = cosf(heading) * nx + sinf(heading) * ex;
    float qy = cosf(heading) * ny + sinf(heading) * ey;
    float qz = cosf(heading) * nz + sinf(heading) * ez;

    // h = p x q defines the orbit-plane normal. The standard element
    // convention gives h = [sin(i)sin(Omega), -sin(i)cos(Omega), cos(i)].
    float hx = py * qz - pz * qy;
    float hy = pz * qx - px * qz;
    float hz = px * qy - py * qx;
    float hnorm = sqrtf(hx * hx + hy * hy + hz * hz);
    hx /= hnorm;
    hy /= hnorm;
    hz /= hnorm;

    *inclination = acosf(clampf(hz, -1.0f, 1.0f));
    *raan = atan2f(hx, -hy);
    if (*raan < 0.0f) {
        *raan += 2.0f * PI;
    }
}

void c_reset(OrbitalEyesCooperative* env) {
    env->tick = 0;
    env->epoch_time = 0.0f;

    // Initialise satellite orbital elements from multi-orbit configuration
    int sat_idx = 0;
    for (int o = 0; o < env->num_orbits; o++) {
        OrbitConfig* oc = &env->orbits[o];
        float inc_rad = oc->inc * DEG2RAD;
        float raan_rad = oc->raan * DEG2RAD;
        float omega_rad = oc->omega * DEG2RAD;

        for (int j = 0; j < oc->num_sats && sat_idx < env->num_agents; j++) {
            Satellite* s = &env->satellites[sat_idx];
            s->oe_a = oc->a;
            s->oe_e = oc->e;
            s->oe_inc = inc_rad;
            s->oe_raan = raan_rad;
            s->oe_omega = omega_rad;
            s->orbit_id = o;
            // Equally space satellites within this orbit in mean anomaly
            s->oe_M0 = 2.0f * PI * (float)j / (float)oc->num_sats;

            // Precompute orbital invariants
            precompute_orbital_cache(
                s->oe_a, s->oe_e, s->oe_inc, env->propagation_mode,
                &s->oe_n, &s->oe_sqrt_mu_a, &s->oe_sqrt_1_e2,
                &s->oe_cos_inc, &s->oe_sin_inc,
                &s->oe_raan_dot, &s->oe_omega_dot);

            // Cache the constant perifocal→ECI rotation columns (used by the
            // fast per-step propagator when there is no J2 secular drift).
            precompute_rotation_columns(
                s->oe_raan, s->oe_omega, s->oe_cos_inc, s->oe_sin_inc,
                &s->oe_Px, &s->oe_Py, &s->oe_Pz,
                &s->oe_Qx, &s->oe_Qy, &s->oe_Qz);

            // Compute initial position
            propagate_cached(
                s->oe_a, s->oe_e, s->oe_raan, s->oe_omega, s->oe_M0,
                0.0f,
                s->oe_n, s->oe_sqrt_mu_a, s->oe_sqrt_1_e2,
                s->oe_cos_inc, s->oe_sin_inc,
                s->oe_raan_dot, s->oe_omega_dot,
                &s->pos[0], &s->pos[1], &s->pos[2],
                &s->vel[0], &s->vel[1], &s->vel[2]);

            // Initialise sensor pointing: nadir for orbiting sensors, zenith for ground sensors.
            s->sensor_dir[0] = -s->pos[0];
            s->sensor_dir[1] = -s->pos[1];
            s->sensor_dir[2] = -s->pos[2];
            vec3_normalize(&s->sensor_dir[0], &s->sensor_dir[1], &s->sensor_dir[2]);
            if (env->ground_sensor) {
                set_ground_sensor_state(env, s, 1);
            }

            s->last_action = -1;
            s->last_action_idx = env->rso_top_k;
            s->last_delta[0] = 0.0f;
            s->last_delta[1] = 0.0f;
            s->observed_any = 0;
            s->prev_sensor_dir[0] = s->sensor_dir[0];
            s->prev_sensor_dir[1] = s->sensor_dir[1];
            s->prev_sensor_dir[2] = s->sensor_dir[2];

            sat_idx++;
        }
    }

    // Initialise RSO population with randomised orbital elements
    for (int i = 0; i < env->num_rso; i++) {
        RSO* r = &env->rsos[i];
        r->oe_a = randf(env->rso_a_min, env->rso_a_max);
        r->oe_e = randf(env->rso_e_min, env->rso_e_max);
        if (env->rso_common_dome_mode) {
            sample_common_dome_orbit_plane(env, &r->oe_inc, &r->oe_raan);
        } else {
            r->oe_inc = randf(env->rso_inc_min * DEG2RAD, env->rso_inc_max * DEG2RAD);
            r->oe_raan = randf(env->rso_raan_min * DEG2RAD, env->rso_raan_max * DEG2RAD);
        }
        r->oe_omega = randf(env->rso_omega_min * DEG2RAD, env->rso_omega_max * DEG2RAD);
        r->oe_M0 = randf(0.0f, 2.0f * PI);

        // Precompute orbital invariants
        precompute_orbital_cache(
            r->oe_a, r->oe_e, r->oe_inc, env->propagation_mode,
            &r->oe_n, &r->oe_sqrt_mu_a, &r->oe_sqrt_1_e2,
            &r->oe_cos_inc, &r->oe_sin_inc,
            &r->oe_raan_dot, &r->oe_omega_dot);

        // Cache the constant perifocal→ECI rotation columns.
        precompute_rotation_columns(
            r->oe_raan, r->oe_omega, r->oe_cos_inc, r->oe_sin_inc,
            &r->oe_Px, &r->oe_Py, &r->oe_Pz,
            &r->oe_Qx, &r->oe_Qy, &r->oe_Qz);

        // Compute initial position
        propagate_cached(
            r->oe_a, r->oe_e, r->oe_raan, r->oe_omega, r->oe_M0,
            0.0f,
            r->oe_n, r->oe_sqrt_mu_a, r->oe_sqrt_1_e2,
            r->oe_cos_inc, r->oe_sin_inc,
            r->oe_raan_dot, r->oe_omega_dot,
            &r->pos[0], &r->pos[1], &r->pos[2],
            &r->vel[0], &r->vel[1], &r->vel[2]);

        // Initial uncertainty state — sampled uniformly within [min, max].
        // If max <= min the value is fixed (backwards-compatible).
        float ur_max  = fmaxf(env->u_init_r,        env->u_init_r_max);
        float ut_max  = fmaxf(env->u_init_t,        env->u_init_t_max);
        float un_max  = fmaxf(env->u_init_n,        env->u_init_n_max);
        float sv_max  = fmaxf(env->sigma_vel_init,  env->sigma_vel_init_max);
        float age_max = fmaxf(env->age_init,        env->age_init_max);
        float init_r   = randf(env->u_init_r,       ur_max);
        float init_t   = randf(env->u_init_t,       ut_max);
        float init_n   = randf(env->u_init_n,       un_max);
        float init_sv  = randf(env->sigma_vel_init, sv_max);
        float init_age = randf(env->age_init,       age_max);
        r->u_r = init_r;
        r->u_t = init_t;
        r->u_n = init_n;
        r->sigma_vel = init_sv;
        r->age = init_age;

        if (env->uncertainty_mode == 2) {
            float age = init_age;
            r->u_r = fmaxf(env->u_min_r,
                init_r + (env->growth_a_r + 0.5f * env->growth_b_r * age) * age);
            r->u_t = fmaxf(env->u_min_t,
                init_t + (env->growth_a_t + 0.5f * env->growth_b_t * age) * age);
            r->u_n = fmaxf(env->u_min_n,
                init_n + (env->growth_a_n + 0.5f * env->growth_b_n * age) * age);
            if (env->growth_mode >= 1) {
                r->sigma_vel = fmaxf(env->vel_u_min, init_sv + env->k_v * age);
            }
        }
        r->u_anchor_sum = r->u_r + r->u_t + r->u_n;

        // Initialize EKF covariance. Mode 1 uses it only at observation events;
        // mode 2 propagates it every step for the whole catalogue.
        ekf_init_P(r->P, r->pos, r->vel,
                   r->u_r, r->u_t, r->u_n, r->sigma_vel);
        r->ekf_anchor_fresh = (env->uncertainty_mode == 2) ? 1 : 0;

        // Random object size
        r->size = randf(env->rso_size_min, env->rso_size_max);
        r->priority = 0.0f;
        r->num_agents_observing = 0;
    }

    // Reset per-agent tracking
    for (int i = 0; i < env->num_agents * env->num_rso; i++) {
        env->steps_since_observed[i] = -1;
    }

    // Zero out top-K indices
    memset(env->top_k_indices, 0,
           env->num_agents * env->rso_top_k * sizeof(int));

    // Zero rewards, terminals, and episode-local accumulators. The Log struct
    // is a report buffer and may be cleared by vec_log between episode ends.
    memset(env->rewards, 0, env->num_agents * sizeof(float));
    memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));
    env->episode_return_accum = 0.0f;
    env->total_observations_accum = 0.0f;
    env->duplicate_assignment_accum = 0.0f;
    env->same_modality_duplicate_accum = 0.0f;
    env->complementary_fusion_accum = 0.0f;
    env->optical_observations_accum = 0.0f;
    env->radar_observations_accum = 0.0f;
    env->local_information_gain_accum = 0.0f;
    env->optical_information_gain_accum = 0.0f;
    env->radar_information_gain_accum = 0.0f;
    env->type_a_observation_range_accum = 0.0f;
    env->type_b_observation_range_accum = 0.0f;
    env->type_a_preferred_observations_accum = 0.0f;
    env->type_b_preferred_observations_accum = 0.0f;
    env->crossover_observations_accum = 0.0f;
    memset(env->local_info_gain, 0, sizeof(env->local_info_gain));

    // Auto-compute reward_baseline if set to 0:
    // Subtract constant growth cost so the agent isn't punished for what it can't control.
    // This centres reward at 0 when the agent does nothing.
    if (env->reward_baseline <= 0.0f) {
        env->reward_baseline = env->alpha_u
            * (env->growth_a_r + env->growth_a_t + env->growth_a_n) * env->dt;
    }

    // Auto-compute reward_scale if set to 0:
    // Normalise by the max single-step observation benefit so that
    // "all agents observe perfectly" → reward ≈ +1.
    if (env->reward_scale <= 0.0f) {
        float obs_reduction = env->alpha_r + env->alpha_t + env->alpha_n;
        env->reward_scale = env->alpha_u * (float)env->num_agents * obs_reduction
                          / (float)env->num_rso;
        if (env->reward_scale < 1e-8f) env->reward_scale = 1.0f;  // safety
    }

    // Compute initial J_prev
    env->J_prev = compute_population_objective(env);

    // Reference uncertainty for level-based reward (reward_mode=1).
    // Computed from the ACTUAL mean initial uncertainty across all RSOs after
    // randomisation, so r = 1 - mean_U/U_ref starts at exactly 0 each episode
    // regardless of the initial distribution shape.
    {
        float sum_U_init = 0.0f;
        for (int i = 0; i < env->num_rso; i++) {
            sum_U_init += surrogate_u_sum_now(&env->rsos[i], env);
        }
        if (env->reward_u_ref > 0.0f) {
            env->U_ref = env->reward_u_ref;
        } else {
            env->U_ref = sum_U_init / (float)env->num_rso;
        }
    }
    if (env->U_ref < 1e-6f) env->U_ref = 1.0f;  // safety
    {
        float sum_U0 = 0.0f;
        float max_U0 = 0.0f;
        int above_threshold0 = 0;
        for (int i = 0; i < env->num_rso; i++) {
            float U0 = surrogate_u_sum_now(&env->rsos[i], env);
            sum_U0 += U0;
            if (U0 > max_U0) max_U0 = U0;
            if (U0 > env->uncertainty_threshold) above_threshold0++;
        }
        env->initial_mean_uncertainty = sum_U0 / (float)env->num_rso;
        env->initial_max_uncertainty = max_U0;
        env->initial_fraction_above_threshold = (float)above_threshold0 / (float)env->num_rso;
    }
    if (env->reward_mode == 4) {
        env->badness_initial = compute_tail_sensitive_badness(env);
    } else if (env->reward_mode == 5) {
        env->badness_initial = compute_precision_badness(env);
    } else {
        env->badness_initial = compute_catalogue_badness(env, NULL, NULL);
    }
    env->badness_prev = env->badness_initial;
    if (env->badness_initial < 1e-6f) env->badness_initial = 1.0f;

    // Build initial observations
    compute_observations(env);
    set_was_my_target(env);
}

// ─── Build observations for all agents ───
static void compute_observations(OrbitalEyesCooperative* env) {
    // ── Pre-compute agent-independent base priority (once per step) ──
    float catalogue_u_sum = 0.0f;
    float catalogue_u_max = 0.0f;
    int catalogue_above_threshold = 0;
    for (int r = 0; r < env->num_rso; r++) {
        RSO* rso = &env->rsos[r];
        float U = surrogate_u_sum_now(rso, env);
        catalogue_u_sum += U;
        catalogue_u_max = fmaxf(catalogue_u_max, U);
        catalogue_above_threshold += U > env->uncertainty_threshold;
        env->base_priority[r] = env->priority_alpha * U
                               + env->priority_beta * rso->age;
        env->priority_sort_indices[r] = r;  // identity permutation
    }
    float catalogue_u_mean = catalogue_u_sum / fmaxf((float)env->num_rso, 1.0f);
    float catalogue_above_fraction =
        (float)catalogue_above_threshold / fmaxf((float)env->num_rso, 1.0f);

    for (int a = 0; a < env->num_agents; a++) {
        Satellite* sat = &env->satellites[a];
        float fov_cos = cosf(sat->fov);
        float* obs = env->observations + a * env->obs_size;

        // Zero all observations
        memset(obs, 0, env->obs_size * sizeof(float));

        // Ego-local east/north/up basis. Relational position, velocity, and
        // pointing features are expressed in this frame so the learned rule is
        // not tied to one absolute longitude.
        float up[3] = {sat->pos[0], sat->pos[1], sat->pos[2]};
        vec3_normalize(&up[0], &up[1], &up[2]);
        float east[3];
        vec3_cross(0.0f, 0.0f, 1.0f, up[0], up[1], up[2],
                   &east[0], &east[1], &east[2]);
        if (vec3_len(east[0], east[1], east[2]) < 1e-6f) {
            vec3_cross(1.0f, 0.0f, 0.0f, up[0], up[1], up[2],
                       &east[0], &east[1], &east[2]);
        }
        vec3_normalize(&east[0], &east[1], &east[2]);
        float north[3];
        vec3_cross(up[0], up[1], up[2], east[0], east[1], east[2],
                   &north[0], &north[1], &north[2]);
        vec3_normalize(&north[0], &north[1], &north[2]);

        // ── Self state (18) ──
        // pos(3), vel(3), local sensor_dir(3), previous action/deltas(2),
        // fov(1), modality id(1), range(1), slew rate(1), and bounded
        // catalogue mean/max/threshold-fraction summaries (3).
        int idx = 0;
        obs[idx++] = tanhf(sat->pos[0] / env->obs_dist_norm);
        obs[idx++] = tanhf(sat->pos[1] / env->obs_dist_norm);
        obs[idx++] = tanhf(sat->pos[2] / env->obs_dist_norm);
        obs[idx++] = tanhf(sat->vel[0] / env->obs_vel_norm);
        obs[idx++] = tanhf(sat->vel[1] / env->obs_vel_norm);
        obs[idx++] = tanhf(sat->vel[2] / env->obs_vel_norm);
        obs[idx++] = vec3_dot(sat->sensor_dir[0], sat->sensor_dir[1],
                              sat->sensor_dir[2], east[0], east[1], east[2]);
        obs[idx++] = vec3_dot(sat->sensor_dir[0], sat->sensor_dir[1],
                              sat->sensor_dir[2], north[0], north[1], north[2]);
        obs[idx++] = vec3_dot(sat->sensor_dir[0], sat->sensor_dir[1],
                              sat->sensor_dir[2], up[0], up[1], up[2]);
        // Previous action (normalised)
        if (env->action_mode == 0) {
            obs[idx++] = (float)sat->last_action_idx / (float)(env->rso_top_k > 0 ? env->rso_top_k : 1);
        } else {
            obs[idx++] = sat->last_delta[0];  // already in [-1,1]
        }
        obs[idx++] = (env->action_mode == 1) ? sat->last_delta[1] : 0.0f;
        obs[idx++] = sat->fov / PI;  // normalised
        obs[idx++] = (float)sat->sensor_type;
        obs[idx++] = clampf(sat->max_obs_range / env->obs_dist_norm, 0.0f, 1.0f);
        obs[idx++] = clampf(sat->max_slew_rate / DEG2RAD, 0.0f, 1.0f);
        obs[idx++] = tanhf(catalogue_u_mean / env->obs_u_norm);
        obs[idx++] = clampf(
            catalogue_u_max / (10.0f * env->obs_u_norm), 0.0f, 1.0f);
        obs[idx++] = catalogue_above_fraction;

        // ── Other agents (num_agents-1 × 11) ──
        // rel_pos(3), sensor_dir(3), is_observing(1), modality(1),
        // fov(1), range(1), slew rate(1).
        int agent_idx = 0;
        for (int j = 0; j < env->num_agents; j++) {
            if (j == a) continue;
            Satellite* other = &env->satellites[j];
            int base = env->agents_obs_offset + agent_idx * OTHER_AGENT_SIZE;
            float rel_x = other->pos[0] - sat->pos[0];
            float rel_y = other->pos[1] - sat->pos[1];
            float rel_z = other->pos[2] - sat->pos[2];
            obs[base + 0] = tanhf(vec3_dot(rel_x, rel_y, rel_z,
                                           east[0], east[1], east[2])
                                  / env->obs_dist_norm);
            obs[base + 1] = tanhf(vec3_dot(rel_x, rel_y, rel_z,
                                           north[0], north[1], north[2])
                                  / env->obs_dist_norm);
            obs[base + 2] = tanhf(vec3_dot(rel_x, rel_y, rel_z,
                                           up[0], up[1], up[2])
                                  / env->obs_dist_norm);
            obs[base + 3] = vec3_dot(other->sensor_dir[0], other->sensor_dir[1],
                                     other->sensor_dir[2],
                                     east[0], east[1], east[2]);
            obs[base + 4] = vec3_dot(other->sensor_dir[0], other->sensor_dir[1],
                                     other->sensor_dir[2],
                                     north[0], north[1], north[2]);
            obs[base + 5] = vec3_dot(other->sensor_dir[0], other->sensor_dir[1],
                                     other->sensor_dir[2],
                                     up[0], up[1], up[2]);
            // is_observing: 1 if the other agent observed at least one RSO last step
            obs[base + 6] = (float)other->observed_any;
            obs[base + 7] = (float)other->sensor_type;
            obs[base + 8] = other->fov / PI;
            obs[base + 9] = clampf(
                other->max_obs_range / env->obs_dist_norm, 0.0f, 1.0f);
            obs[base + 10] = clampf(
                other->max_slew_rate / DEG2RAD, 0.0f, 1.0f);
            agent_idx++;
        }

        // ── RSO tokens (top_k x 19) ──
        // Compute per-agent priority: base_priority + reachability offset.
        // Reachable RSOs keep positive priority; unreachable get pushed
        // below all reachable ones (negative offset).
        //
        // This is the dominant per-step cost, so the reachability test is
        // inlined and written branchlessly to let the compiler vectorise the
        // scan over the full catalogue. The Earth-occlusion quadratic
        // |sat + t d|^2 - R^2 has an agent-constant term c = |sat|^2 - R^2 that
        // is hoisted out of the loop; the segment is blocked iff its closest
        // approach lies inside (0,1) and dips below the surface (min_f < 0).
        {
            float sx = sat->pos[0], sy = sat->pos[1], sz = sat->pos[2];
            float sat_c = sx * sx + sy * sy + sz * sz
                        - EARTH_RADIUS_KM * EARTH_RADIUS_KM;
            int use_earth = env->enable_earth_mask;
            int use_range = env->enable_range_mask;
            float max_r2 = sat->max_obs_range * sat->max_obs_range;
            const RSO* rsos = env->rsos;
            const float* bp = env->base_priority;
            int* rf = env->reachable_flags;
            float* pv = env->priority_sort_values;
            for (int r = 0; r < env->num_rso; r++) {
                float dx = rsos[r].pos[0] - sx;
                float dy = rsos[r].pos[1] - sy;
                float dz = rsos[r].pos[2] - sz;
                float aa = dx * dx + dy * dy + dz * dz;
                float b = 2.0f * (sx * dx + sy * dy + sz * dz);
                float inv2a = 0.5f / aa;
                float tstar = -b * inv2a;
                float min_f = sat_c - b * b * inv2a * 0.5f;
                int blocked = use_earth & (tstar > 0.0f) & (tstar < 1.0f)
                            & (min_f < 0.0f);
                int out_of_range = use_range & (aa > max_r2);
                int reachable = !(blocked | out_of_range);
                rf[r] = reachable;
                pv[r] = reachable ? bp[r] : bp[r] - 1e6f;
            }
        }

        // Select action slots. Fixed-size LSTM baselines keep slot k tied to
        // global RSO k; dynamic attention policies keep the priority top-K.
        if (env->fixed_rso_slots) {
            for (int k = 0; k < env->rso_top_k; k++) {
                env->topk_out_indices[k] = (k < env->num_rso) ? k : -1;
                env->topk_out_values[k] = (k < env->num_rso) ? env->priority_sort_values[k] : -1e9f;
            }
        } else {
            top_k_desc(env->priority_sort_indices, env->priority_sort_values,
                       env->num_rso, env->rso_top_k,
                       env->topk_out_indices, env->topk_out_values);
        }

        int tokens_written = 0;
        for (int k = 0; k < env->rso_top_k; k++) {
            int rso_idx = env->topk_out_indices[k];
            if (rso_idx < 0 || rso_idx >= env->num_rso) {
                env->top_k_indices[a * env->rso_top_k + tokens_written] = -1;
                tokens_written++;
                continue;
            }
            RSO* rso = &env->rsos[rso_idx];

            // Store in top-K mapping
            env->top_k_indices[a * env->rso_top_k + tokens_written] = rso_idx;

            int base = env->rso_obs_offset + tokens_written * RSO_TOKEN_SIZE;

            float dx = rso->pos[0] - sat->pos[0];
            float dy = rso->pos[1] - sat->pos[1];
            float dz = rso->pos[2] - sat->pos[2];
            float local_dx = vec3_dot(dx, dy, dz, east[0], east[1], east[2]);
            float local_dy = vec3_dot(dx, dy, dz, north[0], north[1], north[2]);
            float local_dz = vec3_dot(dx, dy, dz, up[0], up[1], up[2]);

            // Relative position in the ego sensor's local frame.
            obs[base + 0] = tanhf(local_dx / env->obs_dist_norm);
            obs[base + 1] = tanhf(local_dy / env->obs_dist_norm);
            obs[base + 2] = tanhf(local_dz / env->obs_dist_norm);

            float dvx = rso->vel[0] - sat->vel[0];
            float dvy = rso->vel[1] - sat->vel[1];
            float dvz = rso->vel[2] - sat->vel[2];
            // Relative velocity in the same local frame.
            obs[base + 3] = tanhf(
                vec3_dot(dvx, dvy, dvz, east[0], east[1], east[2])
                / env->obs_vel_norm);
            obs[base + 4] = tanhf(
                vec3_dot(dvx, dvy, dvz, north[0], north[1], north[2])
                / env->obs_vel_norm);
            obs[base + 5] = tanhf(
                vec3_dot(dvx, dvy, dvz, up[0], up[1], up[2])
                / env->obs_vel_norm);

            // Uncertainty (log-scaled to keep bounded) — evaluate anchors + growth
            obs[base + 6] = tanhf(logf(1.0f + surrogate_u_r_now(rso, env) / env->obs_u_norm));
            obs[base + 7] = tanhf(logf(1.0f + surrogate_u_t_now(rso, env) / env->obs_u_norm));
            obs[base + 8] = tanhf(logf(1.0f + surrogate_u_n_now(rso, env) / env->obs_u_norm));
            obs[base + 9] = tanhf(logf(1.0f + surrogate_sigma_vel_now(rso, env) / env->obs_u_vel_norm));

            // Age (log-scaled then tanh-bounded)
            obs[base + 10] = tanhf(logf(1.0f + rso->age / env->obs_age_norm));

            // Size (normalised)
            obs[base + 11] = rso->size / 10.0f;

            // Visibility: reuse cached reachability, add FOV check
            float dist = vec3_len(dx, dy, dz);
            int reachable = env->reachable_flags[rso_idx];  // earth + range already checked
            int visible = reachable;

            // FOV check
            if (visible && dist > 1e-6f) {
                float dot = (dx * sat->sensor_dir[0] + dy * sat->sensor_dir[1]
                           + dz * sat->sensor_dir[2]) / dist;
                if (dot < fov_cos) visible = 0;
            }
            // Action feasibility excludes Earth-blocked and out-of-range RSOs,
            // while still allowing the sensor to slew toward a reachable target
            // that is not yet inside its current FoV. Keep this meaning identical
            // for fixed slots and dynamic top-K so OOD comparisons change only
            // candidate identity/order, not the available action set.
            obs[base + 12] = (float)reachable;

            // Do not leak the hidden range-specialization response. Mode 2
            // exposes only neutral angular proximity; the policy must learn
            // modality-by-distance utility from outcomes and reward.
            float quality = visible ? compute_obs_quality(env, sat,
                                                          rso->pos, rso->size) : 0.0f;
            if (env->quality_mode != 1 && dist > 1e-6f) {
                // Angular distance from boresight, normalized by FOV
                float dot_val = (dx * sat->sensor_dir[0] + dy * sat->sensor_dir[1]
                               + dz * sat->sensor_dir[2]) / dist;
                dot_val = clampf(dot_val, -1.0f, 1.0f);
                float ang_dist = acosf(dot_val);  // [0, PI]
                obs[base + 13] = clampf(1.0f - ang_dist / PI, 0.0f, 1.0f);
            } else {
                obs[base + 13] = quality;
            }

            // was_my_target: 1.0 if this RSO was the one I selected last step
            // Compare the selected global RSO against the current token mapping.
            obs[base + 14] = 0.0f;  // Default, set below

            // steps_since_i_observed (normalised)
            int sso = env->steps_since_observed[a * env->num_rso + rso_idx];
            obs[base + 15] = (sso < 0) ? -1.0f
                           : clampf((float)sso / env->obs_age_norm, 0.0f, 1.0f);

            // num_agents_observing (normalised by num_agents)
            obs[base + 16] = (float)rso->num_agents_observing / (float)env->num_agents;

            // ── Angular direction from sensor boresight to RSO (delta_az, delta_el) ──
            // These give the policy a direct signal for continuous control
            // without needing to learn atan2/asin from Cartesian positions.
            {
                // Compare boresight and line of sight in the ego-local frame.
                float bore_x = vec3_dot(sat->sensor_dir[0], sat->sensor_dir[1],
                                        sat->sensor_dir[2],
                                        east[0], east[1], east[2]);
                float bore_y = vec3_dot(sat->sensor_dir[0], sat->sensor_dir[1],
                                        sat->sensor_dir[2],
                                        north[0], north[1], north[2]);
                float bore_z = vec3_dot(sat->sensor_dir[0], sat->sensor_dir[1],
                                        sat->sensor_dir[2],
                                        up[0], up[1], up[2]);
                float s_el = asinf(clampf(bore_z, -1.0f, 1.0f));
                float s_az = atan2f(bore_y, bore_x);

                float r_dir_x = local_dx;
                float r_dir_y = local_dy;
                float r_dir_z = local_dz;
                float r_dist_for_dir = dist;
                if (r_dist_for_dir < 1e-6f) r_dist_for_dir = 1e-6f;
                r_dir_x /= r_dist_for_dir;
                r_dir_y /= r_dist_for_dir;
                r_dir_z /= r_dist_for_dir;

                float r_el = asinf(clampf(r_dir_z, -1.0f, 1.0f));
                float r_az = atan2f(r_dir_y, r_dir_x);

                // Delta angles
                float d_az = r_az - s_az;
                // Wrap to [-π, π]
                if (d_az > PI) d_az -= 2.0f * PI;
                if (d_az < -PI) d_az += 2.0f * PI;
                float d_el = r_el - s_el;

                // Normalize by max single-step slew (max_slew_rate * dt)
                // so ±1 = reachable in one step
                float max_delta = sat->max_slew_rate * env->dt;
                if (max_delta < 1e-8f) max_delta = 1e-8f;

                obs[base + 17] = clampf(d_az / max_delta, -3.0f, 3.0f) / 3.0f;  // [-1, 1]
                obs[base + 18] = clampf(d_el / max_delta, -3.0f, 3.0f) / 3.0f;  // [-1, 1]
            }

            tokens_written++;
        }

        // Zero-pad remaining top-K slots
        for (int k = tokens_written; k < env->rso_top_k; k++) {
            env->top_k_indices[a * env->rso_top_k + k] = -1;  // invalid
        }
    }
}

// ─── Set was_my_target flags (called after observation building) ───
// For discrete mode: marks the RSO that was selected last step.
// For continuous mode: marks the RSO closest to the sensor boresight in the top-K.
static void set_was_my_target(OrbitalEyesCooperative* env) {
    for (int a = 0; a < env->num_agents; a++) {
        Satellite* sat = &env->satellites[a];

        if (env->action_mode == 0) {
            // Discrete: mark the RSO that was explicitly targeted last step
            int prev_global_rso = -1;
            if (env->tick > 0) {
                prev_global_rso = sat->last_action;  // stores global RSO index
            }

            for (int k = 0; k < env->rso_top_k; k++) {
                int global_idx = env->top_k_indices[a * env->rso_top_k + k];
                if (global_idx < 0) continue;
                float* obs = env->observations + a * env->obs_size;
                int base = env->rso_obs_offset + k * RSO_TOKEN_SIZE;
                obs[base + 14] = (global_idx == prev_global_rso) ? 1.0f : 0.0f;
            }
        } else {
            // Continuous: mark the RSO closest to boresight among top-K
            float best_dot = -2.0f;
            int best_k = -1;
            for (int k = 0; k < env->rso_top_k; k++) {
                int global_idx = env->top_k_indices[a * env->rso_top_k + k];
                if (global_idx < 0) continue;
                RSO* rso = &env->rsos[global_idx];
                float dx = rso->pos[0] - sat->pos[0];
                float dy = rso->pos[1] - sat->pos[1];
                float dz = rso->pos[2] - sat->pos[2];
                float dist = vec3_len(dx, dy, dz);
                if (dist < 1e-6f) continue;
                float dot = (dx * sat->sensor_dir[0] + dy * sat->sensor_dir[1]
                           + dz * sat->sensor_dir[2]) / dist;
                if (dot > best_dot) {
                    best_dot = dot;
                    best_k = k;
                }
            }

            for (int k = 0; k < env->rso_top_k; k++) {
                int global_idx = env->top_k_indices[a * env->rso_top_k + k];
                if (global_idx < 0) continue;
                float* obs = env->observations + a * env->obs_size;
                int base = env->rso_obs_offset + k * RSO_TOKEN_SIZE;
                obs[base + 14] = (k == best_k) ? 1.0f : 0.0f;
            }
        }
    }
}

// ─── Step ───
void c_step(OrbitalEyesCooperative* env) {
    PROF_START(total_step);
    env->tick++;
    env->epoch_time += env->dt;

    // Zero rewards
    memset(env->rewards, 0, env->num_agents * sizeof(float));
    memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));
    float step_same_modality_duplicate_rate = 0.0f;

    // ── 1. Update satellite and RSO positions from orbital mechanics ──
    PROF_START(propagation);
    if (env->propagation_mode == 0) {
        // No J2 secular drift: RAAN/omega constant, so the perifocal→ECI basis
        // is an episode-invariant cached at reset. Fast path.
        for (int i = 0; i < env->num_agents; i++) {
            Satellite* s = &env->satellites[i];
            if (env->ground_sensor) {
                set_ground_sensor_state(env, s, 0);
                continue;
            }
            propagate_cached_rot(
                s->oe_a, s->oe_e, s->oe_M0, env->epoch_time,
                s->oe_n, s->oe_sqrt_mu_a, s->oe_sqrt_1_e2,
                s->oe_Px, s->oe_Py, s->oe_Pz, s->oe_Qx, s->oe_Qy, s->oe_Qz,
                &s->pos[0], &s->pos[1], &s->pos[2],
                &s->vel[0], &s->vel[1], &s->vel[2]);
        }
        for (int i = 0; i < env->num_rso; i++) {
            RSO* r = &env->rsos[i];
            propagate_cached_rot(
                r->oe_a, r->oe_e, r->oe_M0, env->epoch_time,
                r->oe_n, r->oe_sqrt_mu_a, r->oe_sqrt_1_e2,
                r->oe_Px, r->oe_Py, r->oe_Pz, r->oe_Qx, r->oe_Qy, r->oe_Qz,
                &r->pos[0], &r->pos[1], &r->pos[2],
                &r->vel[0], &r->vel[1], &r->vel[2]);
        }
    } else {
        // J2 secular drift active: RAAN/omega evolve, recompute rotation each step.
        for (int i = 0; i < env->num_agents; i++) {
            Satellite* s = &env->satellites[i];
            if (env->ground_sensor) {
                set_ground_sensor_state(env, s, 0);
                continue;
            }
            propagate_cached(
                s->oe_a, s->oe_e, s->oe_raan, s->oe_omega, s->oe_M0,
                env->epoch_time,
                s->oe_n, s->oe_sqrt_mu_a, s->oe_sqrt_1_e2,
                s->oe_cos_inc, s->oe_sin_inc,
                s->oe_raan_dot, s->oe_omega_dot,
                &s->pos[0], &s->pos[1], &s->pos[2],
                &s->vel[0], &s->vel[1], &s->vel[2]);
        }

        for (int i = 0; i < env->num_rso; i++) {
            RSO* r = &env->rsos[i];
            propagate_cached(
                r->oe_a, r->oe_e, r->oe_raan, r->oe_omega, r->oe_M0,
                env->epoch_time,
                r->oe_n, r->oe_sqrt_mu_a, r->oe_sqrt_1_e2,
                r->oe_cos_inc, r->oe_sin_inc,
                r->oe_raan_dot, r->oe_omega_dot,
                &r->pos[0], &r->pos[1], &r->pos[2],
                &r->vel[0], &r->vel[1], &r->vel[2]);
        }
    }
    PROF_END(env, propagation);

    // ── 2. Process actions: update sensor pointing ──
    PROF_START(actions);
    for (int a = 0; a < env->num_agents; a++) {
        Satellite* sat = &env->satellites[a];

        // Save pre-action sensor direction for control cost
        sat->prev_sensor_dir[0] = sat->sensor_dir[0];
        sat->prev_sensor_dir[1] = sat->sensor_dir[1];
        sat->prev_sensor_dir[2] = sat->sensor_dir[2];

        if (env->action_mode == 0) {
            // Discrete: agent selects index into top-K
            int action = env->actions[a];
            // Actions [0, K) select an RSO; action K means hold pointing.
            action = clampi(action, 0, env->rso_top_k);
            sat->last_action_idx = action;  // K is the explicit no-op action

            // Default: decode the discrete action into a global RSO via the
            // per-agent top-K map. When the benchmark enables reachable-set
            // access for classical baselines, a global target index is injected
            // directly (bypassing the top-K shortlist). Never active for the NN.
            int global_rso;
            if (env->external_target_active) {
                global_rso = env->external_targets[a];
            } else if (action < env->rso_top_k) {
                global_rso = env->top_k_indices[a * env->rso_top_k + action];
            } else {
                global_rso = -1;  // explicit no-op
            }
            // Holding also clears the previous target marker.
            sat->last_action = -1;
            if (global_rso >= 0 && global_rso < env->num_rso) {
                // Slew toward selected RSO
                RSO* target_rso = &env->rsos[global_rso];
                float target_dir[3];
                target_dir[0] = target_rso->pos[0] - sat->pos[0];
                target_dir[1] = target_rso->pos[1] - sat->pos[1];
                target_dir[2] = target_rso->pos[2] - sat->pos[2];
                vec3_normalize(&target_dir[0], &target_dir[1], &target_dir[2]);

                slew_sensor(sat->sensor_dir, target_dir,
                           sat->max_slew_rate, env->dt);

                // Store resolved global RSO index as last_action for was_my_target
                sat->last_action = global_rso;
            }
            // else: padded slot — no-op, keep current pointing
        } else {
            // Continuous: delta azimuth/elevation in [-1,1]
            float* act = (float*)env->actions + a * 2;
            float a0 = clampf(act[0], -1.0f, 1.0f);
            float a1 = clampf(act[1], -1.0f, 1.0f);
            float d_az = a0 * sat->max_slew_rate * env->dt;
            float d_el = a1 * sat->max_slew_rate * env->dt;

            // Convert current direction to spherical
            float r_len = vec3_len(sat->sensor_dir[0], sat->sensor_dir[1], sat->sensor_dir[2]);
            float el = asinf(clampf(sat->sensor_dir[2] / r_len, -1.0f, 1.0f));
            float az = atan2f(sat->sensor_dir[1], sat->sensor_dir[0]);

            az += d_az;
            el = clampf(el + d_el, -PI / 2.0f + 0.01f, PI / 2.0f - 0.01f);

            sat->sensor_dir[0] = cosf(el) * cosf(az);
            sat->sensor_dir[1] = cosf(el) * sinf(az);
            sat->sensor_dir[2] = sinf(el);

            sat->last_delta[0] = a0;
            sat->last_delta[1] = a1;
        }
    }

    // Pairwise task-duplication diagnostics. Under range specialization both
    // modalities perform the same measurement, so every duplicate assignment
    // consumes redundant capacity and is softly penalized.
    int active_pairs = 0;
    int active_same_modality_pairs = 0;
    int duplicate_pairs = 0;
    int same_modality_duplicate_pairs = 0;
    for (int a = 0; a < env->num_agents; a++) {
        int target_a = env->satellites[a].last_action;
        if (target_a < 0) continue;
        for (int b = a + 1; b < env->num_agents; b++) {
            int target_b = env->satellites[b].last_action;
            if (target_b < 0) continue;
            active_pairs++;
            int same_modality =
                env->satellites[a].sensor_type == env->satellites[b].sensor_type;
            if (same_modality) active_same_modality_pairs++;
            if (target_a != target_b) continue;
            duplicate_pairs++;
            if (same_modality) same_modality_duplicate_pairs++;
        }
    }
    if (active_pairs > 0) {
        env->duplicate_assignment_accum +=
            (float)duplicate_pairs / (float)active_pairs;
    }
    if (active_same_modality_pairs > 0) {
        step_same_modality_duplicate_rate =
            (float)same_modality_duplicate_pairs
            / (float)active_same_modality_pairs;
        env->same_modality_duplicate_accum +=
            step_same_modality_duplicate_rate;
    }
    PROF_END(env, actions);

    // ── 3. Determine observations and accumulate uncertainty reductions ──
    PROF_START(observation_det);
    // Zero reduction and per-sensor measurement buffers.
    memset(env->reduction_buffer, 0, env->num_rso * 4 * sizeof(float));
    memset(env->measurement_quality, 0,
           env->num_agents * env->num_rso * sizeof(float));
    memset(env->local_info_gain, 0, sizeof(env->local_info_gain));

    // Reset num_agents_observing
    for (int i = 0; i < env->num_rso; i++) {
        env->rsos[i].num_agents_observing = 0;
    }

    int total_observations = 0;
    int optical_observations = 0;
    int radar_observations = 0;

    for (int a = 0; a < env->num_agents; a++) {
        Satellite* sat = &env->satellites[a];
        float fov_cos = cosf(sat->fov);
        int agent_observed_any = 0;

        if (env->action_mode == 0 && !env->enable_fov_obs) {
            // Discrete: only the selected RSO is observed (if visible)
            int global_rso = sat->last_action;
            if (global_rso >= 0 && global_rso < env->num_rso) {
                RSO* rso = &env->rsos[global_rso];

                // Check visibility
                float dx = rso->pos[0] - sat->pos[0];
                float dy = rso->pos[1] - sat->pos[1];
                float dz = rso->pos[2] - sat->pos[2];
                float dist = vec3_len(dx, dy, dz);
                int visible = 1;

                if (dist > 1e-6f) {
                    float dot = (dx * sat->sensor_dir[0] + dy * sat->sensor_dir[1]
                               + dz * sat->sensor_dir[2]) / dist;
                    if (dot < fov_cos) visible = 0;
                }
                if (visible && env->enable_earth_mask) {
                    if (earth_occluded(sat->pos, rso->pos)) visible = 0;
                }
                if (visible && env->enable_range_mask) {
                    if (dist > sat->max_obs_range) visible = 0;
                }

                if (visible) {
                    float quality = compute_obs_quality(env, sat,
                                                       rso->pos, rso->size);
                    // Accumulate reduction
                    env->reduction_buffer[global_rso * 4 + 0] += env->alpha_r * quality;
                    env->reduction_buffer[global_rso * 4 + 1] += env->alpha_t * quality;
                    env->reduction_buffer[global_rso * 4 + 2] += env->alpha_n * quality;
                    env->reduction_buffer[global_rso * 4 + 3] += env->alpha_vel * quality;
                    env->measurement_quality[a * env->num_rso + global_rso] = quality;
                    if (sat->sensor_type == SENSOR_RADAR) radar_observations++;
                    else optical_observations++;
                    if (sat->sensor_type == SENSOR_TYPE_B) {
                        env->type_b_observation_range_accum += dist;
                        env->type_b_preferred_observations_accum +=
                            dist >= env->range_crossover_km;
                    } else {
                        env->type_a_observation_range_accum += dist;
                        env->type_a_preferred_observations_accum +=
                            dist <= env->range_crossover_km;
                    }
                    env->crossover_observations_accum +=
                        fabsf(dist - env->range_crossover_km)
                        <= 0.5f * env->range_transition_km;

                    rso->num_agents_observing++;
                    env->steps_since_observed[a * env->num_rso + global_rso] = 0;
                    total_observations++;
                    agent_observed_any = 1;
                }
            }
        } else {
            // Continuous: all RSOs within FOV are observed
            for (int r = 0; r < env->num_rso; r++) {
                RSO* rso = &env->rsos[r];

                float dx = rso->pos[0] - sat->pos[0];
                float dy = rso->pos[1] - sat->pos[1];
                float dz = rso->pos[2] - sat->pos[2];
                float dist = vec3_len(dx, dy, dz);
                int visible = 1;

                if (dist > 1e-6f) {
                    float dot = (dx * sat->sensor_dir[0] + dy * sat->sensor_dir[1]
                               + dz * sat->sensor_dir[2]) / dist;
                    if (dot < fov_cos) visible = 0;
                }
                if (visible && env->enable_earth_mask) {
                    if (earth_occluded(sat->pos, rso->pos)) visible = 0;
                }
                if (visible && env->enable_range_mask) {
                    if (dist > sat->max_obs_range) visible = 0;
                }

                if (visible) {
                    float quality = compute_obs_quality(env, sat,
                                                       rso->pos, rso->size);
                    env->reduction_buffer[r * 4 + 0] += env->alpha_r * quality;
                    env->reduction_buffer[r * 4 + 1] += env->alpha_t * quality;
                    env->reduction_buffer[r * 4 + 2] += env->alpha_n * quality;
                    env->reduction_buffer[r * 4 + 3] += env->alpha_vel * quality;
                    env->measurement_quality[a * env->num_rso + r] = quality;
                    if (sat->sensor_type == SENSOR_RADAR) radar_observations++;
                    else optical_observations++;
                    if (sat->sensor_type == SENSOR_TYPE_B) {
                        env->type_b_observation_range_accum += dist;
                        env->type_b_preferred_observations_accum +=
                            dist >= env->range_crossover_km;
                    } else {
                        env->type_a_observation_range_accum += dist;
                        env->type_a_preferred_observations_accum +=
                            dist <= env->range_crossover_km;
                    }
                    env->crossover_observations_accum +=
                        fabsf(dist - env->range_crossover_km)
                        <= 0.5f * env->range_transition_km;

                    rso->num_agents_observing++;
                    env->steps_since_observed[a * env->num_rso + r] = 0;
                    total_observations++;
                    agent_observed_any = 1;
                }
            }
        }

        // Store whether this agent observed anything this step
        sat->observed_any = agent_observed_any;
    }

    int complementary_fusions = 0;
    for (int r = 0; r < env->num_rso; r++) {
        int optical_seen = 0;
        int radar_seen = 0;
        for (int a = 0; a < env->num_agents; a++) {
            if (env->measurement_quality[a * env->num_rso + r] <= 0.0f) continue;
            if (env->satellites[a].sensor_type == SENSOR_RADAR) radar_seen = 1;
            else optical_seen = 1;
        }
        if (optical_seen && radar_seen) complementary_fusions++;
    }
    env->optical_observations_accum += (float)optical_observations;
    env->radar_observations_accum += (float)radar_observations;
    env->complementary_fusion_accum +=
        (float)complementary_fusions / (float)env->num_rso;
    PROF_END(env, observation_det);

    // ── 4. Apply uncertainty dynamics (visibility-gated, lazy anchors) ──
    //
    // Anchor semantics:
    //   r->u_{r,t,n}      = value at last observation event
    //   r->sigma_vel      = velocity uncertainty at last obs
    //   r->u_anchor_sum   = cached sum of position anchors
    //   r->age            = time since that anchor was set
    //
    // For NOT-observed RSOs we do nothing but bump `age`. Consumers that need
    // the current value call surrogate_*_now(rso, env), which evaluates the
    // closed form on the fly. For OBSERVED RSOs we:
    //   1. compute current value at age = age + dt
    //   2. apply the reduction (additive or multiplicative)
    //   3. clamp to floor, store back as the new anchor, recompute sum
    //   4. reset age = 0
    PROF_START(uncertainty);
    float mean_age = 0.0f;
    for (int i = 0; i < env->num_rso; i++) {
        RSO* r = &env->rsos[i];

        float red_r = env->reduction_buffer[i * 4 + 0];
        float red_t = env->reduction_buffer[i * 4 + 1];
        float red_n = env->reduction_buffer[i * 4 + 2];
        float red_v = env->reduction_buffer[i * 4 + 3];

        if (env->uncertainty_mode == 2) {
            ekf_predict_fixed_dt_order2(r->P, r->pos, env->ekf_sigma_a, env->dt);
            if (red_r > 0.0f || red_t > 0.0f || red_n > 0.0f) {
                for (int a = 0; a < env->num_agents; a++) {
                    float quality =
                        env->measurement_quality[a * env->num_rso + i];
                    if (quality <= 0.0f) continue;
                    env->local_info_gain[a] +=
                        ekf_independent_information_gain(
                            r->P, &env->satellites[a], r, quality);
                }
                ekf_apply_buffered_measurements(env, i);
                r->age = 0.0f;
            } else {
                r->age += env->dt;
            }
            ekf_sync_rso_uncertainty(r, env);
            mean_age += r->age;
            continue;
        }

        if (red_r > 0.0f || red_t > 0.0f || red_n > 0.0f) {
            // ── Observed this step: catch up to NOW, then reduce ──
            float dt_unseen = r->age + env->dt;

            if (env->uncertainty_mode == 1) {
                // ── Hybrid EKF + surrogate: run the real EKF at this edge ──
                // ── Surrogate prior at NOW (physical uncertainty model) ──
                // A measurement can only *reduce* uncertainty, so this is a hard
                // upper bound on any sane post-update EKF sigma. Used by the
                // fresh-anchor re-init below and by the divergence guard.
                float age_now = r->age + env->dt;
                float ur_now = r->u_r + (env->growth_a_r + 0.5f * env->growth_b_r * age_now) * age_now;
                float ut_now = r->u_t + (env->growth_a_t + 0.5f * env->growth_b_t * age_now) * age_now;
                float un_now = r->u_n + (env->growth_a_n + 0.5f * env->growth_b_n * age_now) * age_now;
                ur_now = fmaxf(ur_now, env->u_min_r);
                ut_now = fmaxf(ut_now, env->u_min_t);
                un_now = fmaxf(un_now, env->u_min_n);
                float sv_now = r->sigma_vel + (env->growth_mode >= 1 ? env->k_v * age_now : 0.0f);
                sv_now = fmaxf(sv_now, env->vel_u_min);

                if (!r->ekf_anchor_fresh) {
                    // ── First observation this episode ──
                    // The prior P was built from the catalog (age_init) state.
                    // Propagating a large prior P over a multi-orbit gap causes
                    // Riccati blow-up (gravity-gradient cross-coupling makes P
                    // non-PD after a few substeps). But for a large prior the
                    // measurement dominates regardless, so we skip propagation:
                    // re-init P from the current SURROGATE uncertainty (which
                    // carries the prior correctly via anchor+age growth) then
                    // apply the measurement update directly.
                    ekf_init_P(r->P, r->pos, r->vel, ur_now, ut_now, un_now, sv_now);
                    ekf_apply_buffered_measurements(env, i);
                    r->ekf_anchor_fresh = 1;
                } else {
                    // ── Intra-episode gap: P is small after the last update ──
                    // Safe to substep-propagate with re-linearisation.
                    int n_sub = 1;
                    if (env->ekf_max_substep_dt > 0.0f
                            && dt_unseen > env->ekf_max_substep_dt) {
                        n_sub = (int)ceilf(dt_unseen / env->ekf_max_substep_dt);
                        if (n_sub < 1) n_sub = 1;
                    }
                    float dt_sub = dt_unseen / (float)n_sub;
                    // Gap spans [t_start, epoch_time]; march forward, recomputing
                    // F at each sub-step midpoint along the Keplerian arc.
                    float t_start = env->epoch_time - dt_unseen;
                    float F[36], Phi[36];
                    for (int s = 0; s < n_sub; s++) {
                        float t_mid = t_start + ((float)s + 0.5f) * dt_sub;
                        float rx, ry, rz, vx, vy, vz;
                        propagate_cached(
                            r->oe_a, r->oe_e, r->oe_raan, r->oe_omega, r->oe_M0,
                            t_mid,
                            r->oe_n, r->oe_sqrt_mu_a, r->oe_sqrt_1_e2,
                            r->oe_cos_inc, r->oe_sin_inc,
                            r->oe_raan_dot, r->oe_omega_dot,
                            &rx, &ry, &rz, &vx, &vy, &vz);
                        float r_mid[3] = {rx, ry, rz};
                        ekf_dynamics_jacobian(r_mid, F);
                        ekf_state_transition(F, dt_sub, Phi);
                        ekf_propagate_P(r->P, Phi, env->ekf_sigma_a, dt_sub);
                    }
                    ekf_apply_buffered_measurements(env, i);
                }

                float sig_r, sig_t, sig_n, sig_v;
                ekf_rtn_sigmas(r->P, r->pos, r->vel,
                               &sig_r, &sig_t, &sig_n, &sig_v);
                // ── EKF divergence guard ──
                // In fp32 the Joseph update can lose positive-definiteness when
                // the same RSO is observed repeatedly with a precise measurement
                // (position/velocity dynamic range + dynamics-induced cross-
                // covariance), making P non-PD and causing runaway growth. A
                // measurement can never make the estimate worse than the prior,
                // so if any position-axis sigma exceeds the surrogate prior the
                // filter has diverged numerically. Rebuild P from the surrogate
                // prior (well-conditioned, identical to the fresh-anchor path)
                // and re-apply the measurement.
                if (sig_r > ur_now || sig_t > ut_now || sig_n > un_now) {
                    ekf_init_P(r->P, r->pos, r->vel, ur_now, ut_now, un_now, sv_now);
                    ekf_apply_buffered_measurements(env, i);
                    ekf_rtn_sigmas(r->P, r->pos, r->vel,
                                   &sig_r, &sig_t, &sig_n, &sig_v);
                }
                // Optional hard clamp to bound worst-case spikes.
                if (env->ekf_sigma_max > 0.0f) {
                    float smax = env->ekf_sigma_max;
                    sig_r = fminf(sig_r, smax);
                    sig_t = fminf(sig_t, smax);
                    sig_n = fminf(sig_n, smax);
                    sig_v = fminf(sig_v, smax * 1e-3f);
                }
                r->u_r = fmaxf(env->u_min_r, sig_r);
                r->u_t = fmaxf(env->u_min_t, sig_t);
                r->u_n = fmaxf(env->u_min_n, sig_n);
                r->sigma_vel = fmaxf(env->vel_u_min, sig_v);
            } else {
                // ── Surrogate only: existing additive/multiplicative path ──
                // Advance age so the closed form evaluates at end-of-step.
                r->age = dt_unseen;
                float u_r_now = surrogate_u_r_now(r, env);
                float u_t_now = surrogate_u_t_now(r, env);
                float u_n_now = surrogate_u_n_now(r, env);
                float sv_now  = surrogate_sigma_vel_now(r, env);

                if (env->reduction_mode == 0) {
                    r->u_r = fmaxf(env->u_min_r, u_r_now - red_r);
                    r->u_t = fmaxf(env->u_min_t, u_t_now - red_t);
                    r->u_n = fmaxf(env->u_min_n, u_n_now - red_n);
                    r->sigma_vel = fmaxf(env->vel_u_min, sv_now - red_v);
                } else {
                    float factor_r = fmaxf(0.0f, 1.0f - red_r);
                    float factor_t = fmaxf(0.0f, 1.0f - red_t);
                    float factor_n = fmaxf(0.0f, 1.0f - red_n);
                    r->u_r = fmaxf(env->u_min_r, u_r_now * factor_r);
                    r->u_t = fmaxf(env->u_min_t, u_t_now * factor_t);
                    r->u_n = fmaxf(env->u_min_n, u_n_now * factor_n);
                    r->sigma_vel = fmaxf(env->vel_u_min,
                        sv_now * fmaxf(0.0f, 1.0f - red_v));
                }
            }
            r->u_anchor_sum = r->u_r + r->u_t + r->u_n;
            r->age = 0.0f;
        } else {
            // ── Not observed: just accumulate age. No growth math here. ──
            r->age += env->dt;
        }
        mean_age += r->age;
    }

    // Increment steps_since_observed for all agent-RSO pairs (except those reset to 0 above)
    for (int i = 0; i < env->num_agents * env->num_rso; i++) {
        if (env->steps_since_observed[i] >= 0) {
            env->steps_since_observed[i]++;
        }
        // Reset the ones that were set to 0 this step (they were incremented to 1)
        // Actually, we set to 0 during observation determination, then increment here
        // to 1, which is correct: "1 step since last observation"
    }

    PROF_END(env, uncertainty);

    // ── 5. Compute reward ──
    PROF_START(reward);
    float r_direct;

    float reward_badness_curr = 0.0f;
    if (env->reward_mode == 1) {
        // ── Level-based reward: smooth, progressive ──
        // r = 1 - mean_U / U_ref
        // At reset (mean_U = U_ref): r = 0
        // As uncertainty decreases: r increases toward 1
        // Smooth, no threshold jumps, naturally asymptotes.
        float sum_U = 0.0f;
        for (int i = 0; i < env->num_rso; i++) {
            sum_U += surrogate_u_sum_now(&env->rsos[i], env);
        }
        float mean_U = sum_U / (float)env->num_rso;
        r_direct = 1.0f - mean_U / env->U_ref;
    } else if (env->reward_mode == 2 || env->reward_mode == 3) {
        // ── Risk-aware catalogue reward ──
        // Mode 2 is potential-only: reward reductions in catalogue badness.
        // Mode 3 adds a dense level term, rewarding every step spent with a
        // catalogue healthier than the episode's initial catalogue. This gives
        // PPO a stronger signal than a nearly telescoping potential alone.
        reward_badness_curr = compute_catalogue_badness(env, NULL, NULL);
        r_direct = env->reward_delta_scale * (env->badness_prev - reward_badness_curr);
        if (env->reward_mode == 3) {
            float level = 1.0f - reward_badness_curr / fmaxf(env->badness_initial, 1e-6f);
            r_direct += env->reward_level_scale * level;
        }
        env->badness_prev = reward_badness_curr;
    } else if (env->reward_mode == 4) {
        // Tail-sensitive risk reward: emphasize the worst tail and outlier.
        reward_badness_curr = compute_tail_sensitive_badness(env);
        r_direct = env->reward_delta_scale * (env->badness_prev - reward_badness_curr);
        float level = 1.0f - reward_badness_curr / fmaxf(env->badness_initial, 1e-6f);
        r_direct += env->reward_level_scale * level;
        env->badness_prev = reward_badness_curr;
    } else if (env->reward_mode == 5) {
        // Fixed-target precision reward: no per-episode normalization.
        reward_badness_curr = compute_precision_badness(env);
        r_direct = env->reward_delta_scale
                  * (env->badness_prev - reward_badness_curr);
        r_direct += env->reward_level_scale
                  * (precision_goal_badness(env) - reward_badness_curr);
        // Give a bounded positive signal for entering and maintaining the
        // difficult fine max-U regime. This is deliberately independent of
        // the coarse badness scale so the reward cannot grow with U.
        if (env->reward_fine_bonus_scale > 0.0f) {
            r_direct += env->reward_fine_bonus_scale
                      * compute_fine_max_score(env);
        }
        if (env->reward_fine_mean_bonus_scale > 0.0f) {
            r_direct += env->reward_fine_mean_bonus_scale
                      * compute_fine_mean_score(env);
        }
        env->badness_prev = reward_badness_curr;
    } else if (env->reward_mode == 0) {
        // ── Mode 0: original delta-J reward ──
        float J_curr = compute_population_objective(env);
        r_direct = (env->J_prev - J_curr + env->reward_baseline)
                 / env->reward_scale;
        env->J_prev = J_curr;
    } else {
        // Unknown reward mode: fail closed instead of silently using delta-J.
        r_direct = 0.0f;
    }


    int done = (env->tick >= env->max_steps);
    if (done && (env->reward_mode == 2 || env->reward_mode == 3 || env->reward_mode == 4 || env->reward_mode == 5)
            && env->reward_terminal_bonus != 0.0f) {
        if (reward_badness_curr <= 0.0f) {
            if (env->reward_mode == 4) {
                reward_badness_curr = compute_tail_sensitive_badness(env);
            } else if (env->reward_mode == 5) {
                reward_badness_curr = compute_precision_badness(env);
            } else {
                reward_badness_curr = compute_catalogue_badness(env, NULL, NULL);
            }
        }
        if (env->reward_mode == 5) {
            r_direct += env->reward_terminal_bonus
                      * (precision_goal_badness(env) - reward_badness_curr);
        } else {
            r_direct += env->reward_terminal_bonus
                      * (1.0f - reward_badness_curr / fmaxf(env->badness_initial, 1e-6f));
        }
    }

    // Mix the shared catalogue objective with an order-independent local
    // covariance-gain credit. team_spirit=0 keeps that credit individual;
    // team_spirit=1 shares its mean while retaining the common team objective.
    float mean_local_gain = 0.0f;
    for (int a = 0; a < env->num_agents; a++) {
        env->local_info_gain[a] = clampf(env->local_info_gain[a], 0.0f, 1.0f);
        mean_local_gain += env->local_info_gain[a];
        if (env->satellites[a].sensor_type == SENSOR_RADAR) {
            env->radar_information_gain_accum += env->local_info_gain[a];
        } else {
            env->optical_information_gain_accum += env->local_info_gain[a];
        }
    }
    mean_local_gain /= fmaxf((float)env->num_agents, 1.0f);
    env->local_information_gain_accum += mean_local_gain;
    float spirit = clampf(env->team_spirit, 0.0f, 1.0f);

    for (int a = 0; a < env->num_agents; a++) {
        float local_credit = (1.0f - spirit) * env->local_info_gain[a]
                           + spirit * mean_local_gain;
        float reward = r_direct + env->alpha_local * local_credit;

        // Charge duplicate tasking to all participating sensors.
        int active_peers = 0;
        int duplicate_peers = 0;
        int target_a = env->satellites[a].last_action;
        if (target_a >= 0) {
            for (int b = 0; b < env->num_agents; b++) {
                if (a == b || env->satellites[b].last_action < 0) continue;
                active_peers++;
                if (target_a == env->satellites[b].last_action) {
                    duplicate_peers++;
                }
            }
        }
        if (active_peers > 0) {
            reward -= env->redundancy_penalty
                    * (float)duplicate_peers
                    / (float)active_peers;
        }

        // Optional quadratic control cost, normalized by the maximum possible
        // slew for the sensor modality during one environment step.
        if (env->alpha_ctrl > 0.0f) {
            Satellite* sat = &env->satellites[a];
            float max_angle = sat->max_slew_rate * env->dt;
            if (max_angle > 1e-8f) {
                float dot = sat->prev_sensor_dir[0] * sat->sensor_dir[0]
                          + sat->prev_sensor_dir[1] * sat->sensor_dir[1]
                          + sat->prev_sensor_dir[2] * sat->sensor_dir[2];
                dot = clampf(dot, -1.0f, 1.0f);
                float angle = acosf(dot);
                float frac = clampf(angle / max_angle, 0.0f, 1.0f);
                reward -= env->alpha_ctrl * frac * frac;
            }
        }

        // Clamp only after agent-specific credits and costs are included.
        if (env->reward_clip_min < 0.0f && reward < env->reward_clip_min) {
            reward = env->reward_clip_min;
        }
        if (env->reward_clip_max > 0.0f && reward > env->reward_clip_max) {
            reward = env->reward_clip_max;
        }
        env->rewards[a] = reward;
    }

    // Accumulate current episode stats separately from env->log. vec_log clears
    // env->log every report_interval, so using it as the live accumulator makes
    // episode_return depend on logging cadence instead of the episode.
    for (int a = 0; a < env->num_agents; a++) {
        env->episode_return_accum += env->rewards[a];
    }
    env->total_observations_accum += (float)total_observations;

    PROF_END(env, reward);

    // ── 6. Check termination ──
    if (done) {
        for (int a = 0; a < env->num_agents; a++) {
            env->terminals[a] = 1;
        }

        // Finalise log
        float episode_return = env->episode_return_accum / (float)env->num_agents;
        env->log.episode_return += episode_return;
        env->log.episode_length += (float)env->tick;
        env->log.mean_step_reward += episode_return / fmaxf((float)env->tick, 1.0f);
        env->log.total_observations += env->total_observations_accum;

        // Compute final uncertainty stats
        float sum_U = 0.0f, max_U = 0.0f;
        int above_threshold = 0;
        for (int i = 0; i < env->num_rso; i++) {
            float U = surrogate_u_sum_now(&env->rsos[i], env);
            sum_U += U;
            if (U > max_U) max_U = U;
            if (U > env->uncertainty_threshold) above_threshold++;
        }
        float final_mean_U = sum_U / (float)env->num_rso;
        float final_frac_above = (float)above_threshold / (float)env->num_rso;
        env->log.mean_uncertainty += final_mean_U;
        env->log.max_uncertainty += max_U;
        env->log.fraction_above_threshold += final_frac_above;
        env->log.mean_age += mean_age / (float)env->num_rso;
        env->log.effective_tracked += (float)env->num_rso * (1.0f - final_frac_above);
        env->log.final_u_ratio += final_mean_U / fmaxf(env->U_ref, 1e-6f);
        env->log.u_ref += env->U_ref;
        float final_tail_badness = 0.0f;
        float final_frac_above_for_badness = 0.0f;
        float final_catalogue_badness = compute_catalogue_badness(
            env, &final_tail_badness, &final_frac_above_for_badness);
        (void)final_frac_above_for_badness;
        env->log.catalogue_badness += final_catalogue_badness;
        env->log.tail_badness += final_tail_badness;
        env->log.initial_mean_uncertainty += env->initial_mean_uncertainty;
        env->log.initial_max_uncertainty += env->initial_max_uncertainty;
        env->log.initial_fraction_above_threshold += env->initial_fraction_above_threshold;
        env->log.initial_catalogue_badness += env->badness_initial;
        float inv_steps = 1.0f / fmaxf((float)env->tick, 1.0f);
        env->log.duplicate_assignment_rate +=
            env->duplicate_assignment_accum * inv_steps;
        env->log.same_modality_duplicate_rate +=
            env->same_modality_duplicate_accum * inv_steps;
        env->log.complementary_fusion_rate +=
            env->complementary_fusion_accum * inv_steps;
        env->log.optical_observations += env->optical_observations_accum;
        env->log.radar_observations += env->radar_observations_accum;
        env->log.mean_local_information_gain +=
            env->local_information_gain_accum * inv_steps;
        env->log.optical_information_gain +=
            env->optical_information_gain_accum;
        env->log.radar_information_gain +=
            env->radar_information_gain_accum;
        env->log.mean_type_a_observation_range +=
            env->type_a_observation_range_accum
            / fmaxf(env->optical_observations_accum, 1.0f);
        env->log.mean_type_b_observation_range +=
            env->type_b_observation_range_accum
            / fmaxf(env->radar_observations_accum, 1.0f);
        env->log.type_a_preferred_fraction +=
            env->type_a_preferred_observations_accum
            / fmaxf(env->optical_observations_accum, 1.0f);
        env->log.type_b_preferred_fraction +=
            env->type_b_preferred_observations_accum
            / fmaxf(env->radar_observations_accum, 1.0f);
        env->log.crossover_observation_fraction +=
            env->crossover_observations_accum
            / fmaxf(env->total_observations_accum, 1.0f);

        env->log.n += 1.0f;
        c_reset(env);
        PROF_END(env, total_step);
        env->prof_step_count++;
        return;  // observations already rebuilt by c_reset
    }

    // ── 7. Build observations ──
    PROF_START(build_obs);
    compute_observations(env);
    set_was_my_target(env);
    PROF_END(env, build_obs);

    PROF_END(env, total_step);
    env->prof_step_count++;
}

// ─── Render helpers ───

// Draw a wireframe cone from apex along direction with half-angle and length.
// Uses line segments to form a conical wireframe. Neon look on dark background.
static void draw_sensor_cone(Vector3 apex, float dir_x, float dir_y, float dir_z,
                              float half_angle, float length, int segments, Color col) {
    // Build a local frame: forward = dir, right and up perpendicular
    float fx = dir_x, fy = dir_y, fz = dir_z;

    // Pick an arbitrary non-parallel vector for cross product
    float ux, uy, uz;
    if (fabsf(fx) < 0.9f) { ux = 1; uy = 0; uz = 0; }
    else                   { ux = 0; uy = 1; uz = 0; }

    // right = normalize(forward × up_guess)
    float rx, ry, rz;
    vec3_cross(fx, fy, fz, ux, uy, uz, &rx, &ry, &rz);
    float rlen = vec3_len(rx, ry, rz);
    if (rlen < 1e-12f) return;
    rx /= rlen; ry /= rlen; rz /= rlen;

    // up = right × forward
    vec3_cross(rx, ry, rz, fx, fy, fz, &ux, &uy, &uz);
    float ulen = vec3_len(ux, uy, uz);
    if (ulen < 1e-12f) return;
    ux /= ulen; uy /= ulen; uz /= ulen;

    float cone_radius = length * tanf(half_angle);
    float tip_x = apex.x + fx * length;
    float tip_y = apex.y + fy * length;
    float tip_z = apex.z + fz * length;

    Vector3 prev_ring = {0};
    for (int i = 0; i <= segments; i++) {
        float theta = 2.0f * PI * (float)i / (float)segments;
        float ct = cosf(theta), st = sinf(theta);

        Vector3 ring_pt = {
            tip_x + cone_radius * (ct * rx + st * ux),
            tip_y + cone_radius * (ct * ry + st * uy),
            tip_z + cone_radius * (ct * rz + st * uz)
        };

        // Spoke line from apex to ring
        if (i % (segments / 4) == 0) {
            DrawLine3D(apex, ring_pt, col);
        }
        // Ring segment
        if (i > 0) {
            DrawLine3D(prev_ring, ring_pt, col);
        }
        prev_ring = ring_pt;
    }
}

// ─── Render ───
// Diorama mode: all ECI km positions are scaled so Earth radius = 1.0 unit.
// This keeps everything inside Raylib's default clip planes (near=0.01, far=1000).
// Satellites/RSOs are drawn with exaggerated sizes for visibility.
#define RENDER_SCALE (1.0f / EARTH_RADIUS_KM)
// ECI(x,y,z) → Raylib Y-up (x, z, -y) scaled to diorama. We negate y→z so
// the ECI +x/+y plane maps to Raylib +x/−z, keeping right-hand orientation.
static inline Vector3 eci_to_rl(float ex, float ey, float ez) {
    return (Vector3){ ex * RENDER_SCALE, ez * RENDER_SCALE, -ey * RENDER_SCALE };
}

void c_render(OrbitalEyesCooperative* env) {
    if (!env->client) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(1920, 1080, "Orbital Eyes — SSA Sensor Scheduling");
        if (env->render_fps <= 0) env->render_fps = 60;
        SetTargetFPS(env->render_fps);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->cam_yaw   = 45.0f;
        env->client->cam_pitch = 25.0f;
        env->client->cam_dist  = 3.5f;  // ~3.5 Earth radii from center
        env->client->camera = (Camera3D){
            .position   = (Vector3){0, 2.0f, 3.0f},
            .target     = (Vector3){0, 0, 0},
            .up         = (Vector3){0, 1, 0},
            .fovy       = 60.0f,
            .projection = CAMERA_PERSPECTIVE,
        };
    }

    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    // FPS control: = to increase, - to decrease (tap, not hold)
    if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
        env->render_fps = clampi(env->render_fps + 5, 1, 240);
        SetTargetFPS(env->render_fps);
    }
    if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
        env->render_fps = clampi(env->render_fps - 5, 1, 240);
        SetTargetFPS(env->render_fps);
    }

    Client* cl = env->client;

    // ── Camera controls ──
    float cam_speed = 0.04f;
    if (IsKeyDown(KEY_W)) cl->camera.target.z -= cam_speed;
    if (IsKeyDown(KEY_S)) cl->camera.target.z += cam_speed;
    if (IsKeyDown(KEY_A)) cl->camera.target.x -= cam_speed;
    if (IsKeyDown(KEY_D)) cl->camera.target.x += cam_speed;
    if (IsKeyDown(KEY_UP))    cl->cam_pitch += 1.0f;
    if (IsKeyDown(KEY_DOWN))  cl->cam_pitch -= 1.0f;
    if (IsKeyDown(KEY_LEFT))  cl->cam_yaw   -= 1.0f;
    if (IsKeyDown(KEY_RIGHT)) cl->cam_yaw   += 1.0f;

    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
        Vector2 delta = GetMouseDelta();
        cl->cam_yaw   -= delta.x * 0.3f;
        cl->cam_pitch += delta.y * 0.3f;
    }

    cl->cam_dist += GetMouseWheelMove() * -0.3f;
    if (cl->cam_dist < 1.5f)  cl->cam_dist = 1.5f;
    if (cl->cam_dist > 20.0f) cl->cam_dist = 20.0f;
    cl->cam_pitch = clampf(cl->cam_pitch, 5.0f, 89.0f);

    float yaw_rad  = cl->cam_yaw   * DEG2RAD;
    float pitch_rad = cl->cam_pitch * DEG2RAD;
    cl->camera.position.x = cl->camera.target.x + cl->cam_dist * cosf(pitch_rad) * sinf(yaw_rad);
    cl->camera.position.z = cl->camera.target.z + cl->cam_dist * cosf(pitch_rad) * cosf(yaw_rad);
    cl->camera.position.y = cl->camera.target.y + cl->cam_dist * sinf(pitch_rad);

    BeginDrawing();
    ClearBackground((Color){4, 6, 14, 255});

    BeginMode3D(cl->camera);

    // ── Earth: wireframe globe (radius = 1.0 in diorama units) ──
    DrawSphereWires((Vector3){0, 0, 0}, 1.0f, 24, 24, (Color){20, 60, 130, 200});
    DrawSphereWires((Vector3){0, 0, 0}, 1.003f, 16, 16, (Color){30, 80, 160, 100});
    // Equator ring
    DrawCircle3D((Vector3){0, 0, 0}, 1.005f, (Vector3){1, 0, 0}, 90.0f,
                 (Color){40, 100, 180, 220});

    // Neon color palette (indexed by orbit_id for multi-orbit distinction)
    Color orbit_colors[] = {
        {0, 255, 200, 255},   // orbit 0: cyan-green (SSO default)
        {255, 100, 255, 255}, // orbit 1: magenta
        {255, 255, 0, 255},   // orbit 2: yellow
        {0, 200, 255, 255},   // orbit 3: sky blue
        {255, 150, 0, 255},   // orbit 4: orange
        {100, 255, 50, 255},  // orbit 5: lime green
        {180, 80, 255, 255},  // orbit 6: purple
        {0, 255, 120, 255},   // orbit 7: spring green
    };
    int nsc = 8;

    // Exaggerated visual sizes (in diorama units, 1 unit = 1 Earth radius)
    float sat_size  = 0.03f;   // satellite marker radius (50% of original)
    float rso_scale = (env->render_rso_scale > 0.0f) ? env->render_rso_scale : 1.0f;
    float rso_size  = 0.008f * rso_scale;  // RSO marker base radius, scaled
    // Pre-compute per-RSO visibility: can this RSO be seen by at least one
    // satellite (not earth-occluded, not out of range)? Ignores sensor pointing.
    int rso_visible_any[env->num_rso];
    for (int i = 0; i < env->num_rso; i++) {
        rso_visible_any[i] = 0;
        RSO* r = &env->rsos[i];
        for (int a = 0; a < env->num_agents && !rso_visible_any[i]; a++) {
            Satellite* s = &env->satellites[a];
            float dx = r->pos[0] - s->pos[0];
            float dy = r->pos[1] - s->pos[1];
            float dz = r->pos[2] - s->pos[2];
            float dist = vec3_len(dx, dy, dz);
            if (dist < 1e-6f) { rso_visible_any[i] = 1; continue; }
            if (env->enable_earth_mask && earth_occluded(s->pos, r->pos)) continue;
            if (env->enable_range_mask && dist > s->max_obs_range) continue;
            rso_visible_any[i] = 1;
        }
    }

    // ── Satellites ──
    for (int i = 0; i < env->num_agents; i++) {
        Satellite* s = &env->satellites[i];
        Vector3 pos = eci_to_rl(s->pos[0], s->pos[1], s->pos[2]);
        Color col = (s->sensor_type == SENSOR_RADAR)
                    ? (Color){247, 37, 133, 255}
                    : (Color){0, 184, 217, 255};
        float cone_len = env->enable_range_mask
                       ? s->max_obs_range * RENDER_SCALE : 0.5f;

        // Sensor body
        DrawSphereWires(pos, sat_size, 8, 8, col);
        // Glow halo
        Color glow = col; glow.a = 60;
        DrawSphereWires(pos, sat_size * 1.6f, 6, 6, glow);

        // Nadir line (satellite to Earth center)
        Color nadir_col = col; nadir_col.a = 50;
        DrawLine3D(pos, (Vector3){0, 0, 0}, nadir_col);

        // Sensor boresight direction (convert to Raylib frame, normalize only)
        Vector3 bore_dir = eci_to_rl(s->sensor_dir[0], s->sensor_dir[1], s->sensor_dir[2]);
        // eci_to_rl applies RENDER_SCALE so re-normalize the direction
        float bd_len = sqrtf(bore_dir.x*bore_dir.x + bore_dir.y*bore_dir.y + bore_dir.z*bore_dir.z);
        if (bd_len > 1e-9f) { bore_dir.x /= bd_len; bore_dir.y /= bd_len; bore_dir.z /= bd_len; }

        // Boresight line
        Vector3 bore_end = {
            pos.x + bore_dir.x * cone_len * 1.5f,
            pos.y + bore_dir.y * cone_len * 1.5f,
            pos.z + bore_dir.z * cone_len * 1.5f
        };
        DrawLine3D(pos, bore_end, col);

        // FOV cone (wireframe)
        Color cone_col = col; cone_col.a = 200;
        draw_sensor_cone(pos, bore_dir.x, bore_dir.y, bore_dir.z,
                         s->fov, cone_len, 16, cone_col);
    }

    // ── RSOs: dual-shell visualization ──
    // Outer shell = uncertainty level (cyan=low → magenta=high)
    // Inner shell = visibility status (green=visible to ≥1 sat, red=masked)
    for (int i = 0; i < env->num_rso; i++) {
        RSO* r = &env->rsos[i];
        float U = surrogate_u_sum_now(r, env);

        // Outer shell: binary threshold — cyan if below, magenta if above
        Color outer_col;
        if (U > env->uncertainty_threshold) {
            outer_col = (Color){255, 0, 180, 200};   // magenta
        } else {
            outer_col = (Color){0, 255, 220, 200};   // cyan
        }

        // Inner shell: visibility binary
        //   Visible to ≥1 satellite → bright green
        //   Masked (earth-occluded / out-of-range) → dim gray-blue
        Color inner_col;
        if (rso_visible_any[i]) {
            inner_col = (Color){0, 255, 80, 255};   // neon green
        } else {
            inner_col = (Color){60, 60, 80, 140};   // dim gray-blue (clearly not magenta)
        }

        Vector3 pos = eci_to_rl(r->pos[0], r->pos[1], r->pos[2]);
        float sz = rso_size + r->size * 0.003f * rso_scale;

        // Override colors when actively being observed: orange
        if (r->num_agents_observing > 0) {
            outer_col = (Color){255, 160, 0, 255};   // bright orange
            inner_col = (Color){255, 200, 50, 255};   // warm yellow-orange
        }

        // Outer wireframe (uncertainty / observed)
        DrawSphereWires(pos, sz * 1.6f, 6, 6, outer_col);
        // Inner wireframe (visibility / observed)
        DrawSphereWires(pos, sz, 5, 5, inner_col);

        // If actively being observed this step, add orange pulse
        if (r->num_agents_observing > 0) {
            DrawSphereWires(pos, sz * 2.2f, 4, 4, (Color){255, 180, 30, 120});
        }
    }

    // ── Target lines: satellite → selected RSO ──
    for (int i = 0; i < env->num_agents; i++) {
        Satellite* s = &env->satellites[i];
        if (env->action_mode == 0 && s->last_action >= 0 && s->last_action < env->num_rso) {
            RSO* target = &env->rsos[s->last_action];
            Vector3 sp = eci_to_rl(s->pos[0], s->pos[1], s->pos[2]);
            Vector3 rp = eci_to_rl(target->pos[0], target->pos[1], target->pos[2]);
            Color link_col = orbit_colors[s->orbit_id % nsc]; link_col.a = 100;
            DrawLine3D(sp, rp, link_col);
        }
    }

    EndMode3D();

    // ── HUD ──
    int hx = 10, hy = 10;
    DrawRectangle(hx, hy, 420, 290, (Color){0, 0, 0, 170});
    Color txt = {200, 220, 240, 255};
    char buf[256];

    snprintf(buf, sizeof(buf), "Step: %d / %d", env->tick, env->max_steps);
    DrawText(buf, hx+10, hy+10, 20, txt);

    float mean_U = 0.0f, max_U = 0.0f;
    int above_thresh = 0, total_obs = 0, total_visible = 0;
    for (int i = 0; i < env->num_rso; i++) {
        float U = surrogate_u_sum_now(&env->rsos[i], env);
        mean_U += U;
        if (U > max_U) max_U = U;
        if (U > env->uncertainty_threshold) above_thresh++;
        if (env->rsos[i].num_agents_observing > 0) total_obs++;
        if (rso_visible_any[i]) total_visible++;
    }
    mean_U /= (float)env->num_rso;
    float frac_above = (float)above_thresh / (float)env->num_rso;

    snprintf(buf, sizeof(buf), "Agents: %d   RSOs: %d   Top-K: %d   Orbits: %d",
             env->num_agents, env->num_rso, env->rso_top_k, env->num_orbits);
    DrawText(buf, hx+10, hy+38, 16, (Color){0, 255, 200, 255});

    snprintf(buf, sizeof(buf), "Mean Uncertainty: %.3f", mean_U);
    DrawText(buf, hx+10, hy+62, 16, (Color){100, 220, 255, 255});

    snprintf(buf, sizeof(buf), "Max  Uncertainty: %.3f", max_U);
    DrawText(buf, hx+10, hy+82, 16, (Color){100, 220, 255, 255});

    snprintf(buf, sizeof(buf), "Above threshold: %d / %d (%.0f%%)",
             above_thresh, env->num_rso, frac_above * 100.0f);
    DrawText(buf, hx+10, hy+104, 16,
             frac_above > 0.5f ? (Color){255, 80, 180, 255} : (Color){0, 255, 180, 255});

    snprintf(buf, sizeof(buf), "Currently observed: %d RSOs", total_obs);
    DrawText(buf, hx+10, hy+124, 16, (Color){255, 160, 0, 255});

    snprintf(buf, sizeof(buf), "Visible to >= 1 sat: %d / %d   Masked: %d",
             total_visible, env->num_rso, env->num_rso - total_visible);
    DrawText(buf, hx+10, hy+146, 16, (Color){0, 255, 80, 255});

    snprintf(buf, sizeof(buf), "Propagation: %s",
             env->propagation_mode >= 1 ? "Keplerian+J2" : "Keplerian");
    DrawText(buf, hx+10, hy+172, 14, (Color){140, 140, 160, 200});

    // Legend
    DrawRectangle(hx+10, hy+194, 10, 10, (Color){0, 255, 220, 255});
    DrawText("Low uncert.", hx+24, hy+193, 12, (Color){160, 180, 200, 200});
    DrawRectangle(hx+120, hy+194, 10, 10, (Color){255, 0, 180, 255});
    DrawText("High uncert.", hx+134, hy+193, 12, (Color){160, 180, 200, 200});
    DrawRectangle(hx+240, hy+194, 10, 10, (Color){0, 255, 80, 255});
    DrawText("Visible", hx+254, hy+193, 12, (Color){160, 180, 200, 200});
    DrawRectangle(hx+310, hy+194, 10, 10, (Color){60, 60, 80, 255});
    DrawText("Masked", hx+324, hy+193, 12, (Color){160, 180, 200, 200});
    DrawRectangle(hx+10, hy+210, 10, 10, (Color){255, 160, 0, 255});
    DrawText("Observed", hx+24, hy+209, 12, (Color){160, 180, 200, 200});

    DrawText("Outer ring = uncertainty | Inner = visibility | Orange = observed",
             hx+10, hy+228, 11, (Color){120, 120, 140, 180});

    DrawText("RMB drag: orbit | Scroll: zoom | WASD: pan | Arrows: tilt | ESC: quit",
             hx+10, hy+246, 13, (Color){100, 100, 120, 180});

    EndDrawing();
}

// ─── Close ───
void c_close(OrbitalEyesCooperative* env) {
    if (env->client) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
    free(env->satellites);
    free(env->rsos);
    free(env->steps_since_observed);
    free(env->top_k_indices);
    free(env->priority_sort_indices);
    free(env->priority_sort_values);
    free(env->topk_out_indices);
    free(env->topk_out_values);
    free(env->base_priority);
    free(env->reachable_flags);
    free(env->reduction_buffer);
    free(env->measurement_quality);
    free(env->external_targets);
}

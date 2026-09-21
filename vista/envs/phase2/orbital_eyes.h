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
 *   [self(12) | other_agents(N-1)*7 | rso_tokens(K)*17]
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

// Observation feature sizes
#define SELF_OBS_SIZE     12
#define OTHER_AGENT_SIZE   7
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

// ─── Per-satellite agent ───
typedef struct {
    float pos[3];           // ECI position (km)
    float vel[3];           // ECI velocity (km/s)
    float sensor_dir[3];    // unit vector, current pointing direction
    float prev_sensor_dir[3]; // sensor_dir before this step's action
    float fov;              // half-cone angle (radians)
    float max_slew_rate;    // rad/s
    int last_action;        // discrete: global RSO index of selected target
    int last_action_idx;    // discrete: top-K action index (0 to K-1)
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
    float n;
} Log;

// ─── Rendering client ───
#define OE_NUM_STARS 512
#define OE_UHIST_LEN 2048
typedef struct {
    Camera3D camera;
    float cam_yaw;
    float cam_pitch;
    float cam_dist;
    // Fixed starfield backdrop (generated once at init)
    Vector3 stars[OE_NUM_STARS];
    unsigned char star_b[OE_NUM_STARS];
    // Mean-uncertainty history ring buffer for the HUD sparkline
    float u_hist[OE_UHIST_LEN];
    int u_hist_head;
    int u_hist_len;
    int last_tick;
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

    // Config: sensor
    float fov_deg;          // sensor FOV half-angle (degrees)
    float max_slew_rate_deg; // max slew rate (deg/s)

    // Config: satellite orbits (multi-orbit constellation)
    int num_orbits;
    OrbitConfig orbits[MAX_ORBITS];

    // Config: RSO population orbital element bounds
    float rso_a_min, rso_a_max;
    float rso_e_min, rso_e_max;
    float rso_inc_min, rso_inc_max;
    float rso_size_min, rso_size_max;

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
    int quality_mode;       // 0=simple (binary), 1=geometric
    float quality_dist_weight;
    float quality_angle_weight;
    float quality_size_weight;
    float max_obs_range;    // km

    // Config: masking
    int enable_earth_mask;
    int enable_range_mask;
    int enable_fov_obs;     // 0=discrete observes only selected RSO, 1=all RSOs in FOV

    // Config: reward
    int reward_mode;        // 0=delta_J (original), 1=level (smooth progressive)
    float alpha_u;
    float alpha_v;
    float alpha_local;
    float alpha_ctrl;       // control cost: penalty per unit of slew
    float reward_scale;     // divisor for r_direct; 0 = auto-compute from problem size
    float reward_baseline;  // per-step growth cost to add before scaling; 0 = auto
    float reward_clip_min;  // lower clamp for per-step r_direct (level reward is
                            // bounded above by 1; this bounds it below so a few
                            // high-uncertainty RSOs cannot drive the episode
                            // return to large negatives). 0 = disabled.
    float uncertainty_threshold;
    float team_spirit;

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
} OrbitalEyes;

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
static inline float surrogate_u_r_now(const RSO* r, const OrbitalEyes* env) {
    if (env->uncertainty_mode == 2) return r->u_r;
    float age = r->age;
    return r->u_r + (env->growth_a_r + 0.5f * env->growth_b_r * age) * age;
}
static inline float surrogate_u_t_now(const RSO* r, const OrbitalEyes* env) {
    if (env->uncertainty_mode == 2) return r->u_t;
    float age = r->age;
    return r->u_t + (env->growth_a_t + 0.5f * env->growth_b_t * age) * age;
}
static inline float surrogate_u_n_now(const RSO* r, const OrbitalEyes* env) {
    if (env->uncertainty_mode == 2) return r->u_n;
    float age = r->age;
    return r->u_n + (env->growth_a_n + 0.5f * env->growth_b_n * age) * age;
}
static inline float surrogate_sigma_vel_now(const RSO* r, const OrbitalEyes* env) {
    if (env->uncertainty_mode == 2) return r->sigma_vel;
    if (env->growth_mode >= 1) {
        return r->sigma_vel + env->k_v * r->age;
    }
    return r->sigma_vel;
}
// Sum form used by priority / reward / log aggregates. Uses cached u_anchor_sum
// so the per-axis anchors are not summed every call.
static inline float surrogate_u_sum_now(const RSO* r, const OrbitalEyes* env) {
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

static inline void ekf_sync_rso_uncertainty(RSO* r, const OrbitalEyes* env) {
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
void init(OrbitalEyes* env) {
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

    // Reduction accumulator
    env->reduction_buffer = (float*)calloc(env->num_rso * 4, sizeof(float));

    // External-target override buffer (benchmark reachable-set access; off by default)
    env->external_targets = (int*)calloc(env->num_agents, sizeof(int));
    env->external_target_active = 0;

    // Compute observation size
    env->obs_size = calc_obs_size(env->num_agents, env->rso_top_k);

    // Observation layout offsets
    env->self_obs_offset = 0;
    env->agents_obs_offset = SELF_OBS_SIZE;
    env->rso_obs_offset = SELF_OBS_SIZE + OTHER_AGENT_SIZE * (env->num_agents - 1);

    // Convert degrees to radians for sensor params
    float fov_rad = env->fov_deg * DEG2RAD;
    float slew_rad = env->max_slew_rate_deg * DEG2RAD;
    for (int i = 0; i < env->num_agents; i++) {
        env->satellites[i].fov = fov_rad;
        env->satellites[i].max_slew_rate = slew_rad;
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
    OrbitalEyes* env, const float* sat_pos, const float* sat_dir,
    const float* rso_pos, float rso_size)
{
    if (env->quality_mode == 0) return 1.0f;  // Simple binary

    float dx = rso_pos[0] - sat_pos[0];
    float dy = rso_pos[1] - sat_pos[1];
    float dz = rso_pos[2] - sat_pos[2];
    float dist = vec3_len(dx, dy, dz);
    if (dist < 1e-6f) return 1.0f;

    // Distance factor: linearly decreasing with distance
    float dist_factor = 1.0f - clampf(dist / env->max_obs_range, 0.0f, 1.0f);

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
static float compute_population_objective(OrbitalEyes* env) {
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

// Forward declaration
static void compute_observations(OrbitalEyes* env);
static void set_was_my_target(OrbitalEyes* env);

// ─── Reset ───
void c_reset(OrbitalEyes* env) {
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

            // Initialise sensor pointing: nadir (toward Earth center)
            s->sensor_dir[0] = -s->pos[0];
            s->sensor_dir[1] = -s->pos[1];
            s->sensor_dir[2] = -s->pos[2];
            vec3_normalize(&s->sensor_dir[0], &s->sensor_dir[1], &s->sensor_dir[2]);

            s->last_action = 0;
            s->last_action_idx = 0;
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
        r->oe_inc = randf(env->rso_inc_min * DEG2RAD, env->rso_inc_max * DEG2RAD);
        r->oe_raan = randf(0.0f, 2.0f * PI);
        r->oe_omega = randf(0.0f, 2.0f * PI);
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

    // Zero rewards and terminals
    memset(env->rewards, 0, env->num_agents * sizeof(float));
    memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));

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
        env->U_ref = sum_U_init / (float)env->num_rso;
    }
    if (env->U_ref < 1e-6f) env->U_ref = 1.0f;  // safety

    // Build initial observations
    compute_observations(env);
    set_was_my_target(env);
}

// ─── Build observations for all agents ───
static void compute_observations(OrbitalEyes* env) {
    float fov_cos = cosf(env->satellites[0].fov);  // all sensors have same FOV

    // ── Pre-compute agent-independent base priority (once per step) ──
    for (int r = 0; r < env->num_rso; r++) {
        RSO* rso = &env->rsos[r];
        float U = surrogate_u_sum_now(rso, env);
        env->base_priority[r] = env->priority_alpha * U
                               + env->priority_beta * rso->age;
        env->priority_sort_indices[r] = r;  // identity permutation
    }

    for (int a = 0; a < env->num_agents; a++) {
        Satellite* sat = &env->satellites[a];
        float* obs = env->observations + a * env->obs_size;

        // Zero all observations
        memset(obs, 0, env->obs_size * sizeof(float));

        // ── Self state (12) ──
        // pos(3), vel(3), sensor_dir(3), prev_action_norm(1), last_delta(1), fov_norm(1)
        int idx = 0;
        obs[idx++] = sat->pos[0] / env->obs_dist_norm;
        obs[idx++] = sat->pos[1] / env->obs_dist_norm;
        obs[idx++] = sat->pos[2] / env->obs_dist_norm;
        obs[idx++] = sat->vel[0] / env->obs_vel_norm;
        obs[idx++] = sat->vel[1] / env->obs_vel_norm;
        obs[idx++] = sat->vel[2] / env->obs_vel_norm;
        obs[idx++] = sat->sensor_dir[0];
        obs[idx++] = sat->sensor_dir[1];
        obs[idx++] = sat->sensor_dir[2];
        // Previous action (normalised)
        if (env->action_mode == 0) {
            obs[idx++] = (float)sat->last_action_idx / (float)(env->rso_top_k > 1 ? env->rso_top_k - 1 : 1);
        } else {
            obs[idx++] = sat->last_delta[0];  // already in [-1,1]
        }
        obs[idx++] = (env->action_mode == 1) ? sat->last_delta[1] : 0.0f;
        obs[idx++] = sat->fov / PI;  // normalised

        // ── Other agents (num_agents-1 × 7) ──
        // rel_pos(3), sensor_dir(3), is_observing(1)
        int agent_idx = 0;
        for (int j = 0; j < env->num_agents; j++) {
            if (j == a) continue;
            Satellite* other = &env->satellites[j];
            int base = env->agents_obs_offset + agent_idx * OTHER_AGENT_SIZE;
            obs[base + 0] = (other->pos[0] - sat->pos[0]) / env->obs_dist_norm;
            obs[base + 1] = (other->pos[1] - sat->pos[1]) / env->obs_dist_norm;
            obs[base + 2] = (other->pos[2] - sat->pos[2]) / env->obs_dist_norm;
            obs[base + 3] = other->sensor_dir[0];
            obs[base + 4] = other->sensor_dir[1];
            obs[base + 5] = other->sensor_dir[2];
            // is_observing: 1 if the other agent observed at least one RSO last step
            obs[base + 6] = (float)other->observed_any;
            agent_idx++;
        }

        // ── RSO tokens (top_k × 17) ──
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
            float max_r2 = env->max_obs_range * env->max_obs_range;
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

        // Select top-K by priority (O(n log K) instead of O(n²) full sort)
        top_k_desc(env->priority_sort_indices, env->priority_sort_values,
                   env->num_rso, env->rso_top_k,
                   env->topk_out_indices, env->topk_out_values);

        int tokens_written = 0;
        // Sensor boresight in spherical coordinates is agent-constant across the
        // top-K tokens; compute it once instead of per candidate.
        float s_el = asinf(clampf(sat->sensor_dir[2], -1.0f, 1.0f));
        float s_az = atan2f(sat->sensor_dir[1], sat->sensor_dir[0]);
        float max_delta = sat->max_slew_rate * env->dt;
        if (max_delta < 1e-8f) max_delta = 1e-8f;
        float inv_max_delta = 1.0f / max_delta;
        for (int k = 0; k < env->rso_top_k; k++) {
            int rso_idx = env->topk_out_indices[k];
            RSO* rso = &env->rsos[rso_idx];

            // Store in top-K mapping
            env->top_k_indices[a * env->rso_top_k + tokens_written] = rso_idx;

            int base = env->rso_obs_offset + tokens_written * RSO_TOKEN_SIZE;

            // Relative position
            obs[base + 0] = (rso->pos[0] - sat->pos[0]) / env->obs_dist_norm;
            obs[base + 1] = (rso->pos[1] - sat->pos[1]) / env->obs_dist_norm;
            obs[base + 2] = (rso->pos[2] - sat->pos[2]) / env->obs_dist_norm;

            // Relative velocity
            obs[base + 3] = (rso->vel[0] - sat->vel[0]) / env->obs_vel_norm;
            obs[base + 4] = (rso->vel[1] - sat->vel[1]) / env->obs_vel_norm;
            obs[base + 5] = (rso->vel[2] - sat->vel[2]) / env->obs_vel_norm;

            // Uncertainty (log-scaled to keep bounded) — evaluate anchors + growth
            obs[base + 6] = logf(1.0f + surrogate_u_r_now(rso, env) / env->obs_u_norm);
            obs[base + 7] = logf(1.0f + surrogate_u_t_now(rso, env) / env->obs_u_norm);
            obs[base + 8] = logf(1.0f + surrogate_u_n_now(rso, env) / env->obs_u_norm);
            obs[base + 9] = logf(1.0f + surrogate_sigma_vel_now(rso, env) / env->obs_u_vel_norm);

            // Age (log-scaled)
            obs[base + 10] = logf(1.0f + rso->age / env->obs_age_norm);

            // Size (normalised)
            obs[base + 11] = rso->size / 10.0f;

            // Visibility: reuse cached reachability, add FOV check
            float dx = rso->pos[0] - sat->pos[0];
            float dy = rso->pos[1] - sat->pos[1];
            float dz = rso->pos[2] - sat->pos[2];
            float dist = vec3_len(dx, dy, dz);
            int visible = env->reachable_flags[rso_idx];  // earth + range already checked

            // FOV check
            if (visible && dist > 1e-6f) {
                float dot = (dx * sat->sensor_dir[0] + dy * sat->sensor_dir[1]
                           + dz * sat->sensor_dir[2]) / dist;
                if (dot < fov_cos) visible = 0;
            }
            obs[base + 12] = (float)visible;

            // Observation quality (when quality_mode>0) or angular proximity to boresight
            float quality = visible ? compute_obs_quality(env, sat->pos, sat->sensor_dir,
                                                          rso->pos, rso->size) : 0.0f;
            if (env->quality_mode == 0 && dist > 1e-6f) {
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
            // For step 0: last_action = 0, so compare against previous top_k mapping
            // We compare the global RSO index against what was at last_action in prev top_k
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
                // Current sensor direction in spherical
                float s_el = asinf(clampf(sat->sensor_dir[2], -1.0f, 1.0f));
                float s_az = atan2f(sat->sensor_dir[1], sat->sensor_dir[0]);

                // RSO direction from satellite in spherical
                // (reuse dx, dy, dz already computed above)
                float r_dir_x = dx, r_dir_y = dy, r_dir_z = dz;
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
static void set_was_my_target(OrbitalEyes* env) {
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
void c_step(OrbitalEyes* env) {
    PROF_START(total_step);
    env->tick++;
    env->epoch_time += env->dt;

    // Zero rewards
    memset(env->rewards, 0, env->num_agents * sizeof(float));
    memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));

    // ── 1. Update satellite and RSO positions from orbital mechanics ──
    PROF_START(propagation);
    if (env->propagation_mode == 0) {
        // No J2 secular drift: RAAN/omega constant, so the perifocal→ECI basis
        // is an episode-invariant cached at reset. Fast path.
        for (int i = 0; i < env->num_agents; i++) {
            Satellite* s = &env->satellites[i];
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
            action = clampi(action, 0, env->rso_top_k - 1);
            sat->last_action_idx = action;  // store top-K index for observation encoding

            // Default: decode the discrete action into a global RSO via the
            // per-agent top-K map. When the benchmark enables reachable-set
            // access for classical baselines, a global target index is injected
            // directly (bypassing the top-K shortlist). Never active for the NN.
            int global_rso;
            if (env->external_target_active) {
                global_rso = env->external_targets[a];
            } else {
                global_rso = env->top_k_indices[a * env->rso_top_k + action];
            }
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
    PROF_END(env, actions);

    // ── 3. Determine observations and accumulate uncertainty reductions ──
    PROF_START(observation_det);
    // Zero the reduction buffer
    memset(env->reduction_buffer, 0, env->num_rso * 4 * sizeof(float));

    // Reset num_agents_observing
    for (int i = 0; i < env->num_rso; i++) {
        env->rsos[i].num_agents_observing = 0;
    }

    float fov_cos = cosf(env->satellites[0].fov);
    int total_observations = 0;

    for (int a = 0; a < env->num_agents; a++) {
        Satellite* sat = &env->satellites[a];
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
                    if (dist > env->max_obs_range) visible = 0;
                }

                if (visible) {
                    float quality = compute_obs_quality(env, sat->pos, sat->sensor_dir,
                                                       rso->pos, rso->size);
                    // Accumulate reduction
                    env->reduction_buffer[global_rso * 4 + 0] += env->alpha_r * quality;
                    env->reduction_buffer[global_rso * 4 + 1] += env->alpha_t * quality;
                    env->reduction_buffer[global_rso * 4 + 2] += env->alpha_n * quality;
                    env->reduction_buffer[global_rso * 4 + 3] += env->alpha_vel * quality;

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
                    if (dist > env->max_obs_range) visible = 0;
                }

                if (visible) {
                    float quality = compute_obs_quality(env, sat->pos, sat->sensor_dir,
                                                       rso->pos, rso->size);
                    env->reduction_buffer[r * 4 + 0] += env->alpha_r * quality;
                    env->reduction_buffer[r * 4 + 1] += env->alpha_t * quality;
                    env->reduction_buffer[r * 4 + 2] += env->alpha_n * quality;
                    env->reduction_buffer[r * 4 + 3] += env->alpha_vel * quality;

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
                float q_eff = red_r / fmaxf(env->alpha_r, 1e-12f);
                if (q_eff < 0.05f) q_eff = 0.05f;
                ekf_update_position(r->P, env->ekf_meas_sigma / q_eff);
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
                // Effective summed quality across observers (red_r = Σ α_r·q_i).
                float q_eff = red_r / fmaxf(env->alpha_r, 1e-12f);
                if (q_eff < 0.05f) q_eff = 0.05f;            // floor
                float meas_sigma = env->ekf_meas_sigma / q_eff;

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
                    ekf_update_position(r->P, meas_sigma);
                    r->ekf_anchor_fresh = 1;
                } else {
                    // ── Intra-episode gap: P is small (last update gave ~meas_sigma) ──
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
                    ekf_update_position(r->P, meas_sigma);
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
                    ekf_update_position(r->P, meas_sigma);
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
    } else {
        // ── Mode 0: original delta-J reward ──
        float J_curr = compute_population_objective(env);
        r_direct = (env->J_prev - J_curr + env->reward_baseline)
                 / env->reward_scale;
        env->J_prev = J_curr;
    }

    // Clamp the per-step reward below to keep the episode return bounded. The
    // level reward is already bounded above by 1; without a floor a handful of
    // high-uncertainty (e.g. long-unobserved) RSOs can drive mean_U ≫ U_ref and
    // produce large-magnitude negative rewards that destabilise the value
    // function early in training. 0 = disabled (legacy behaviour).
    if (env->reward_clip_min < 0.0f && r_direct < env->reward_clip_min) {
        r_direct = env->reward_clip_min;
    }

    // Distribute reward: all agents get the same team reward
    for (int a = 0; a < env->num_agents; a++) {
        env->rewards[a] = r_direct;
    }

    // Control cost: penalize sensor slew to discourage rapid scanning
    // slew_frac = angular distance moved / max possible angle (max_slew_rate * dt)
    // cost = alpha_ctrl * slew_frac^2  (quadratic: small corrections cheap, big slews expensive)
    if (env->alpha_ctrl > 0.0f) {
        float max_angle = env->satellites[0].max_slew_rate * env->dt;
        if (max_angle > 1e-8f) {
            for (int a = 0; a < env->num_agents; a++) {
                Satellite* sat = &env->satellites[a];
                float dot = sat->prev_sensor_dir[0] * sat->sensor_dir[0]
                          + sat->prev_sensor_dir[1] * sat->sensor_dir[1]
                          + sat->prev_sensor_dir[2] * sat->sensor_dir[2];
                dot = clampf(dot, -1.0f, 1.0f);
                float angle = acosf(dot);  // actual angular change
                float frac = clampf(angle / max_angle, 0.0f, 1.0f);
                env->rewards[a] -= env->alpha_ctrl * frac * frac;
            }
        }
    }

    // Accumulate episode return
    for (int a = 0; a < env->num_agents; a++) {
        env->log.episode_return += env->rewards[a];
    }
    env->log.total_observations += (float)total_observations;

    PROF_END(env, reward);

    // ── 6. Check termination ──
    int done = (env->tick >= env->max_steps);
    if (done) {
        for (int a = 0; a < env->num_agents; a++) {
            env->terminals[a] = 1;
        }

        // Finalise log
        env->log.episode_length += (float)env->tick;
        env->log.episode_return /= (float)env->num_agents;  // per-agent average

        // Compute final uncertainty stats
        float sum_U = 0.0f, max_U = 0.0f;
        int above_threshold = 0;
        for (int i = 0; i < env->num_rso; i++) {
            float U = surrogate_u_sum_now(&env->rsos[i], env);
            sum_U += U;
            if (U > max_U) max_U = U;
            if (U > env->uncertainty_threshold) above_threshold++;
        }
        env->log.mean_uncertainty += sum_U / (float)env->num_rso;
        env->log.max_uncertainty += max_U;
        env->log.fraction_above_threshold += (float)above_threshold / (float)env->num_rso;
        env->log.mean_age += mean_age / (float)env->num_rso;
        env->log.effective_tracked += (float)env->num_rso * (1.0f - (float)above_threshold / (float)env->num_rso);

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

// Diorama mode: all ECI km positions are scaled so Earth radius = 1.0 unit.
// This keeps everything inside Raylib's default clip planes (near=0.01, far=1000).
// Satellites/RSOs are drawn with exaggerated sizes for visibility.
#define RENDER_SCALE (1.0f / EARTH_RADIUS_KM)
// ECI(x,y,z) → Raylib Y-up (x, z, -y) scaled to diorama. We negate y→z so
// the ECI +x/+y plane maps to Raylib +x/−z, keeping right-hand orientation.
static inline Vector3 eci_to_rl(float ex, float ey, float ez) {
    return (Vector3){ ex * RENDER_SCALE, ez * RENDER_SCALE, -ey * RENDER_SCALE };
}

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

// Translucent filled cone (triangle fan apex→ring). Both winding orders are
// emitted so the fill is visible regardless of backface culling / view side.
static void draw_sensor_cone_fill(Vector3 apex, float dir_x, float dir_y, float dir_z,
                                   float half_angle, float length, int segments, Color col) {
    float fx = dir_x, fy = dir_y, fz = dir_z;
    float ux, uy, uz;
    if (fabsf(fx) < 0.9f) { ux = 1; uy = 0; uz = 0; }
    else                   { ux = 0; uy = 1; uz = 0; }
    float rx, ry, rz;
    vec3_cross(fx, fy, fz, ux, uy, uz, &rx, &ry, &rz);
    float rlen = vec3_len(rx, ry, rz);
    if (rlen < 1e-12f) return;
    rx /= rlen; ry /= rlen; rz /= rlen;
    vec3_cross(rx, ry, rz, fx, fy, fz, &ux, &uy, &uz);

    float cone_radius = length * tanf(half_angle);
    Vector3 tip = { apex.x + fx * length, apex.y + fy * length, apex.z + fz * length };

    Vector3 prev = {0};
    for (int i = 0; i <= segments; i++) {
        float theta = 2.0f * PI * (float)i / (float)segments;
        float ct = cosf(theta), st = sinf(theta);
        Vector3 p = {
            tip.x + cone_radius * (ct * rx + st * ux),
            tip.y + cone_radius * (ct * ry + st * uy),
            tip.z + cone_radius * (ct * rz + st * uz)
        };
        if (i > 0) {
            DrawTriangle3D(apex, prev, p, col);
            DrawTriangle3D(apex, p, prev, col);
        }
        prev = p;
    }
}

// Faint full orbit ellipse from Keplerian elements (RAAN/omega in radians,
// already including any J2 secular drift at the current epoch).
static void draw_orbit_ring(float a, float e, float raan, float omega,
                             float cos_inc, float sin_inc, int segments, Color col) {
    float Px, Py, Pz, Qx, Qy, Qz;
    precompute_rotation_columns(raan, omega, cos_inc, sin_inc,
                                &Px, &Py, &Pz, &Qx, &Qy, &Qz);
    float p_semi = a * (1.0f - e * e);
    Vector3 prev = {0};
    for (int i = 0; i <= segments; i++) {
        float nu = 2.0f * PI * (float)i / (float)segments;
        float cn = cosf(nu), sn = sinf(nu);
        float r = p_semi / (1.0f + e * cn);
        float x = r * cn, y = r * sn;
        Vector3 pt = eci_to_rl(Px * x + Qx * y, Py * x + Qy * y, Pz * x + Qz * y);
        if (i > 0) DrawLine3D(prev, pt, col);
        prev = pt;
    }
}

// ─── Render ───
void c_render(OrbitalEyes* env) {
    if (!env->client) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(1920, 1080, "VISTA — SSA Sensor Scheduling");
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
        // Starfield: fixed points on a far sphere (inside clip planes), with a
        // deterministic LCG so the sky is identical every run.
        unsigned int seed = 0x9E3779B9u;
        for (int i = 0; i < OE_NUM_STARS; i++) {
            seed = seed * 1664525u + 1013904223u;
            float u = (float)(seed >> 8) / 16777216.0f;
            seed = seed * 1664525u + 1013904223u;
            float v = (float)(seed >> 8) / 16777216.0f;
            seed = seed * 1664525u + 1013904223u;
            float w = (float)(seed >> 8) / 16777216.0f;
            float z = 2.0f * u - 1.0f;
            float th = 2.0f * PI * v;
            float rr = sqrtf(fmaxf(0.0f, 1.0f - z * z));
            float R = 80.0f;
            env->client->stars[i] = (Vector3){R * rr * cosf(th), R * z, R * rr * sinf(th)};
            env->client->star_b[i] = (unsigned char)(60 + (int)(w * 160.0f));
        }
        env->client->last_tick = -1;
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

    // Render-only sideways offset: shift the whole view so the orbit pivot
    // projects at ~65% of screen width, clearing the left HUD column.
    Camera3D view = cl->camera;
    {
        float fx = view.target.x - view.position.x;
        float fy = view.target.y - view.position.y;
        float fz = view.target.z - view.position.z;
        float rx, ry, rz;
        vec3_cross(fx, fy, fz, 0.0f, 1.0f, 0.0f, &rx, &ry, &rz);
        float rl = vec3_len(rx, ry, rz);
        if (rl > 1e-6f) {
            float aspect = (float)GetScreenWidth() / (float)GetScreenHeight();
            float s = 0.15f * 2.0f * tanf(view.fovy * 0.5f * DEG2RAD) * aspect * cl->cam_dist;
            rx /= rl; ry /= rl; rz /= rl;
            view.position.x -= rx * s; view.position.y -= ry * s; view.position.z -= rz * s;
            view.target.x   -= rx * s; view.target.y   -= ry * s; view.target.z   -= rz * s;
        }
    }

    BeginDrawing();
    ClearBackground((Color){4, 6, 14, 255});

    BeginMode3D(view);

    // ── Starfield backdrop ──
    for (int i = 0; i < OE_NUM_STARS; i++) {
        unsigned char b = cl->star_b[i];
        DrawPoint3D(cl->stars[i], (Color){b, b, (unsigned char)fminf(255, b + 30), 255});
    }

    // ── Earth: solid dark globe (real depth occlusion) + neon wireframe ──
    DrawSphereEx((Vector3){0, 0, 0}, 0.985f, 32, 32, (Color){5, 9, 22, 255});
    DrawSphereWires((Vector3){0, 0, 0}, 1.0f, 24, 24, (Color){20, 60, 130, 200});
    DrawSphereWires((Vector3){0, 0, 0}, 1.003f, 16, 16, (Color){30, 80, 160, 100});
    // Equator ring
    DrawCircle3D((Vector3){0, 0, 0}, 1.005f, (Vector3){1, 0, 0}, 90.0f,
                 (Color){40, 100, 180, 220});
    // Atmosphere rim: camera-facing glow rings just above the surface
    {
        Vector3 cp = view.position;
        float cp_len = sqrtf(cp.x*cp.x + cp.y*cp.y + cp.z*cp.z);
        if (cp_len > 1e-6f) {
            // DrawCircle3D rotates the XY-plane circle; align its normal (+z
            // after rotation) with the camera direction.
            Vector3 n = {cp.x / cp_len, cp.y / cp_len, cp.z / cp_len};
            Vector3 axis = {-n.y, n.x, 0.0f};  // z-hat × n
            float axis_len = sqrtf(axis.x*axis.x + axis.y*axis.y);
            float ang = acosf(clampf(n.z, -1.0f, 1.0f)) * RAD2DEG;
            if (axis_len < 1e-6f) { axis = (Vector3){1, 0, 0}; ang = (n.z > 0) ? 0.0f : 180.0f; }
            DrawCircle3D((Vector3){0, 0, 0}, 1.025f, axis, ang, (Color){60, 160, 255, 90});
            DrawCircle3D((Vector3){0, 0, 0}, 1.045f, axis, ang, (Color){40, 120, 220, 50});
            DrawCircle3D((Vector3){0, 0, 0}, 1.070f, axis, ang, (Color){25, 80, 180, 25});
        }
    }

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
    float sat_size  = 0.018f;  // satellite marker radius
    float rso_scale = (env->render_rso_scale > 0.0f) ? env->render_rso_scale : 1.0f;
    // Density adaptation: shrink RSO markers as the catalogue grows so dense
    // populations (e.g. 20k) read as a debris field instead of a solid shell.
    // Reference 2000 RSOs = baseline size; never enlarged below that.
    float density_f = sqrtf(2000.0f / (float)env->num_rso);
    density_f = clampf(density_f, 0.2f, 1.0f);
    float rso_size  = 0.008f * rso_scale * density_f;  // RSO marker base radius
    // Very large catalogues: draw RSOs as single points instead of wire spheres
    int rso_point_mode = (env->num_rso > 5000);
    float cone_len  = env->enable_range_mask
                        ? env->max_obs_range * RENDER_SCALE
                        : 0.5f;  // sensor FOV cone length = actual observation range

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
            if (env->enable_range_mask && dist > env->max_obs_range) continue;
            rso_visible_any[i] = 1;
        }
    }

    // ── Orbit rings: one faint neon ellipse per orbital plane ──
    {
        int drawn[64];
        int n_drawn = 0;
        for (int i = 0; i < env->num_agents; i++) {
            Satellite* s = &env->satellites[i];
            int seen = 0;
            for (int k = 0; k < n_drawn; k++) if (drawn[k] == s->orbit_id) { seen = 1; break; }
            if (seen || n_drawn >= 64) continue;
            drawn[n_drawn++] = s->orbit_id;
            Color rc = orbit_colors[s->orbit_id % nsc]; rc.a = 80;
            // Include J2 secular drift so the ring tracks the actual plane
            float raan  = s->oe_raan  + s->oe_raan_dot  * env->epoch_time;
            float omega = s->oe_omega + s->oe_omega_dot * env->epoch_time;
            draw_orbit_ring(s->oe_a, s->oe_e, raan, omega,
                            s->oe_cos_inc, s->oe_sin_inc, 96, rc);
        }
    }

    // ── Satellites ──
    for (int i = 0; i < env->num_agents; i++) {
        Satellite* s = &env->satellites[i];
        Vector3 pos = eci_to_rl(s->pos[0], s->pos[1], s->pos[2]);
        Color col = orbit_colors[s->orbit_id % nsc];

        // Satellite body
        DrawSphereWires(pos, sat_size, 8, 8, col);
        // Glow halo
        Color glow = col; glow.a = 60;
        DrawSphereWires(pos, sat_size * 1.6f, 6, 6, glow);
        // Velocity tick (direction of motion along the orbit)
        {
            float vlen = vec3_len(s->vel[0], s->vel[1], s->vel[2]);
            if (vlen > 1e-6f) {
                Vector3 vd = eci_to_rl(s->vel[0], s->vel[1], s->vel[2]);
                float vl = sqrtf(vd.x*vd.x + vd.y*vd.y + vd.z*vd.z);
                if (vl > 1e-9f) {
                    Vector3 vtip = { pos.x + vd.x / vl * sat_size * 3.0f,
                                     pos.y + vd.y / vl * sat_size * 3.0f,
                                     pos.z + vd.z / vl * sat_size * 3.0f };
                    Color vcol = col; vcol.a = 140;
                    DrawLine3D(pos, vtip, vcol);
                }
            }
        }

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

        // FOV cone (wireframe): soft outline, the translucent fill carries it
        Color cone_col = col; cone_col.a = 90;
        draw_sensor_cone(pos, bore_dir.x, bore_dir.y, bore_dir.z,
                         s->fov, cone_len, 32, cone_col);
    }

    // ── RSOs: dual-shell visualization ──
    // Outer shell = uncertainty level (cyan=low → magenta=high, continuous ramp)
    // Inner shell = visibility status (green=visible to ≥1 sat, gray=masked)
    // Camera-space basis for point-mode cross markers (billboarded dots).
    // NOTE: raylib's DrawPoint3D is a 0.1-unit line — far too long here.
    Vector3 cam_right, cam_up;
    {
        Vector3 fwd = { view.target.x - view.position.x,
                        view.target.y - view.position.y,
                        view.target.z - view.position.z };
        float fl = sqrtf(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
        fwd.x /= fl; fwd.y /= fl; fwd.z /= fl;
        cam_right = (Vector3){ fwd.z, 0.0f, -fwd.x };  // fwd × world-up (y)
        float rl = sqrtf(cam_right.x*cam_right.x + cam_right.z*cam_right.z);
        if (rl < 1e-6f) { cam_right = (Vector3){1, 0, 0}; rl = 1.0f; }
        cam_right.x /= rl; cam_right.z /= rl;
        cam_up = (Vector3){ cam_right.y*fwd.z - cam_right.z*fwd.y,
                            cam_right.z*fwd.x - cam_right.x*fwd.z,
                            cam_right.x*fwd.y - cam_right.y*fwd.x };
    }
    for (int i = 0; i < env->num_rso; i++) {
        RSO* r = &env->rsos[i];
        float U = surrogate_u_sum_now(r, env);

        // Outer shell: continuous cyan→magenta ramp, saturating at threshold
        float f = clampf(U / env->uncertainty_threshold, 0.0f, 1.0f);
        f = f * f;  // bias toward cyan so mid-range doesn't wash out pink
        Color outer_col = {
            (unsigned char)(255.0f * f),
            (unsigned char)(255.0f * (1.0f - f)),
            (unsigned char)(220.0f - 40.0f * f),
            200
        };

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
        float sz = rso_size + r->size * 0.003f * rso_scale * density_f;

        // Override colors when actively being observed: orange
        if (r->num_agents_observing > 0) {
            outer_col = (Color){255, 160, 0, 255};   // bright orange
            inner_col = (Color){255, 200, 50, 255};   // warm yellow-orange
        }

        // Dense-catalogue mode: one tiny cross marker per RSO (uncertainty ramp
        // color, dimmed when masked); spheres + pulse only for observed objects.
        if (rso_point_mode) {
            if (r->num_agents_observing > 0) {
                DrawSphereWires(pos, sz * 1.6f, 5, 5, outer_col);
                DrawSphereWires(pos, sz * 2.4f, 4, 4, (Color){255, 180, 30, 120});
            } else {
                Color pc = outer_col;
                pc.a = rso_visible_any[i] ? 255 : 90;
                float h = sz;
                DrawLine3D((Vector3){pos.x - cam_right.x*h, pos.y - cam_right.y*h, pos.z - cam_right.z*h},
                           (Vector3){pos.x + cam_right.x*h, pos.y + cam_right.y*h, pos.z + cam_right.z*h}, pc);
                DrawLine3D((Vector3){pos.x - cam_up.x*h, pos.y - cam_up.y*h, pos.z - cam_up.z*h},
                           (Vector3){pos.x + cam_up.x*h, pos.y + cam_up.y*h, pos.z + cam_up.z*h}, pc);
            }
            continue;
        }

        // Motion trail: fading streak opposite the velocity vector
        {
            Vector3 vd = eci_to_rl(r->vel[0], r->vel[1], r->vel[2]);
            float vl = sqrtf(vd.x*vd.x + vd.y*vd.y + vd.z*vd.z);
            if (vl > 1e-9f) {
                // Hotter objects get longer tails; whole trail shrinks with density
                float trail_len = (0.06f + 0.06f * f) * density_f;
                Color t1 = outer_col; t1.a = 90;
                Color t2 = outer_col; t2.a = 25;
                Vector3 mid = { pos.x - vd.x / vl * trail_len * 0.5f,
                                pos.y - vd.y / vl * trail_len * 0.5f,
                                pos.z - vd.z / vl * trail_len * 0.5f };
                Vector3 tail = { pos.x - vd.x / vl * trail_len,
                                 pos.y - vd.y / vl * trail_len,
                                 pos.z - vd.z / vl * trail_len };
                DrawLine3D(pos, mid, t1);
                DrawLine3D(mid, tail, t2);
            }
        }

        // Outer wireframe (uncertainty / observed) — swells slightly with U
        DrawSphereWires(pos, sz * (1.6f + 0.8f * f), 6, 6, outer_col);
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
            if (s->observed_any && target->num_agents_observing > 0) {
                // Observation succeeded this step: hot photon-link
                DrawLine3D(sp, rp, (Color){255, 200, 60, 230});
            } else {
                Color link_col = orbit_colors[s->orbit_id % nsc]; link_col.a = 100;
                DrawLine3D(sp, rp, link_col);
            }
        }
    }

    // ── Translucent FOV cone fills (drawn last so they blend over the scene) ──
    for (int i = 0; i < env->num_agents; i++) {
        Satellite* s = &env->satellites[i];
        Vector3 pos = eci_to_rl(s->pos[0], s->pos[1], s->pos[2]);
        Vector3 bore_dir = eci_to_rl(s->sensor_dir[0], s->sensor_dir[1], s->sensor_dir[2]);
        float bd_len = sqrtf(bore_dir.x*bore_dir.x + bore_dir.y*bore_dir.y + bore_dir.z*bore_dir.z);
        if (bd_len < 1e-9f) continue;
        bore_dir.x /= bd_len; bore_dir.y /= bd_len; bore_dir.z /= bd_len;
        Color fill = orbit_colors[s->orbit_id % nsc];
        fill.a = s->observed_any ? 48 : 24;  // brighter beam on a successful obs
        draw_sensor_cone_fill(pos, bore_dir.x, bore_dir.y, bore_dir.z,
                              s->fov, cone_len, 32, fill);
    }

    EndMode3D();

    // ── Satellite labels (screen-space, only when in front of the camera) ──
    {
        Vector3 cam_fwd = {
            view.target.x - view.position.x,
            view.target.y - view.position.y,
            view.target.z - view.position.z
        };
        for (int i = 0; i < env->num_agents; i++) {
            Satellite* s = &env->satellites[i];
            Vector3 pos = eci_to_rl(s->pos[0], s->pos[1], s->pos[2]);
            Vector3 rel = { pos.x - view.position.x,
                            pos.y - view.position.y,
                            pos.z - view.position.z };
            if (rel.x * cam_fwd.x + rel.y * cam_fwd.y + rel.z * cam_fwd.z <= 0.0f)
                continue;  // behind camera
            Vector2 sp = GetWorldToScreen(pos, view);
            if (sp.x < -50 || sp.x > GetScreenWidth() + 50 ||
                sp.y < -50 || sp.y > GetScreenHeight() + 50) continue;
            Color lc = orbit_colors[s->orbit_id % nsc]; lc.a = 220;
            char lbl[16];
            snprintf(lbl, sizeof(lbl), "S%d", i);
            DrawText(lbl, (int)sp.x + 10, (int)sp.y - 18, 14, lc);
        }
    }

    // ── HUD ── left 30% column: stats (top half) + U(t) (bottom half)
    int sw = GetScreenWidth(), sh = GetScreenHeight();
    Color txt = {200, 220, 240, 255};
    Color panel_bg   = {0, 0, 0, 170};
    Color panel_edge = {0, 255, 200, 60};
    Color label_col  = {140, 160, 180, 220};
    char buf[256];

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

    int m = 16;                         // outer margin
    int col_w = (int)(0.30f * sw);      // HUD column width
    int pw_panel = col_w - 2 * m;

    // ── Top-left half: mission stats ──
    {
        int hx = m, hy = m, hh = sh / 2 - 2 * m;
        DrawRectangle(hx, hy, pw_panel, hh, panel_bg);
        DrawRectangleLines(hx, hy, pw_panel, hh, panel_edge);
        int x = hx + 22, y = hy + 20, row = 44;

        DrawText("MISSION", x, y, 26, label_col); y += row;

        snprintf(buf, sizeof(buf), "Sensors  %d    Orbits  %d", env->num_agents, env->num_orbits);
        DrawText(buf, x, y, 26, (Color){0, 255, 200, 255}); y += row;
        snprintf(buf, sizeof(buf), "RSOs  %d    Top-K  %d", env->num_rso, env->rso_top_k);
        DrawText(buf, x, y, 26, (Color){0, 255, 200, 255}); y += row + 10;

        snprintf(buf, sizeof(buf), "Mean U      %.2f km", mean_U);
        DrawText(buf, x, y, 26, (Color){100, 220, 255, 255}); y += row;
        snprintf(buf, sizeof(buf), "Max U       %.2f km", max_U);
        DrawText(buf, x, y, 26, (Color){100, 220, 255, 255}); y += row;
        snprintf(buf, sizeof(buf), "Above thr.  %d  (%.0f%%)", above_thresh, frac_above * 100.0f);
        DrawText(buf, x, y, 26,
                 frac_above > 0.5f ? (Color){255, 80, 180, 255} : (Color){0, 255, 180, 255}); y += row;
        snprintf(buf, sizeof(buf), "Observed    %d RSOs", total_obs);
        DrawText(buf, x, y, 26, (Color){255, 160, 0, 255}); y += row;
        snprintf(buf, sizeof(buf), "Visible     %d / %d", total_visible, env->num_rso);
        DrawText(buf, x, y, 26, (Color){0, 255, 80, 255}); y += row + 10;

        // Legend
        DrawRectangle(x, y + 4, 18, 18, (Color){0, 255, 220, 255});
        DrawText("Low U", x + 26, y, 22, label_col);
        DrawRectangle(x + 150, y + 4, 18, 18, (Color){255, 0, 180, 255});
        DrawText("High U", x + 176, y, 22, label_col);
        DrawRectangle(x + 310, y + 4, 18, 18, (Color){255, 160, 0, 255});
        DrawText("Observed", x + 336, y, 22, label_col); y += row;

        snprintf(buf, sizeof(buf), "Propagation  %s",
                 env->propagation_mode >= 1 ? "Keplerian + J2" : "Keplerian");
        DrawText(buf, x, y, 20, (Color){140, 140, 160, 200});
        DrawText("RMB orbit | Scroll zoom | WASD pan | ESC quit",
                 x, hy + hh - 34, 18, (Color){100, 100, 120, 180});
    }

    // ── Bottom-left half: U(t) evolution ──
    // Record one sample per env step (render may run more often than step)
    if (env->tick != cl->last_tick) {
        cl->last_tick = env->tick;
        if (env->tick == 0) { cl->u_hist_len = 0; cl->u_hist_head = 0; }
        cl->u_hist[cl->u_hist_head] = mean_U;
        cl->u_hist_head = (cl->u_hist_head + 1) % OE_UHIST_LEN;
        if (cl->u_hist_len < OE_UHIST_LEN) cl->u_hist_len++;
    }
    if (cl->u_hist_len >= 2) {
        int gx = m, gy = sh / 2 + m, gh = sh / 2 - 2 * m;
        DrawRectangle(gx, gy, pw_panel, gh, panel_bg);
        DrawRectangleLines(gx, gy, pw_panel, gh, panel_edge);
        DrawText("MEAN UNCERTAINTY  U(t)", gx + 22, gy + 18, 26, label_col);
        snprintf(buf, sizeof(buf), "%.2f km", mean_U);
        DrawText(buf, gx + pw_panel - 22 - MeasureText(buf, 40), gy + 54, 40,
                 (Color){100, 220, 255, 255});

        float u_max = env->uncertainty_threshold;
        for (int k = 0; k < cl->u_hist_len; k++) {
            int idx = (cl->u_hist_head - cl->u_hist_len + k + OE_UHIST_LEN) % OE_UHIST_LEN;
            if (cl->u_hist[idx] > u_max) u_max = cl->u_hist[idx];
        }
        int px = gx + 22, py = gy + 110, pw = pw_panel - 44, ph = gh - 160;
        // Scale history over the episode length so the curve fills as time passes
        int span = env->max_steps > 0 ? env->max_steps : OE_UHIST_LEN;
        if (span > OE_UHIST_LEN) span = OE_UHIST_LEN;
        if (span < cl->u_hist_len) span = cl->u_hist_len;
        // Plot frame + threshold reference
        DrawRectangleLines(px, py, pw, ph, (Color){60, 80, 100, 120});
        int ty = py + ph - (int)((env->uncertainty_threshold / u_max) * (float)ph);
        DrawLine(px, ty, px + pw, ty, (Color){255, 0, 180, 110});
        DrawText("threshold", px + pw - MeasureText("threshold", 20) - 6,
                 ty - 26 > py ? ty - 26 : ty + 6, 20, (Color){255, 0, 180, 160});
        // History polyline, colored by level vs threshold
        int prev_x = 0, prev_y = 0;
        for (int k = 0; k < cl->u_hist_len; k++) {
            int idx = (cl->u_hist_head - cl->u_hist_len + k + OE_UHIST_LEN) % OE_UHIST_LEN;
            float v = cl->u_hist[idx];
            int x = px + (int)((float)k / (float)(span - 1) * (float)pw);
            int y = py + ph - (int)(clampf(v / u_max, 0.0f, 1.0f) * (float)ph);
            if (k > 0) {
                Color lc = (v > env->uncertainty_threshold)
                    ? (Color){255, 60, 200, 255} : (Color){0, 255, 200, 255};
                DrawLine(prev_x, prev_y, x, y, lc);
                DrawLine(prev_x, prev_y + 1, x, y + 1, lc);  // 2px for readability
            }
            prev_x = x; prev_y = y;
        }
        // Axis extremes
        snprintf(buf, sizeof(buf), "%.0f km", u_max);
        DrawText(buf, px + 6, py + 6, 20, label_col);
        DrawText("0", px + 6, py + ph - 26, 20, label_col);
        int tmin = (int)(env->max_steps * env->dt) / 60;
        snprintf(buf, sizeof(buf), "t = %d min", tmin);
        DrawText(buf, px + pw - MeasureText(buf, 20) - 6, py + ph - 26, 20, label_col);
    }

    // ── Top-right: step + sim clock ──
    {
        int tsec = (int)env->epoch_time;
        int days = tsec / 86400, hrs = (tsec / 3600) % 24;
        int mins = (tsec / 60) % 60, secs = tsec % 60;
        int cw = 460, ch = 116, cx = sw - cw - m, cy = m;
        DrawRectangle(cx, cy, cw, ch, panel_bg);
        DrawRectangleLines(cx, cy, cw, ch, panel_edge);
        snprintf(buf, sizeof(buf), "STEP  %d / %d", env->tick, env->max_steps);
        DrawText(buf, cx + cw - 22 - MeasureText(buf, 26), cy + 16, 26, txt);
        snprintf(buf, sizeof(buf), "T+ %dd %02d:%02d:%02d", days, hrs, mins, secs);
        DrawText(buf, cx + cw - 22 - MeasureText(buf, 36), cy + 58, 36, (Color){0, 255, 200, 255});
    }

    EndDrawing();
}

// ─── Close ───
void c_close(OrbitalEyes* env) {
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
    free(env->external_targets);
}

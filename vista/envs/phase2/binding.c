#include <Python.h>
#include "orbital_eyes.h"

#define Env OrbitalEyes

// Forward-declare custom helpers (defined after env_binding.h provides VecEnv)
static PyObject* vec_profile(PyObject* self, PyObject* args);
static PyObject* vec_rso_state(PyObject* self, PyObject* args);
static PyObject* vec_agent_activity(PyObject* self, PyObject* args);
static PyObject* vec_rso_axes(PyObject* self, PyObject* args);
static PyObject* vec_rso_positions(PyObject* self, PyObject* args);
static PyObject* vec_feasible(PyObject* self, PyObject* args);
static PyObject* vec_future_reach(PyObject* self, PyObject* args);
static PyObject* vec_future_positions(PyObject* self, PyObject* args);
static PyObject* vec_set_external_targets(PyObject* self, PyObject* args);
static PyObject* vec_clear_external_targets(PyObject* self, PyObject* args);
#define MY_METHODS \
    {"vec_profile", vec_profile, METH_VARARGS, "Get profiling data from all envs"}, \
    {"vec_rso_state", vec_rso_state, METH_VARARGS, "Write per-RSO total uncertainty and age (num_envs x num_rso) into provided float32 arrays"}, \
    {"vec_agent_activity", vec_agent_activity, METH_VARARGS, "Write each agent's resolved global RSO target and last-step observation success into int32/uint8 arrays"}, \
    {"vec_rso_axes", vec_rso_axes, METH_VARARGS, "Write per-RSO per-axis uncertainty (u_r,u_t,u_n) and age (num_envs x num_rso) into 4 provided float32 arrays"}, \
    {"vec_rso_positions", vec_rso_positions, METH_VARARGS, "Write per-RSO ECI position (num_envs x num_rso x 3) into provided float32 array"}, \
    {"vec_feasible", vec_feasible, METH_VARARGS, "Write per-(agent,RSO) reachability flag (num_envs x num_agents x num_rso) into provided uint8 array, using the env earth/range masks"}, \
    {"vec_future_reach", vec_future_reach, METH_VARARGS, "Write per-(epoch,agent,RSO) reachability (num_envs x H x num_agents x num_rso uint8) at future epochs t=epoch_time+k*dt (k=0..H-1), propagating sats+RSOs with the env Keplerian(+J2) model; k=0 reproduces the current geometry. Read-only."}, \
    {"vec_future_positions", vec_future_positions, METH_VARARGS, "Write propagated future ECI positions for satellites and RSOs at epochs t=epoch_time+k*dt (k=0..H-1) into provided float32 arrays: sat_pos[num_envs,H,num_agents,3], rso_pos[num_envs,H,num_rso,3]. Read-only."}, \
    {"vec_set_external_targets", vec_set_external_targets, METH_VARARGS, "Enable reachable-set override: copy global RSO target indices (num_envs x num_agents int32) and activate the external-target action path (classical baselines only)"}, \
    {"vec_clear_external_targets", vec_clear_external_targets, METH_VARARGS, "Disable the external-target override (restore normal top-K action decode)"}

#include "../env_binding.h"

// ── Per-RSO state endpoint (read-only) ──
// args: (handle, out_uncertainty[num_envs, num_rso], out_age[num_envs, num_rso])
// Fills the provided contiguous float32 arrays with the current total RTN
// position uncertainty (surrogate_u_sum_now) and age for every RSO in every env.
// Used by the offline benchmark harness for within-episode time series and
// end-of-episode tail distributions. Assumes all envs share the same num_rso.
static PyObject* vec_rso_state(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    PyObject* u_obj = PyTuple_GetItem(args, 1);
    PyObject* age_obj = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(u_obj, &PyArray_Type) ||
        !PyObject_TypeCheck(age_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "out arrays must be NumPy arrays");
        return NULL;
    }
    PyArrayObject* u_arr = (PyArrayObject*)u_obj;
    PyArrayObject* age_arr = (PyArrayObject*)age_obj;
    if (!PyArray_ISCONTIGUOUS(u_arr) || !PyArray_ISCONTIGUOUS(age_arr)) {
        PyErr_SetString(PyExc_ValueError, "out arrays must be contiguous");
        return NULL;
    }
    float* u_data = (float*)PyArray_DATA(u_arr);
    float* age_data = (float*)PyArray_DATA(age_arr);

    for (int i = 0; i < vec->num_envs; i++) {
        Env* e = vec->envs[i];
        int n = e->num_rso;
        float* u_row = u_data + (size_t)i * n;
        float* age_row = age_data + (size_t)i * n;
        for (int r = 0; r < n; r++) {
            u_row[r] = surrogate_u_sum_now(&e->rsos[r], e);
            age_row[r] = e->rsos[r].age;
        }
    }
    Py_RETURN_NONE;
}
// Read-only cooperation telemetry. Arrays are [num_envs, num_agents].
static PyObject* vec_agent_activity(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    PyObject* target_obj = PyTuple_GetItem(args, 1);
    PyObject* observed_obj = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(target_obj, &PyArray_Type) ||
        !PyObject_TypeCheck(observed_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "target and observed outputs must be NumPy arrays");
        return NULL;
    }

    PyArrayObject* target_arr = (PyArrayObject*)target_obj;
    PyArrayObject* observed_arr = (PyArrayObject*)observed_obj;
    if (!PyArray_ISCONTIGUOUS(target_arr) || !PyArray_ISCONTIGUOUS(observed_arr) ||
        PyArray_TYPE(target_arr) != NPY_INT32 ||
        PyArray_TYPE(observed_arr) != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError,
                        "target must be contiguous int32 and observed contiguous uint8");
        return NULL;
    }

    int* targets = (int*)PyArray_DATA(target_arr);
    uint8_t* observed = (uint8_t*)PyArray_DATA(observed_arr);
    for (int i = 0; i < vec->num_envs; i++) {
        Env* e = vec->envs[i];
        for (int a = 0; a < e->num_agents; a++) {
            size_t idx = (size_t)i * e->num_agents + a;
            targets[idx] = e->satellites[a].last_action;
            observed[idx] = (uint8_t)(e->satellites[a].observed_any != 0);
        }
    }
    Py_RETURN_NONE;
}

// ── Per-RSO per-axis uncertainty endpoint (read-only) ──
// args: (handle, out_u_r, out_u_t, out_u_n, out_age) each [num_envs, num_rso] f32
// Exposes the per-axis surrogate RTN position uncertainty and age so the offline
// benchmark can score the full reachable candidate set with the same metrics the
// observation tokens carry. Assumes all envs share num_rso.
static PyObject* vec_rso_axes(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;
    PyObject* objs[4];
    float* data[4];
    for (int k = 0; k < 4; k++) {
        objs[k] = PyTuple_GetItem(args, 1 + k);
        if (!PyObject_TypeCheck(objs[k], &PyArray_Type)) {
            PyErr_SetString(PyExc_TypeError, "out arrays must be NumPy arrays");
            return NULL;
        }
        if (!PyArray_ISCONTIGUOUS((PyArrayObject*)objs[k])) {
            PyErr_SetString(PyExc_ValueError, "out arrays must be contiguous");
            return NULL;
        }
        data[k] = (float*)PyArray_DATA((PyArrayObject*)objs[k]);
    }
    for (int i = 0; i < vec->num_envs; i++) {
        Env* e = vec->envs[i];
        int n = e->num_rso;
        size_t off = (size_t)i * n;
        for (int r = 0; r < n; r++) {
            RSO* rso = &e->rsos[r];
            data[0][off + r] = surrogate_u_r_now(rso, e);
            data[1][off + r] = surrogate_u_t_now(rso, e);
            data[2][off + r] = surrogate_u_n_now(rso, e);
            data[3][off + r] = rso->age;
        }
    }
    Py_RETURN_NONE;
}

// ── Per-RSO ECI position endpoint (read-only) ──
// args: (handle, out_pos[num_envs, num_rso, 3] f32)
static PyObject* vec_rso_positions(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;
    PyObject* p_obj = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(p_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "out array must be a NumPy array");
        return NULL;
    }
    PyArrayObject* p_arr = (PyArrayObject*)p_obj;
    if (!PyArray_ISCONTIGUOUS(p_arr)) {
        PyErr_SetString(PyExc_ValueError, "out array must be contiguous");
        return NULL;
    }
    float* p_data = (float*)PyArray_DATA(p_arr);
    for (int i = 0; i < vec->num_envs; i++) {
        Env* e = vec->envs[i];
        int n = e->num_rso;
        float* row = p_data + (size_t)i * n * 3;
        for (int r = 0; r < n; r++) {
            row[r * 3 + 0] = e->rsos[r].pos[0];
            row[r * 3 + 1] = e->rsos[r].pos[1];
            row[r * 3 + 2] = e->rsos[r].pos[2];
        }
    }
    Py_RETURN_NONE;
}

// ── Per-(agent, RSO) reachability endpoint (read-only) ──
// args: (handle, out_feasible[num_envs, num_agents, num_rso] uint8)
// Reachability = stage-1 of the env filter: earth-occlusion (if enabled) and the
// optional range mask, matching the env config. This is the FULL candidate set a
// classical baseline may command under reachable-set access (no top-K truncation).
static PyObject* vec_feasible(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;
    PyObject* f_obj = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(f_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "out array must be a NumPy array");
        return NULL;
    }
    PyArrayObject* f_arr = (PyArrayObject*)f_obj;
    if (!PyArray_ISCONTIGUOUS(f_arr)) {
        PyErr_SetString(PyExc_ValueError, "out array must be contiguous");
        return NULL;
    }
    unsigned char* f_data = (unsigned char*)PyArray_DATA(f_arr);
    for (int i = 0; i < vec->num_envs; i++) {
        Env* e = vec->envs[i];
        int A = e->num_agents;
        int R = e->num_rso;
        for (int a = 0; a < A; a++) {
            Satellite* sat = &e->satellites[a];
            unsigned char* row = f_data + ((size_t)i * A + a) * R;
            for (int r = 0; r < R; r++) {
                int reachable = 1;
                if (e->enable_earth_mask) {
                    if (earth_occluded(sat->pos, e->rsos[r].pos)) reachable = 0;
                }
                if (reachable && e->enable_range_mask) {
                    float dx = e->rsos[r].pos[0] - sat->pos[0];
                    float dy = e->rsos[r].pos[1] - sat->pos[1];
                    float dz = e->rsos[r].pos[2] - sat->pos[2];
                    if (vec3_len(dx, dy, dz) > e->max_obs_range) reachable = 0;
                }
                row[r] = (unsigned char)reachable;
            }
        }
    }
    Py_RETURN_NONE;
}

// ── Future reachability tensor (read-only) ──
// args: (handle, H (int), out[num_envs, H, num_agents, num_rso] uint8)
// For each epoch k=0..H-1 at sim time t = epoch_time + k*dt, propagate every
// satellite and RSO with the env's own Keplerian(+J2) model and evaluate the
// SAME earth-occlusion + range reachability used by vec_feasible. k=0 reproduces
// the current geometry (== vec_feasible). No env state is mutated — propagation
// writes into scratch buffers only. Used by the receding-horizon classical
// planner for a geometry-aware cost-to-go.
static PyObject* vec_future_reach(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;
    PyObject* h_obj = PyTuple_GetItem(args, 1);
    PyObject* o_obj = PyTuple_GetItem(args, 2);
    long H = PyLong_AsLong(h_obj);
    if (H == -1 && PyErr_Occurred()) return NULL;
    if (H <= 0) {
        PyErr_SetString(PyExc_ValueError, "H must be a positive integer");
        return NULL;
    }
    if (!PyObject_TypeCheck(o_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "out array must be a NumPy array");
        return NULL;
    }
    PyArrayObject* o_arr = (PyArrayObject*)o_obj;
    if (!PyArray_ISCONTIGUOUS(o_arr) || PyArray_TYPE(o_arr) != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "out array must be contiguous uint8");
        return NULL;
    }
    unsigned char* o_data = (unsigned char*)PyArray_DATA(o_arr);
    for (int i = 0; i < vec->num_envs; i++) {
        Env* e = vec->envs[i];
        int A = e->num_agents;
        int R = e->num_rso;
        float* sat_pos = (float*)malloc((size_t)A * 3 * sizeof(float));
        float* rso_pos = (float*)malloc((size_t)R * 3 * sizeof(float));
        if (!sat_pos || !rso_pos) {
            free(sat_pos); free(rso_pos);
            return PyErr_NoMemory();
        }
        for (long k = 0; k < H; k++) {
            float t = e->epoch_time + (float)k * e->dt;
            float vx, vy, vz;
            for (int a = 0; a < A; a++) {
                Satellite* s = &e->satellites[a];
                propagate_cached(
                    s->oe_a, s->oe_e, s->oe_raan, s->oe_omega, s->oe_M0, t,
                    s->oe_n, s->oe_sqrt_mu_a, s->oe_sqrt_1_e2,
                    s->oe_cos_inc, s->oe_sin_inc, s->oe_raan_dot, s->oe_omega_dot,
                    &sat_pos[a * 3 + 0], &sat_pos[a * 3 + 1], &sat_pos[a * 3 + 2],
                    &vx, &vy, &vz);
            }
            for (int r = 0; r < R; r++) {
                RSO* ro = &e->rsos[r];
                propagate_cached(
                    ro->oe_a, ro->oe_e, ro->oe_raan, ro->oe_omega, ro->oe_M0, t,
                    ro->oe_n, ro->oe_sqrt_mu_a, ro->oe_sqrt_1_e2,
                    ro->oe_cos_inc, ro->oe_sin_inc, ro->oe_raan_dot, ro->oe_omega_dot,
                    &rso_pos[r * 3 + 0], &rso_pos[r * 3 + 1], &rso_pos[r * 3 + 2],
                    &vx, &vy, &vz);
            }
            for (int a = 0; a < A; a++) {
                float* sp = &sat_pos[a * 3];
                unsigned char* row = o_data
                    + (((((size_t)i * H + k) * A) + a) * R);
                for (int r = 0; r < R; r++) {
                    float* rp = &rso_pos[r * 3];
                    int reachable = 1;
                    if (e->enable_earth_mask) {
                        if (earth_occluded(sp, rp)) reachable = 0;
                    }
                    if (reachable && e->enable_range_mask) {
                        float dx = rp[0] - sp[0];
                        float dy = rp[1] - sp[1];
                        float dz = rp[2] - sp[2];
                        if (vec3_len(dx, dy, dz) > e->max_obs_range) reachable = 0;
                    }
                    row[r] = (unsigned char)reachable;
                }
            }
        }
        free(sat_pos); free(rso_pos);
    }
    Py_RETURN_NONE;
}

// ── Future propagated positions (read-only) ──
// args: (handle, H (int), sat_pos[num_envs,H,num_agents,3] f32,
//        rso_pos[num_envs,H,num_rso,3] f32)
// Uses the env's own Keplerian(+J2) propagator to write future positions at
// epochs t = epoch_time + k*dt for k=0..H-1. No env state is mutated.
static PyObject* vec_future_positions(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;
    PyObject* h_obj = PyTuple_GetItem(args, 1);
    PyObject* s_obj = PyTuple_GetItem(args, 2);
    PyObject* r_obj = PyTuple_GetItem(args, 3);
    long H = PyLong_AsLong(h_obj);
    if (H == -1 && PyErr_Occurred()) return NULL;
    if (H <= 0) {
        PyErr_SetString(PyExc_ValueError, "H must be a positive integer");
        return NULL;
    }
    if (!PyObject_TypeCheck(s_obj, &PyArray_Type) ||
        !PyObject_TypeCheck(r_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "output arrays must be NumPy arrays");
        return NULL;
    }
    PyArrayObject* s_arr = (PyArrayObject*)s_obj;
    PyArrayObject* r_arr = (PyArrayObject*)r_obj;
    if (!PyArray_ISCONTIGUOUS(s_arr) || PyArray_TYPE(s_arr) != NPY_FLOAT32 ||
        !PyArray_ISCONTIGUOUS(r_arr) || PyArray_TYPE(r_arr) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_ValueError, "output arrays must be contiguous float32");
        return NULL;
    }
    float* s_data = (float*)PyArray_DATA(s_arr);
    float* r_data = (float*)PyArray_DATA(r_arr);
    for (int i = 0; i < vec->num_envs; i++) {
        Env* e = vec->envs[i];
        int A = e->num_agents;
        int R = e->num_rso;
        for (long k = 0; k < H; k++) {
            float t = e->epoch_time + (float)k * e->dt;
            float* s_row = s_data + ((((size_t)i * H) + k) * A * 3);
            float* r_row = r_data + ((((size_t)i * H) + k) * R * 3);
            float vx, vy, vz;
            for (int a = 0; a < A; a++) {
                Satellite* s = &e->satellites[a];
                propagate_cached(
                    s->oe_a, s->oe_e, s->oe_raan, s->oe_omega, s->oe_M0, t,
                    s->oe_n, s->oe_sqrt_mu_a, s->oe_sqrt_1_e2,
                    s->oe_cos_inc, s->oe_sin_inc, s->oe_raan_dot, s->oe_omega_dot,
                    &s_row[a * 3 + 0], &s_row[a * 3 + 1], &s_row[a * 3 + 2],
                    &vx, &vy, &vz);
            }
            for (int r = 0; r < R; r++) {
                RSO* ro = &e->rsos[r];
                propagate_cached(
                    ro->oe_a, ro->oe_e, ro->oe_raan, ro->oe_omega, ro->oe_M0, t,
                    ro->oe_n, ro->oe_sqrt_mu_a, ro->oe_sqrt_1_e2,
                    ro->oe_cos_inc, ro->oe_sin_inc, ro->oe_raan_dot, ro->oe_omega_dot,
                    &r_row[r * 3 + 0], &r_row[r * 3 + 1], &r_row[r * 3 + 2],
                    &vx, &vy, &vz);
            }
        }
    }
    Py_RETURN_NONE;
}

// ── External-target override (write) ──
// args: (handle, targets[num_envs, num_agents] int32)
// Copies the supplied global RSO target indices into each env and activates the
// override so the next c_step slews each agent toward its assigned RSO (or holds
// pointing if the index is < 0). For classical baselines only \u2014 never the NN.
static PyObject* vec_set_external_targets(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;
    PyObject* t_obj = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(t_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "targets must be a NumPy array");
        return NULL;
    }
    PyArrayObject* t_arr = (PyArrayObject*)t_obj;
    if (!PyArray_ISCONTIGUOUS(t_arr) ||
        PyArray_TYPE(t_arr) != NPY_INT32) {
        PyErr_SetString(PyExc_ValueError, "targets must be a contiguous int32 array");
        return NULL;
    }
    int* t_data = (int*)PyArray_DATA(t_arr);
    for (int i = 0; i < vec->num_envs; i++) {
        Env* e = vec->envs[i];
        int A = e->num_agents;
        int* row = t_data + (size_t)i * A;
        for (int a = 0; a < A; a++) {
            e->external_targets[a] = row[a];
        }
        e->external_target_active = 1;
    }
    Py_RETURN_NONE;
}

// ── External-target override (clear) ──
// args: (handle,)
static PyObject* vec_clear_external_targets(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;
    for (int i = 0; i < vec->num_envs; i++) {
        vec->envs[i]->external_target_active = 0;
    }
    Py_RETURN_NONE;
}

// ── Profiling endpoint ──
static PyObject* vec_profile(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) return NULL;

    PyObject* dict = PyDict_New();
#ifdef PROFILE_ENV
    double propagation = 0, actions = 0, observation_det = 0;
    double uncertainty = 0, reward = 0, build_obs = 0, total_step = 0;
    long step_count = 0;
    for (int i = 0; i < vec->num_envs; i++) {
        Env* e = vec->envs[i];
        propagation    += e->prof_propagation;
        actions        += e->prof_actions;
        observation_det += e->prof_observation_det;
        uncertainty    += e->prof_uncertainty;
        reward         += e->prof_reward;
        build_obs      += e->prof_build_obs;
        total_step     += e->prof_total_step;
        step_count     += e->prof_step_count;
    }
    PyDict_SetItemString(dict, "propagation",     PyFloat_FromDouble(propagation));
    PyDict_SetItemString(dict, "actions",          PyFloat_FromDouble(actions));
    PyDict_SetItemString(dict, "observation_det",  PyFloat_FromDouble(observation_det));
    PyDict_SetItemString(dict, "uncertainty",      PyFloat_FromDouble(uncertainty));
    PyDict_SetItemString(dict, "reward",           PyFloat_FromDouble(reward));
    PyDict_SetItemString(dict, "build_obs",        PyFloat_FromDouble(build_obs));
    PyDict_SetItemString(dict, "total_step",       PyFloat_FromDouble(total_step));
    PyDict_SetItemString(dict, "step_count",       PyFloat_FromDouble((double)step_count));
#endif
    return dict;
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    // Core
    env->num_agents       = unpack(kwargs, "num_agents");
    env->num_rso          = unpack(kwargs, "num_rso");
    env->max_steps        = unpack(kwargs, "max_steps");
    env->dt               = unpack(kwargs, "dt");
    env->action_mode      = (int)unpack(kwargs, "action_mode");
    env->rso_top_k        = (int)unpack(kwargs, "rso_top_k");
    env->propagation_mode = (int)unpack(kwargs, "propagation_mode");

    // Sensor
    env->fov_deg          = unpack(kwargs, "fov_deg");
    env->max_slew_rate_deg = unpack(kwargs, "max_slew_rate_deg");

    // Satellite orbits (multi-orbit constellation)
    env->num_orbits = (int)unpack(kwargs, "num_orbits");
    if (env->num_orbits < 1) env->num_orbits = 1;
    if (env->num_orbits > MAX_ORBITS) env->num_orbits = MAX_ORBITS;
    for (int o = 0; o < env->num_orbits; o++) {
        char key[64];
        snprintf(key, sizeof(key), "orbit_%d_a", o);
        env->orbits[o].a = unpack(kwargs, key);
        snprintf(key, sizeof(key), "orbit_%d_e", o);
        env->orbits[o].e = unpack(kwargs, key);
        snprintf(key, sizeof(key), "orbit_%d_inc", o);
        env->orbits[o].inc = unpack(kwargs, key);
        snprintf(key, sizeof(key), "orbit_%d_raan", o);
        env->orbits[o].raan = unpack(kwargs, key);
        snprintf(key, sizeof(key), "orbit_%d_omega", o);
        env->orbits[o].omega = unpack(kwargs, key);
        snprintf(key, sizeof(key), "orbit_%d_num_sats", o);
        env->orbits[o].num_sats = (int)unpack(kwargs, key);
    }

    // RSO population bounds
    env->rso_a_min        = unpack(kwargs, "rso_a_min");
    env->rso_a_max        = unpack(kwargs, "rso_a_max");
    env->rso_e_min        = unpack(kwargs, "rso_e_min");
    env->rso_e_max        = unpack(kwargs, "rso_e_max");
    env->rso_inc_min      = unpack(kwargs, "rso_inc_min");
    env->rso_inc_max      = unpack(kwargs, "rso_inc_max");
    env->rso_size_min     = unpack(kwargs, "rso_size_min");
    env->rso_size_max     = unpack(kwargs, "rso_size_max");

    // Uncertainty model
    env->growth_mode      = (int)unpack(kwargs, "growth_mode");
    env->growth_a_r       = unpack(kwargs, "growth_a_r");
    env->growth_a_t       = unpack(kwargs, "growth_a_t");
    env->growth_a_n       = unpack(kwargs, "growth_a_n");
    env->growth_b_r       = unpack(kwargs, "growth_b_r");
    env->growth_b_t       = unpack(kwargs, "growth_b_t");
    env->growth_b_n       = unpack(kwargs, "growth_b_n");
    env->k_v              = unpack(kwargs, "k_v");
    env->reduction_mode   = (int)unpack(kwargs, "reduction_mode");
    env->alpha_r          = unpack(kwargs, "alpha_r");
    env->alpha_t          = unpack(kwargs, "alpha_t");
    env->alpha_n          = unpack(kwargs, "alpha_n");
    env->alpha_vel        = unpack(kwargs, "alpha_vel");
    env->u_min_r          = unpack(kwargs, "u_min_r");
    env->u_min_t          = unpack(kwargs, "u_min_t");
    env->u_min_n          = unpack(kwargs, "u_min_n");
    env->vel_u_min        = unpack(kwargs, "vel_u_min");

    // EKF backend (Step 2)
    env->uncertainty_mode = (int)unpack(kwargs, "uncertainty_mode");
    env->ekf_sigma_a      = unpack(kwargs, "ekf_sigma_a");
    env->ekf_meas_sigma   = unpack(kwargs, "ekf_meas_sigma");
    env->ekf_max_substep_dt = unpack(kwargs, "ekf_max_substep_dt");
    env->ekf_sigma_max    = unpack(kwargs, "ekf_sigma_max");

    // Initial uncertainty (ranges; max==min means fixed)
    env->u_init_r         = unpack(kwargs, "u_init_r");
    env->u_init_t         = unpack(kwargs, "u_init_t");
    env->u_init_n         = unpack(kwargs, "u_init_n");
    env->u_init_r_max     = unpack(kwargs, "u_init_r_max");
    env->u_init_t_max     = unpack(kwargs, "u_init_t_max");
    env->u_init_n_max     = unpack(kwargs, "u_init_n_max");
    env->sigma_vel_init   = unpack(kwargs, "sigma_vel_init");
    env->sigma_vel_init_max = unpack(kwargs, "sigma_vel_init_max");
    env->age_init         = unpack(kwargs, "age_init");
    env->age_init_max     = unpack(kwargs, "age_init_max");

    // Observation quality
    env->quality_mode     = (int)unpack(kwargs, "quality_mode");
    env->quality_dist_weight  = unpack(kwargs, "quality_dist_weight");
    env->quality_angle_weight = unpack(kwargs, "quality_angle_weight");
    env->quality_size_weight  = unpack(kwargs, "quality_size_weight");
    env->max_obs_range    = unpack(kwargs, "max_obs_range");

    // Masking
    env->enable_earth_mask = (int)unpack(kwargs, "enable_earth_mask");
    env->enable_range_mask = (int)unpack(kwargs, "enable_range_mask");
    env->enable_fov_obs    = (int)unpack(kwargs, "enable_fov_obs");

    // Reward
    env->reward_mode      = (int)unpack(kwargs, "reward_mode");
    env->alpha_u          = unpack(kwargs, "alpha_u");
    env->alpha_v          = unpack(kwargs, "alpha_v");
    env->alpha_local      = unpack(kwargs, "alpha_local");
    env->alpha_ctrl       = unpack(kwargs, "alpha_ctrl");
    env->reward_scale     = unpack(kwargs, "reward_scale");
    env->reward_baseline  = unpack(kwargs, "reward_baseline");
    env->reward_clip_min  = unpack(kwargs, "reward_clip_min");
    env->uncertainty_threshold = unpack(kwargs, "uncertainty_threshold");
    env->team_spirit      = unpack(kwargs, "team_spirit");

    // Priority weights
    env->priority_alpha   = unpack(kwargs, "priority_alpha");
    env->priority_beta    = unpack(kwargs, "priority_beta");
    env->priority_gamma   = unpack(kwargs, "priority_gamma");

    // Normalization
    env->obs_dist_norm    = unpack(kwargs, "obs_dist_norm");
    env->obs_vel_norm     = unpack(kwargs, "obs_vel_norm");
    env->obs_age_norm     = unpack(kwargs, "obs_age_norm");
    env->obs_u_norm       = unpack(kwargs, "obs_u_norm");
    env->obs_u_vel_norm   = unpack(kwargs, "obs_u_vel_norm");

    // Rendering
    env->render_scale     = unpack(kwargs, "render_scale");
    env->render_rso_scale = unpack(kwargs, "render_rso_scale");
    env->render_fps       = (int)unpack(kwargs, "render_fps");

    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "episode_return",             log->episode_return);
    assign_to_dict(dict, "episode_length",             log->episode_length);
    assign_to_dict(dict, "mean_uncertainty",            log->mean_uncertainty);
    assign_to_dict(dict, "max_uncertainty",             log->max_uncertainty);
    assign_to_dict(dict, "fraction_above_threshold",    log->fraction_above_threshold);
    assign_to_dict(dict, "total_observations",          log->total_observations);
    assign_to_dict(dict, "mean_age",                    log->mean_age);
    assign_to_dict(dict, "effective_tracked",            log->effective_tracked);
    assign_to_dict(dict, "n",                           log->n);
    return 0;
}

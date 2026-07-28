"""JAX mirror of the numeric kernels in ``vbc_landscape.py``.

Optional. Only Layer 4 (local Hessian stability, ``stability_jax.py``) needs JAX. Layers
1 to 3 in ``vbc_landscape.py`` are numpy-only. These reimplementations exist so the
energy is differentiable (for autodiff Hessians) and so the whole stack can run in one
JAX environment. The union-find minimax code stays in ``vbc_landscape.py`` (numpy), it is
a discrete flooding algorithm with no meaningful gradient.
"""
from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        'vbc_landscape_jax needs JAX (Layer 4 only). Install with  pip install "jax[cpu]"  '
        '(or "jax[cuda12]" on a GPU node). Layers 1 to 3 in vbc_landscape.py need only numpy.'
    ) from exc

# float64 throughout, so the Hessian eigenvalues match numpy and the p_stable sign test
# near lambda = 0 is not dominated by float32 roundoff.
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Base potential and gate. Branch-free mirrors of vbc_landscape.py, so they are
# traceable, vmappable and twice-differentiable by autodiff.
# ---------------------------------------------------------------------------

def density_phi(r, lo, hi):
    """Neighbour-counting weight. 1 below ``lo``, a smooth shoulder to 0 at ``hi``."""
    s = (r - lo) / (hi - lo)
    shoulder = (1.0 - s ** 2) ** 2
    return jnp.where(r < lo, 1.0, jnp.where(r <= hi, shoulder, 0.0))


def omega_gate(phi, ncut):
    """Valency gate. 1 below ``ncut``, a smooth shoulder to 0 at ``ncut + 1``."""
    s = phi - ncut
    shoulder = (1.0 - s ** 2) ** 2
    return jnp.where(phi < ncut, 1.0, jnp.where(phi <= ncut + 1.0, shoulder, 0.0))


def V_rep(r, r_rep, E_r):
    """Soft-core repulsion on ``[0, r_rep]``."""
    z = r / r_rep
    return jnp.where((r >= 0.0) & (r <= r_rep), E_r * (1.0 - z ** 2) ** 2, 0.0)


def V_att(r, r_rep, r_att, E_a):
    """Attractive well on ``(r_rep, r_att)`` with its minimum at the midpoint."""
    c = 0.5 * (r_att + r_rep)
    hw = 0.5 * (r_att - r_rep)
    z = (r - c) / hw
    return jnp.where((r > r_rep) & (r < r_att), E_a * (1.0 - z ** 2) ** 2, 0.0)


# ---------------------------------------------------------------------------
# Differentiable total cluster energy. The DOF are the neighbour radii, so
# jax.hessian(cluster_energy) is the m x m radial Hessian used by Layer 4.
# ---------------------------------------------------------------------------

def cluster_energy(r_vec, *, r_low, r_high, ncut, r_rep, r_att, E_r, E_a):
    """Option-1 total energy of a central particle C with its ``m`` neighbours at radial
    distances ``r_vec`` (shape ``(m,)``), C fixed at the origin and angles fixed.

        phi_C = sum_j density_phi(r_j)
        E     = sum_i V_rep(r_i)
                + omega_gate(phi_C) * sum_i omega_gate(density_phi(r_i)) * V_att(r_i)

    The shared gate ``omega_gate(phi_C)`` multiplies every bond, so adding a neighbour
    weakens all the others (the back-reaction) automatically. Returns a scalar.
    """
    w = density_phi(r_vec, r_low, r_high)
    gC = omega_gate(jnp.sum(w), ncut)
    gi = omega_gate(w, ncut)
    rep = jnp.sum(V_rep(r_vec, r_rep, E_r))
    att = jnp.sum(gi * V_att(r_vec, r_rep, r_att, E_a))
    return rep + gC * att


# ---------------------------------------------------------------------------
# JAX ports of the spurious-depth layers. For parity with the numpy versions and
# a one-environment story, not the fast path (Layers 1 to 3 use numpy).
# ---------------------------------------------------------------------------

def _interior_depth(U, baseline, E_a):
    """Depth of the interior minimum of a 1D profile below ``baseline`` (else 0)."""
    i = int(jnp.argmin(U[1:-1])) + 1
    is_trap = (U[i] < U[i - 1]) & (U[i] < U[i + 1]) & (U[i] < baseline)
    return float(jnp.where(is_trap, (baseline - U[i]) / abs(E_a), 0.0))


def naive_spurious_depth(valency, r_low, r_high, n_target, *, r_rep, r_att, E_r, E_a, n_r=501):
    """Layer 1 (JAX). Bare C-N pair well, no back-reaction, baseline zero."""
    r_arr = jnp.linspace(r_rep, r_att, n_r)
    ncut = valency + (n_target - 1.0)
    r_star = 0.5 * (r_rep + r_att)
    rho_rs = density_phi(jnp.asarray(r_star), r_low, r_high)
    rho_r = density_phi(r_arr, r_low, r_high)
    W_C = omega_gate(valency * rho_rs + rho_r, ncut)
    W_N = omega_gate(rho_r, ncut)
    U = V_rep(r_arr, r_rep, E_r) + W_C * W_N * V_att(r_arr, r_rep, r_att, E_a)
    return _interior_depth(U, 0.0, E_a)


def sporadic_min_depth(valency, r_low, r_high, n_target, *, r_rep, r_att, E_r, E_a, n_r=501):
    """Layer 2 (JAX). Deterministic total-cluster-energy spurious depth."""
    r_arr = jnp.linspace(r_rep, r_att, n_r)
    ncut = valency + (n_target - 1.0)
    r_star = 0.5 * (r_rep + r_att)
    rho_rs = density_phi(jnp.asarray(r_star), r_low, r_high)
    vatt_rs = V_att(jnp.asarray(r_star), r_rep, r_att, E_a)
    W_S = omega_gate(rho_rs, ncut)
    rho_r = density_phi(r_arr, r_low, r_high)
    W_C = omega_gate(valency * rho_rs + rho_r, ncut)
    W_N = omega_gate(rho_r, ncut)
    U = (V_rep(r_arr, r_rep, E_r) + W_C * W_N * V_att(r_arr, r_rep, r_att, E_a)
         + valency * W_C * W_S * vatt_rs)
    baseline = float(valency * omega_gate(valency * rho_rs, ncut) * W_S * vatt_rs)
    return _interior_depth(U, baseline, E_a)


def spurious_depth(valency, r_low, r_high, n_target, *, r_rep, r_att, E_r, E_a,
                   key="deterministic", n_sample=100, seed=0, n_r=501):
    """Layer 3 (JAX). ``key="deterministic"`` calls ``sporadic_min_depth``. ``key="random"``
    scatters ``valency`` neighbours uniformly in the binding zone for each of ``n_sample``
    placements (vectorised with ``vmap``) and averages the depth."""
    if key == "deterministic":
        return sporadic_min_depth(valency, r_low, r_high, n_target,
                                  r_rep=r_rep, r_att=r_att, E_r=E_r, E_a=E_a, n_r=n_r)
    if key != "random":
        raise ValueError("key must be 'deterministic' or 'random'")

    r_arr = jnp.linspace(r_rep, r_att, n_r)
    ncut = valency + (n_target - 1.0)
    rho_r = density_phi(r_arr, r_low, r_high)

    def one(r_spec):
        w = density_phi(r_spec, r_low, r_high)
        S = jnp.sum(omega_gate(w, ncut) * V_att(r_spec, r_rep, r_att, E_a))
        sum_w = jnp.sum(w)
        W_C = omega_gate(sum_w + rho_r, ncut)
        W_N = omega_gate(rho_r, ncut)
        U = V_rep(r_arr, r_rep, E_r) + W_C * W_N * V_att(r_arr, r_rep, r_att, E_a) + W_C * S
        baseline = omega_gate(sum_w, ncut) * S
        i = jnp.argmin(U[1:-1]) + 1
        is_trap = (U[i] < U[i - 1]) & (U[i] < U[i + 1]) & (U[i] < baseline)
        return jnp.where(is_trap, (baseline - U[i]) / abs(E_a), 0.0)

    keys = jax.random.split(jax.random.PRNGKey(seed), n_sample)
    r_spec = jax.vmap(lambda k: jax.random.uniform(k, (valency,), minval=r_rep, maxval=r_att))(keys)
    return float(jnp.mean(jax.vmap(one)(r_spec)))

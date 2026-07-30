"""Over-coordination Hessian stability test (optional, needs JAX).

The only stability test here, and the only JAX code; the rest is numpy. Does a centre holding
``valency`` bonds trap one extra particle (``m = valency + 1`` radii)? find_minima relaxes
every grid config to ``grad U = 0`` and keeps the interior positive-definite minima. Radii are
the only DOF, as neighbours couple only to the centre.
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import minimize

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        'stability_jax needs JAX. Install with  pip install "jax[cpu]"  (or "jax[cuda12]" on '
        "a GPU node). The rest of swap_calibration is numpy-only.") from exc

jax.config.update("jax_enable_x64", True)


def density_phi(r, lo, hi):
    s = (r - lo) / (hi - lo)
    return jnp.where(r < lo, 1.0, jnp.where(r <= hi, (1.0 - s ** 2) ** 2, 0.0))


def omega_gate(phi, ncut):
    s = phi - ncut
    return jnp.where(phi < ncut, 1.0, jnp.where(phi <= ncut + 1.0, (1.0 - s ** 2) ** 2, 0.0))


def V_rep(r, r_rep, E_r):
    z = r / r_rep
    return jnp.where((r >= 0.0) & (r <= r_rep), E_r * (1.0 - z ** 2) ** 2, 0.0)


def V_att(r, r_rep, r_att, E_a):
    c, hw = 0.5 * (r_att + r_rep), 0.5 * (r_att - r_rep)
    z = (r - c) / hw
    return jnp.where((r > r_rep) & (r < r_att), E_a * (1.0 - z ** 2) ** 2, 0.0)


def cluster_energy(r_vec, *, r_low, r_high, ncut, r_rep, r_att, E_r, E_a):
    """Differentiable total energy of a centre with neighbours at radii ``r_vec``. The shared
    gate ``omega_gate(sum_j phi_j)`` scales every bond, so a new neighbour weakens the rest."""
    w = density_phi(r_vec, r_low, r_high)
    gate = omega_gate(jnp.sum(w), ncut)
    return jnp.sum(V_rep(r_vec, r_rep, E_r)) + gate * jnp.sum(
        omega_gate(w, ncut) * V_att(r_vec, r_rep, r_att, E_a))


def curvature_scale(E_a, r_rep, r_att):
    """Isolated-bond curvature ``16|E_a|/(r_att-r_rep)^2`` (a decoupled bond gives lambda_min/this = 1)."""
    return 16.0 * abs(E_a) / (r_att - r_rep) ** 2


def find_minima(valency, r_low, r_high, n_target, *, r_rep, r_att, E_r, E_a,
                n_grid=10, grad_tol=1e-4, pd_tol=1e-6, tol=1e-3):
    """Interior positive-definite minima of the over-coordinated cluster (``m = valency + 1``).

    Relaxes each of the ``C(n_grid + m - 1, m)`` distinct grid configs (sorted radii multisets,
    since neighbours are identical) to ``grad U = 0`` in ``(r_rep, r_att)`` and keeps the
    interior, strictly convex ones. Returns distinct ``(r_vec_sorted, U, lambda_min)`` by
    energy; empty means no trap.
    """
    m = valency + 1
    ncut = valency + (n_target - 1.0)
    kw = dict(r_low=r_low, r_high=r_high, ncut=ncut, r_rep=r_rep, r_att=r_att, E_r=E_r, E_a=E_a)
    E = jax.jit(lambda r: cluster_energy(r, **kw))
    dE = jax.jit(jax.grad(lambda r: cluster_energy(r, **kw)))
    d2E = jax.jit(jax.hessian(lambda r: cluster_energy(r, **kw)))
    f = lambda r: float(E(jnp.asarray(r)))
    g = lambda r: np.asarray(dE(jnp.asarray(r)), float)

    grid = np.linspace(r_rep, r_att, n_grid)
    out = []
    for combo in combinations_with_replacement(grid, m):
        x = np.sort(minimize(f, np.array(combo), jac=g, method="L-BFGS-B",
                             bounds=[(r_rep, r_att)] * m).x)
        if np.linalg.norm(g(x)) > grad_tol:                      # not stationary
            continue
        if x[0] <= r_rep + tol or x[-1] >= r_att - tol:          # a bond on the boundary
            continue
        lam = float(np.linalg.eigvalsh(np.asarray(d2E(jnp.asarray(x)))).min())
        if lam <= pd_tol:                                        # not strictly convex
            continue
        if any(np.linalg.norm(x - y[0]) < tol for y in out):     # already found
            continue
        out.append((x, f(x), lam))
    out.sort(key=lambda t: t[1])
    return out


def optimal_state_energy(valency, r_low, r_high, n_target, *, r_rep, r_att, E_r, E_a):
    """Energy of the ideal ``valency``-coordinated state (all bonds at the well centre), the
    depth reference. equals ``valency * E_a`` when the cap covers the valency."""
    ncut = valency + (n_target - 1.0)
    r_star = 0.5 * (r_rep + r_att)
    return float(cluster_energy(jnp.full(valency, r_star), r_low=r_low, r_high=r_high,
                                ncut=ncut, r_rep=r_rep, r_att=r_att, E_r=E_r, E_a=E_a))

"""Local Hessian stability of neighbour configurations (optional, needs JAX).

For a central particle C with ``m`` neighbours at radial distances ``r_vec`` (C fixed at
the origin, angles fixed, interactions between non-central neighbours omitted), the total
cluster energy ``cluster_energy`` is differentiable, so its gradient and Hessian follow by
autodiff. The Hessian eigenvalues report the local curvature. all positive means the
arrangement is a locally convex (stable) minimum in the radial coordinates.

The scenario is OVER-COORDINATION. A central particle already holding its ``valency`` bonds
plus one extra approaching particle, so the DOF are the ``m = valency + 1`` neighbour radii
and the Hessian is ``(valency+1) x (valency+1)``. A spurious trap is a stable arrangement
in which all ``valency + 1`` neighbours stay bonded.

DOF. Radial distances only. In this energy neighbours interact with C but not with each
other, so a full-3D Hessian would carry 2m exact zero modes (rotating a neighbour about C
at fixed radius costs nothing). The radial coordinates are the physically meaningful and
tractable DOF.

Terminology. This LOCAL, per-configuration convexity is distinct from the paper's
macroscopic H-stability (the Fisher-Ruelle phi_min bound), do not call it "H-stable".

Two tools. ``stability_random`` and ``stability_grid`` CHARACTERISE the configuration space.
they return the Hessian eigenvalues and the gradient norm of every sampled or gridded
arrangement, and the fraction that are locally convex (``p_stable``). A random or gridded
arrangement is almost never exactly stationary, and the gate kinks make the discrete energy
bumpy, so reading true traps straight off the grid is unreliable. ``find_minima`` therefore
refines a grid of seeds to the actual stationary points (grad U = 0) and keeps only the
strictly convex interior ones. the grid screens, the relaxation confirms.
"""
from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        'stability_jax needs JAX. Install with  pip install "jax[cpu]"  '
        '(or "jax[cuda12]" on a GPU node). Layers 1 to 3 in vbc_landscape.py need only numpy.'
    ) from exc

from vbc_landscape_jax import cluster_energy


class StabilityResult(NamedTuple):
    lambda_min: np.ndarray   # (n_config,) smallest Hessian eigenvalue per configuration
    p_stable: float          # fraction of configs with a positive-definite Hessian
    grad_norm: np.ndarray    # (n_config,) |grad E| at the config (0 => stationary)
    n_config: int
    m: int


def curvature_scale(E_a, r_rep, r_att):
    """Curvature of one isolated bond at the well centre, the scale for ``lambda_min``.

    ``|d^2 V_att/dr^2|`` at ``r = r_star`` equals ``16 |E_a| / (r_att - r_rep)^2`` (from
    ``V_att = E_a (1 - z^2)^2`` with ``z = (r - r_star)/((r_att - r_rep)/2)``). A single
    decoupled bond then has ``lambda_min / curvature_scale = 1``.
    """
    return 16.0 * abs(E_a) / (r_att - r_rep) ** 2


def _make_batch(r_low, r_high, ncut, r_rep, r_att, E_r, E_a):
    """Return a jitted, vmapped map  r_vec -> (eigenvalues, |grad|)  per config."""
    def energy(r):
        return cluster_energy(r, r_low=r_low, r_high=r_high, ncut=ncut,
                              r_rep=r_rep, r_att=r_att, E_r=E_r, E_a=E_a)
    grad = jax.grad(energy)
    hess = jax.hessian(energy)

    def one(r_vec):
        H = hess(r_vec)
        H = 0.5 * (H + H.T)                     # kill autodiff round-off asymmetry
        return jnp.linalg.eigvalsh(H), jnp.linalg.norm(grad(r_vec))

    return jax.jit(jax.vmap(one))


def _grid_configs(m, n_grid, r_min, r_max, dedup):
    """Radial grid configurations, sorted multisets (dedup) or the full product."""
    pts = np.linspace(r_min, r_max, n_grid)
    combos = (itertools.combinations_with_replacement(range(n_grid), m) if dedup
              else itertools.product(range(n_grid), repeat=m))
    idx = np.fromiter(itertools.chain.from_iterable(combos), dtype=np.int64).reshape(-1, m)
    return pts[idx]


def _reduce(batch, r_all, chunk):
    """Stream configs through ``batch`` in chunks, keep only per-config scalars."""
    r_all = jnp.asarray(r_all)
    n, m = int(r_all.shape[0]), int(r_all.shape[1])
    lam = np.empty(n)
    gn = np.empty(n)
    for s in range(0, n, chunk):
        w, g = batch(r_all[s:s + chunk])
        lam[s:s + chunk] = np.asarray(w)[:, 0]
        gn[s:s + chunk] = np.asarray(g)
    return StabilityResult(lam, float(np.mean(lam > 0.0)), gn, n, m)


def stability_random(valency, r_low, r_high, n_target, *, r_rep, r_att, E_r, E_a,
                     n_sample=100_000, seed=0, chunk=50_000):
    """Local Hessian convexity over ``n_sample`` uniform placements of the ``valency + 1``
    neighbours (the saturating shell plus one extra particle) in the binding zone
    ``(r_rep, r_att)``. Returns a ``StabilityResult`` (a screen, not the traps)."""
    m = valency + 1
    ncut = valency + (n_target - 1.0)
    batch = _make_batch(r_low, r_high, ncut, r_rep, r_att, E_r, E_a)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_sample)
    r_all = jax.vmap(
        lambda k: jax.random.uniform(k, (m,), minval=r_rep, maxval=r_att))(keys)
    return _reduce(batch, r_all, chunk)


def stability_grid(valency, r_low, r_high, n_target, *, r_rep, r_att, E_r, E_a,
                   n_grid=10, r_min=None, r_max=None, dedup=True, chunk=50_000):
    """Local Hessian convexity on a radial grid of the ``valency + 1`` neighbour distances.

    Neighbours are identical, so two radial tuples differing only by a permutation are the
    SAME physical configuration (same energy, same Hessian spectrum). With ``dedup=True`` we
    enumerate sorted multisets of ``m`` grid indices, giving ``C(n_grid + m - 1, m)`` unique
    configs instead of the ``n_grid ** m`` product (e.g. n_grid=10, m=7 gives C(16,7)=11440
    versus 1e7). ``dedup=False`` reproduces the full product, for a throughput benchmark.
    Returns a ``StabilityResult`` (a screen, ``find_minima`` locates the actual traps).
    """
    m = valency + 1
    ncut = valency + (n_target - 1.0)
    r_min = r_rep if r_min is None else r_min
    r_max = r_att if r_max is None else r_max
    r_all = _grid_configs(m, n_grid, r_min, r_max, dedup)
    batch = _make_batch(r_low, r_high, ncut, r_rep, r_att, E_r, E_a)
    return _reduce(batch, r_all, chunk)


def find_minima(valency, r_low, r_high, n_target, *, r_rep, r_att, E_r, E_a,
                n_grid=21, n_seed=500, seed=0, grad_tol=1e-4, boundary_eps=1e-4,
                dedup_tol=1e-3, pd_tol=1e-6):
    """Strictly convex interior minima of the over-coordination cluster energy.

    The cluster is a central particle plus ``m = valency + 1`` neighbours (the saturating
    shell and one extra particle), all free radial DOF. Seeds (the symmetric state, an
    ordered one-neighbour-displaced grid through it, and a random interior sample) are
    relaxed to grad U = 0 with a bounded quasi-Newton step, then kept only if the Hessian is
    strictly positive-definite (``lambda_min > pd_tol``). Relaxation is what makes this
    reliable. reading discrete minima straight off a coarse grid gives kink artefacts and
    is degenerate on the flat ``U = 0`` plateau of a gate that switches every bond off.
    Boundary-touching points (a neighbour on the repulsive wall at ``r_rep``) are dropped, and
    interior points are deduplicated (neighbours are identical, radial tuples compared
    sorted). Returns ``(r_vec_sorted, U, lambda_min)`` ordered by energy. each entry is a
    genuine spurious trap, a stable arrangement with all ``valency + 1`` neighbours bonded.
    """
    m = valency + 1
    ncut = valency + (n_target - 1.0)
    p = dict(r_low=r_low, r_high=r_high, ncut=ncut, r_rep=r_rep, r_att=r_att, E_r=E_r, E_a=E_a)
    energy = jax.jit(lambda r: cluster_energy(r, **p))
    grad = jax.jit(jax.grad(lambda r: cluster_energy(r, **p)))
    hess = jax.jit(jax.hessian(lambda r: cluster_energy(r, **p)))
    E_np = lambda r: float(energy(jnp.asarray(r)))
    G_np = lambda r: np.asarray(grad(jnp.asarray(r)), dtype=float)

    r_star = 0.5 * (r_rep + r_att)
    seeds = [np.full(m, r_star)]
    for j in range(m):                          # ordered one-neighbour-displaced grid
        for v in np.linspace(r_rep, r_att, n_grid):
            s = np.full(m, r_star)
            s[j] = v
            seeds.append(s)
    rng = np.random.default_rng(seed)
    seeds += [rng.uniform(r_rep, r_att, m) for _ in range(n_seed)]

    kept = []
    for x0 in seeds:
        res = minimize(E_np, x0, jac=G_np, method="L-BFGS-B",
                       bounds=[(r_rep, r_att)] * m,
                       options=dict(maxiter=500, ftol=1e-14, gtol=1e-10))
        x = np.sort(res.x)
        if np.linalg.norm(G_np(res.x)) > grad_tol:      # not a stationary point
            continue
        if x[0] < r_rep + boundary_eps or x[-1] > r_att - boundary_eps:  # on the boundary
            continue
        lam = float(np.linalg.eigvalsh(np.asarray(hess(jnp.asarray(x)))).min())
        if lam <= pd_tol:                                # flat/degenerate, not a trap
            continue
        if any(np.linalg.norm(x - y[0]) < dedup_tol for y in kept):      # already found
            continue
        kept.append((x, E_np(x), lam))
    kept.sort(key=lambda t: t[1])
    return kept


def optimal_state_energy(valency, r_low, r_high, n_target, *, r_rep, r_att, E_r, E_a):
    """Energy of the ideal ``valency``-coordinated state, all neighbours at the well centre.

    For a gate whose cap is at least the valency this is ``valency * E_a`` (``valency`` fully
    satisfied bonds), the natural reference for the depth of an over-coordinated trap. The
    trap depth is ``find_minima(...)[0][1] - optimal_state_energy(...)``. negative means the
    extra particle is net bound (over-coordination lies below the ideal state), positive
    means a metastable pocket above it.
    """
    ncut = valency + (n_target - 1.0)
    r_star = 0.5 * (r_rep + r_att)
    return float(cluster_energy(jnp.full(valency, r_star), r_low=r_low, r_high=r_high,
                                ncut=ncut, r_rep=r_rep, r_att=r_att, E_r=E_r, E_a=E_a))

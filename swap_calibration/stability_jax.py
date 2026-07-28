"""Layer 4. Local Hessian stability of neighbour configurations (optional, needs JAX).

For a central particle C with ``m`` neighbours at radial distances ``r_vec`` (C fixed at
the origin, angles fixed), the total cluster energy ``cluster_energy`` is differentiable,
so its Hessian ``d2E/dr_i dr_j`` is an ``m x m`` matrix. Its eigenvalues report the local
curvature. All positive means the arrangement is a locally stable (convex) minimum in the
radial coordinates. We sample this over an ensemble of neighbour arrangements and report a
stochastic stability metric.

Terminology. This is a LOCAL, per-configuration notion (Hessian positive-definite / local
convexity). It is distinct from the paper's macroscopic H-stability (the Fisher-Ruelle
bound on the total energy via phi_min); do not call this "H-stable".

DOF. Radial distances only. In this energy neighbours interact with C but not with each
other, so a full-3D Hessian would carry 2m exact zero modes (rotating a neighbour about C
at fixed radius costs nothing). The radial coordinates are the physically meaningful and
tractable DOF.

Caveat. A sampled or gridded configuration is generally NOT a stationary point
(grad E != 0), so ``p_stable`` here measures the fraction of arrangements that sit in a
locally convex region, not the fraction of relaxed basin minima. ``grad_norm`` is reported
alongside. A strict relaxed-minimum mode (relax to grad ~ 0, then Hessian) is scaffolded
via the ``relax`` flag but not yet implemented.
"""
from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        'stability_jax needs JAX (Layer 4 only). Install with  pip install "jax[cpu]"  '
        '(or "jax[cuda12]" on a GPU node). Layers 1 to 3 in vbc_landscape.py need only numpy.'
    ) from exc

from vbc_landscape_jax import cluster_energy


class StabilityResult(NamedTuple):
    lambda_min: np.ndarray   # (n_config,) smallest Hessian eigenvalue per configuration
    p_stable: float          # fraction of configs with a positive-definite Hessian
    grad_norm: np.ndarray    # (n_config,) |grad E| at the sampled config (0 => stationary)
    logdet_pd: np.ndarray    # log-det of the Hessian over the PD subset (basin softness)
    n_config: int
    m: int


def curvature_scale(E_a, r_rep, r_att):
    """Base-attraction curvature ``|V_att''|`` at the well centre, a scale for lambda_min."""
    return 8.0 * abs(E_a) / (r_att - r_rep) ** 2


def _make_batch(r_low, r_high, ncut, r_rep, r_att, E_r, E_a):
    """Return a jitted, vmapped map  r_vec -> (eigenvalues, |grad|)  for one config."""
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


def _reduce(batch, r_all, chunk):
    """Stream configs through ``batch`` in chunks, keep only per-config scalars."""
    r_all = jnp.asarray(r_all)
    n, m = int(r_all.shape[0]), int(r_all.shape[1])
    lam = np.empty(n)
    gn = np.empty(n)
    logdets = []
    for s in range(0, n, chunk):
        w, g = batch(r_all[s:s + chunk])
        w = np.asarray(w)
        lam[s:s + chunk] = w[:, 0]
        gn[s:s + chunk] = np.asarray(g)
        pd = np.all(w > 0.0, axis=1)
        if pd.any():
            logdets.append(np.sum(np.log(w[pd]), axis=1))
    logdet_pd = np.concatenate(logdets) if logdets else np.empty(0)
    return StabilityResult(lam, float(np.mean(lam > 0.0)), gn, logdet_pd, n, m)


def stability_random(valency, r_low, r_high, n_target, *, r_rep, r_att, E_r, E_a,
                     n_sample=100_000, seed=0, relax=False, chunk=50_000):
    """Local Hessian stability over ``n_sample`` uniform neighbour placements in the
    binding zone ``(r_rep, r_att)``. Returns a ``StabilityResult``."""
    if relax:
        raise NotImplementedError(
            "relax=True (strict relaxed-basin minima) is a documented stub. "
            "The current metric is local convexity at the sampled config (relax=False).")
    ncut = valency + (n_target - 1.0)
    batch = _make_batch(r_low, r_high, ncut, r_rep, r_att, E_r, E_a)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_sample)
    r_all = jax.vmap(
        lambda k: jax.random.uniform(k, (valency,), minval=r_rep, maxval=r_att))(keys)
    return _reduce(batch, r_all, chunk)


def stability_grid(valency, r_low, r_high, n_target, *, r_rep, r_att, E_r, E_a,
                   n_grid=10, r_min=None, r_max=None, dedup=True, chunk=50_000):
    """Local Hessian stability on a radial grid of neighbour distances.

    Neighbours are identical, so two radial tuples differing only by a permutation are the
    SAME physical configuration (same energy, same Hessian spectrum). With ``dedup=True``
    we enumerate sorted multisets of ``m`` grid indices,
    ``combinations_with_replacement(range(n_grid), m)``, giving ``C(n_grid + m - 1, m)``
    unique configs instead of the ``n_grid ** m`` product, a factor of up to ``m!`` fewer
    (e.g. n_grid=10, m=7 gives C(16,7)=11440 versus 1e7). ``dedup=False`` reproduces the
    full product and is intended only as a throughput benchmark.
    """
    ncut = valency + (n_target - 1.0)
    m = valency
    r_min = r_rep if r_min is None else r_min
    r_max = r_att if r_max is None else r_max
    pts = np.linspace(r_min, r_max, n_grid)

    if dedup:
        idx = np.fromiter(
            itertools.chain.from_iterable(itertools.combinations_with_replacement(range(n_grid), m)),
            dtype=np.int64,
        ).reshape(-1, m)
    else:
        idx = np.fromiter(
            itertools.chain.from_iterable(itertools.product(range(n_grid), repeat=m)),
            dtype=np.int64,
        ).reshape(-1, m)

    r_all = pts[idx]
    batch = _make_batch(r_low, r_high, ncut, r_rep, r_att, E_r, E_a)
    return _reduce(batch, r_all, chunk)

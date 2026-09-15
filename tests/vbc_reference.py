# Copyright (c) 2025 Goloborodko Lab.
# Released under the BSD 3-Clause License.

"""Independent float64 reference implementation of the VBC potential.

Implemented directly from ``docs/potential.md`` (smooth per-type neighbour
counts, valency gates on both partners, gated attraction plus intact repulsion,
and the many-body chain-rule force term) — deliberately NOT from the C++ code,
so agreement between the plugin and this module validates both.

Pure numpy, O(N^2); intended for the small test systems in ``conftest.py``.
Also holds the shared test parameter regimes (pure data), so pytest fixtures
and the standalone tools use the exact same configurations.
"""
import numpy as np

PARAM_KEYS = ['E_r', 'E_a', 'attr_rcut', 'rep_rcut', 'density_lower_rcut',
              'density_upper_rcut', 'ncut_ij', 'ncut_ji', 'use_global_dens']


def _omega(t):
    """Quartic switch on t: 1 below 0, (1-t^2)^2 on [0,1], 0 above 1."""
    tc = np.clip(t, 0.0, 1.0)
    return np.where(t <= 0.0, 1.0, np.where(t >= 1.0, 0.0, (1.0 - tc ** 2) ** 2))


def _omega_deriv(t):
    """d/dt of the quartic switch; nonzero only inside (0, 1)."""
    tc = np.clip(t, 0.0, 1.0)
    return np.where((t <= 0.0) | (t >= 1.0), 0.0, -4.0 * tc * (1.0 - tc ** 2))


def _param_matrices(type_names, pair_params):
    """Expand {sorted (a,b) pair: dict} into full (T,T) per-ordered-pair arrays,
    swapping ncut_ij/ncut_ji for the transposed entry as the plugin does."""
    T = len(type_names)
    idx = {n: i for i, n in enumerate(type_names)}
    P = {k: np.zeros((T, T)) for k in PARAM_KEYS}
    for (a, b), d in pair_params.items():
        ia, ib = idx[a], idx[b]
        for k in PARAM_KEYS:
            P[k][ia, ib] = d[k]
            P[k][ib, ia] = d[k]
        P['ncut_ij'][ib, ia] = d['ncut_ji']
        P['ncut_ji'][ib, ia] = d['ncut_ij']
    return P


def vbc_reference(pos, typeid, L, type_names, pair_params, r_cut):
    """Return per-particle energies (N,), forces (N,3), virials (N,6) in float64.

    Virial rows use HOOMD's order: xx, xy, xz, yy, yz, zz. Pairs at or beyond
    r_cut are excluded; every standard parameter set keeps all windows inside
    r_cut, where the plugin agrees exactly."""
    pos = np.asarray(pos, dtype=np.float64)
    N = len(pos)
    ti = np.asarray(typeid)
    T = len(type_names)
    P = _param_matrices(type_names, pair_params)

    def M(key):
        return P[key][ti[:, None], ti[None, :]]

    d = pos[:, None, :] - pos[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    cut = r < r_cut

    lo, hi = M('density_lower_rcut'), M('density_upper_rcut')
    with np.errstate(divide='ignore', invalid='ignore'):
        tw = (r - lo) / (hi - lo)
        inv_w = np.where(hi > lo, 1.0 / (hi - lo), 0.0)
    w = _omega(tw) * cut
    wd = _omega_deriv(tw) * inv_w * cut

    onehot = ti[None, :, None] == np.arange(T)[None, None, :]        # (1,N,T)
    phi = (w[:, :, None] * onehot).sum(axis=1)                        # (N,T)
    phi_tot = phi.sum(axis=1)                                         # (N,)

    glob = M('use_global_dens') != 0
    ii = np.arange(N)
    phi_i_typed = phi[ii[:, None], ti[None, :]]     # [i,j] = phi[i, type_j]
    phi_j_typed = phi[ii[None, :], ti[:, None]]     # [i,j] = phi[j, type_i]
    n_i = np.where(glob, phi_tot[:, None], phi_i_typed)
    n_j = np.where(glob, phi_tot[None, :], phi_j_typed)

    W1 = _omega(n_i - M('ncut_ij'))
    W2 = _omega(n_j - M('ncut_ji'))
    W1d = _omega_deriv(n_i - M('ncut_ij'))
    psi = W1 * W2

    # base pair potential
    E_r, E_a = M('E_r'), M('E_a')
    rep, attr = M('rep_rcut'), M('attr_rcut')
    with np.errstate(divide='ignore', invalid='ignore'):
        t_rep = np.where(rep > 0, r / rep, np.inf)
        inv_rep = np.where(rep > 0, 1.0 / rep, 0.0)
        hw = (attr - rep) / 2.0
        re = (attr + rep) / 2.0
        t_att = np.where(hw > 0, (r - re) / hw, np.inf)
        inv_hw = np.where(hw > 0, 1.0 / hw, 0.0)

    in_rep = (t_rep < 1.0) & cut
    u = 1.0 - np.clip(t_rep, 0, 1) ** 2
    Vr = np.where(in_rep, E_r * u ** 2, 0.0)
    Vr_d = np.where(in_rep, -4.0 * E_r * np.clip(t_rep, 0, 1) * u * inv_rep, 0.0)

    in_att = (r > rep) & (r < attr) & cut
    ua = 1.0 - np.clip(t_att, -1, 1) ** 2
    Va = np.where(in_att, E_a * ua ** 2, 0.0)
    Va_d = np.where(in_att, -4.0 * E_a * np.clip(t_att, -1, 1) * ua * inv_hw, 0.0)

    # per-particle energies: E_i = 0.5 sum_j (Vr + psi Va)
    E_pair = Vr + psi * Va
    energies = 0.5 * E_pair.sum(axis=1)

    # many-body sums S[i, b] = sum_{j of type b} g'(n_i) g(n_j) Va(r_ij)
    contrib = W1d * W2 * Va
    S = (contrib[:, :, None] * onehot).sum(axis=1)                    # (N,T)
    S_tot = S.sum(axis=1)
    S_i_pick = np.where(glob, S_tot[:, None], S[ii[:, None], ti[None, :]])
    S_j_pick = np.where(glob, S_tot[None, :], S[ii[None, :], ti[:, None]])

    dedr = Vr_d + psi * Va_d + wd * (S_i_pick + S_j_pick)
    with np.errstate(divide='ignore', invalid='ignore'):
        fdr = np.where(cut, -dedr / r, 0.0)                           # force/r on i from j

    forces = (d * fdr[:, :, None]).sum(axis=1)

    virials = np.stack([
        0.5 * (d[..., 0] * d[..., 0] * fdr).sum(axis=1),
        0.5 * (d[..., 0] * d[..., 1] * fdr).sum(axis=1),
        0.5 * (d[..., 0] * d[..., 2] * fdr).sum(axis=1),
        0.5 * (d[..., 1] * d[..., 1] * fdr).sum(axis=1),
        0.5 * (d[..., 1] * d[..., 2] * fdr).sum(axis=1),
        0.5 * (d[..., 2] * d[..., 2] * fdr).sum(axis=1),
    ], axis=1)

    return energies, forces, virials


# ---------------------------------------------------------------------------
# shared test parameter regimes (pure data; used by conftest and the tools)
# ---------------------------------------------------------------------------

BASE = dict(E_r=20.0, rep_rcut=1.0, attr_rcut=1.5,
            density_lower_rcut=1.5, density_upper_rcut=2.0)
R_CUT = 2.0

_INERT0 = dict(BASE, E_a=0.0, ncut_ij=1.0, ncut_ji=1.0, use_global_dens=0)


def _global_uniform():
    """One type, global cap 2 — the standard smoke system, fairly dense."""
    types = ['A']
    pp = {('A', 'A'): dict(BASE, E_a=-4.0, ncut_ij=2.0, ncut_ji=2.0,
                           use_global_dens=1)}
    return types, pp, dict(N=400, rho=0.4)


def _local_asymmetric():
    """Three types, per-type mode, asymmetric caps, one inert pair."""
    types = ['A', 'B', 'C']
    pp = {
        ('A', 'C'): dict(BASE, E_a=-4.0, ncut_ij=1.0, ncut_ji=3.0, use_global_dens=0),
        ('B', 'C'): dict(BASE, E_a=-4.0, ncut_ij=1.0, ncut_ji=3.0, use_global_dens=0),
        ('A', 'B'): dict(BASE, E_a=-2.0, ncut_ij=4.0, ncut_ji=1.0, use_global_dens=0),
        ('A', 'A'): _INERT0,
        ('B', 'B'): dict(BASE, E_a=-3.0, ncut_ij=2.0, ncut_ji=2.0, use_global_dens=0),
        ('C', 'C'): _INERT0,
    }
    return types, pp, dict(N=500, rho=0.5)


def _global_mixed_caps():
    """Three types, global mode, couple-selective caps, dense so gates engage."""
    types = ['A', 'B', 'C']
    pp = {}
    for pair in [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'C')]:
        pp[pair] = dict(BASE, E_a=-4.0, ncut_ij=4.0, ncut_ji=4.0, use_global_dens=1)
    pp[('B', 'B')] = dict(BASE, E_a=-4.0, ncut_ij=2.0, ncut_ji=2.0, use_global_dens=1)
    return types, pp, dict(N=500, rho=0.7)


CONFIGS = {
    'global-uniform': _global_uniform,
    'local-asymmetric': _local_asymmetric,
    'global-mixed-caps': _global_mixed_caps,
}

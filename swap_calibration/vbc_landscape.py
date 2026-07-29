from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Base potential and gate (match the installed VBC evaluator)
# ---------------------------------------------------------------------------

def density_phi(r: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Neighbour-counting weight. 1 below ``lo``, a smooth shoulder to 0 at ``hi``."""
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    out = np.where(r < lo, 1.0, out)
    mid = (r >= lo) & (r <= hi)
    if np.any(mid):
        s = (r[mid] - lo) / (hi - lo)
        out[mid] = (1.0 - s ** 2) ** 2
    return out


def omega_gate(phi: np.ndarray, ncut: float) -> np.ndarray:
    """Valency gate. 1 below ``ncut``, a smooth shoulder to 0 at ``ncut + 1``."""
    phi = np.asarray(phi, dtype=float)
    out = np.zeros_like(phi)
    out = np.where(phi < ncut, 1.0, out)
    mid = (phi >= ncut) & (phi <= ncut + 1.0)
    if np.any(mid):
        s = phi[mid] - ncut
        out[mid] = (1.0 - s ** 2) ** 2
    return out


def V_rep(r: np.ndarray, r_rep: float, E_r: float) -> np.ndarray:
    """Soft-core repulsion on ``[0, r_rep]``."""
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    m = (r >= 0.0) & (r <= r_rep)
    if np.any(m):
        z = r[m] / r_rep
        out[m] = E_r * (1.0 - z ** 2) ** 2
    return out


def V_att(r: np.ndarray, r_rep: float, r_att: float, E_a: float) -> np.ndarray:
    """Attractive well on ``(r_rep, r_att)`` with its minimum at the midpoint."""
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    m = (r > r_rep) & (r < r_att)
    if np.any(m):
        c = 0.5 * (r_att + r_rep)
        hw = 0.5 * (r_att - r_rep)
        z = (r[m] - c) / hw
        out[m] = E_a * (1.0 - z ** 2) ** 2
    return out



def swap_landscape(valency: int, r_grid: np.ndarray,
                   r_low: float, r_high: float, n_target: float,
                   *, r_rep: float, r_att: float,
                   E_r: float, E_a: float) -> np.ndarray:
    """Sum of the central particle's bonds on the ``(r_A, r_B)`` grid."""
    ncut = valency + (n_target - 1.0)
    n_spec = valency - 1
    r_star = 0.5 * (r_rep + r_att)

    rho_rs = float(density_phi(np.array([r_star]), r_low, r_high)[0])
    vatt_rstar = float(V_att(np.array([r_star]), r_rep, r_att, E_a)[0])

    RA, RB = np.meshgrid(r_grid, r_grid, indexing="ij")
    rho_A = density_phi(RA, r_low, r_high)
    rho_B = density_phi(RB, r_low, r_high)

    phi_C = n_spec * rho_rs + rho_A + rho_B
    W_C = omega_gate(phi_C, ncut)
    W_A = omega_gate(rho_A, ncut)
    W_B = omega_gate(rho_B, ncut)
    W_S = float(omega_gate(np.array([rho_rs]), ncut)[0])

    U = (V_rep(RA, r_rep, E_r) + W_C * W_A * V_att(RA, r_rep, r_att, E_a)
         + V_rep(RB, r_rep, E_r) + W_C * W_B * V_att(RB, r_rep, r_att, E_a)
         + n_spec * W_C * W_S * vatt_rstar)
    return U


def naive_spurious_depth(valency: int, r_low: float, r_high: float, n_target: float,
                         *, r_rep: float, r_att: float, E_r: float, E_a: float,
                         n_r: int = 501) -> float:

    r_arr = np.linspace(r_rep, r_att, n_r)
    ncut = valency + (n_target - 1.0)
    r_star = 0.5 * (r_rep + r_att)

    rho_rs = float(density_phi(np.array([r_star]), r_low, r_high)[0])
    rho_r = density_phi(r_arr, r_low, r_high)

    phi_C = valency * rho_rs + rho_r
    phi_N = rho_r
    W_C = omega_gate(phi_C, ncut)
    W_N = omega_gate(phi_N, ncut)

    # bare C-N pair well only, no back-reaction on the existing bonds, baseline zero
    U = V_rep(r_arr, r_rep, E_r) + W_C * W_N * V_att(r_arr, r_rep, r_att, E_a)

    U_int = U[1:-1]
    if len(U_int) == 0:
        return 0.0
    i = int(np.argmin(U_int)) + 1
    if U[i] < U[i - 1] and U[i] < U[i + 1] and U[i] < 0.0:
        return -float(U[i]) / abs(E_a)
    return 0.0


def sporadic_min_depth(valency: int, r_low: float, r_high: float, n_target: float,
                       *, r_rep: float, r_att: float, E_r: float, E_a: float,
                       n_r: int = 501) -> float:
    """Depth of the off-channel minimum, as a fraction of ``|E_a|``.

    A single particle approaches a central particle whose shell of ``valency``
    partners is already full, so the gate switches the attraction on and then off
    again as the count crosses the cap. Any interior minimum of that one-dimensional
    profile is the spurious trap.
    """
    r_arr = np.linspace(r_rep, r_att, n_r)
    ncut = valency + (n_target - 1.0)
    r_star = 0.5 * (r_rep + r_att)

    rho_rs = float(density_phi(np.array([r_star]), r_low, r_high)[0])
    vatt_rs = float(V_att(np.array([r_star]), r_rep, r_att, E_a)[0])
    W_S = float(omega_gate(np.array([rho_rs]), ncut)[0])
    rho_r = density_phi(r_arr, r_low, r_high)

    phi_C = valency * rho_rs + rho_r
    phi_N = rho_r
    W_C = omega_gate(phi_C, ncut)
    W_N = omega_gate(phi_N, ncut)

    # total cluster energy -  the C-N bond plus the back-reaction on C's valency existing bonds
    U = (V_rep(r_arr, r_rep, E_r) + W_C * W_N * V_att(r_arr, r_rep, r_att, E_a)
         + valency * W_C * W_S * vatt_rs)
    baseline = valency * float(omega_gate(np.array([valency * rho_rs]), ncut)[0]) * W_S * vatt_rs

    U_int = U[1:-1]
    if len(U_int) == 0:
        return 0.0
    i = int(np.argmin(U_int)) + 1
    if U[i] < U[i - 1] and U[i] < U[i + 1] and U[i] < baseline:
        return (baseline - float(U[i])) / abs(E_a)
    return 0.0


def spurious_depth(valency: int, r_low: float, r_high: float, n_target: float,
                   *, r_rep: float, r_att: float, E_r: float, E_a: float,
                   key: str = "deterministic", n_sample: int = 100, seed: int = 0,
                   n_r: int = 501) -> float:
    """Spurious depth with the passive neighbours placed by ``key``.

    ``key="deterministic"`` puts all ``valency`` passive neighbours at the well centre
    and returns ``sporadic_min_depth``. ``key="random"`` draws ``valency`` neighbour
    distances uniformly in the binding zone ``(r_rep, r_att)`` for each of ``n_sample``
    placements, so each contributes its own weight to ``phi_C`` and its own bond to the
    back-reaction, and returns the depth averaged over the placements.
    """
    if key == "deterministic":
        return sporadic_min_depth(valency, r_low, r_high, n_target,
                                  r_rep=r_rep, r_att=r_att, E_r=E_r, E_a=E_a, n_r=n_r)
    if key != "random":
        raise ValueError("key must be 'deterministic' or 'random'")

    rng = np.random.default_rng(seed)
    r_arr = np.linspace(r_rep, r_att, n_r)
    ncut = valency + (n_target - 1.0)
    depths = []
    for _ in range(n_sample):
        r_spec = rng.uniform(r_rep, r_att, size=valency)
        w_spec = density_phi(r_spec, r_low, r_high)
        S = float(np.sum(omega_gate(w_spec, ncut) * V_att(r_spec, r_rep, r_att, E_a)))
        sum_w = float(w_spec.sum())

        rho_r = density_phi(r_arr, r_low, r_high)
        W_C = omega_gate(sum_w + rho_r, ncut)
        W_N = omega_gate(rho_r, ncut)
        U = V_rep(r_arr, r_rep, E_r) + W_C * W_N * V_att(r_arr, r_rep, r_att, E_a) + W_C * S
        baseline = float(omega_gate(np.array([sum_w]), ncut)[0]) * S

        U_int = U[1:-1]
        if len(U_int) == 0:
            depths.append(0.0)
            continue
        i = int(np.argmin(U_int)) + 1
        if U[i] < U[i - 1] and U[i] < U[i + 1] and U[i] < baseline:
            depths.append((baseline - float(U[i])) / abs(E_a))
        else:
            depths.append(0.0)
    return float(np.mean(depths))



class UnionFind:
    """Disjoint-set with path compression and union by rank (Tarjan 1975)."""

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _neighbors_8(i, j, n1, n2):
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < n1 and 0 <= nj < n2:
                yield ni, nj


def minimax_barrier(U: np.ndarray,
                    start: Tuple[int, int],
                    goal: Tuple[int, int]) -> float:
    """Lowest energy level connecting ``start`` and ``goal`` by flooding (Hu 1961)."""
    n1, n2 = U.shape
    s = start[0] * n2 + start[1]
    g = goal[0] * n2 + goal[1]

    Uflat = U.reshape(-1)
    order = np.argsort(Uflat, kind="mergesort")
    active = np.zeros(Uflat.size, dtype=bool)
    uf = UnionFind(Uflat.size)

    for idx in order:
        idx = int(idx)
        active[idx] = True
        i, j = divmod(idx, n2)
        for ni, nj in _neighbors_8(i, j, n1, n2):
            nidx = ni * n2 + nj
            if active[nidx]:
                uf.union(idx, int(nidx))
        if active[s] and active[g] and uf.find(int(s)) == uf.find(int(g)):
            return float(Uflat[idx])

    raise RuntimeError("Start and goal never connected.")


def find_swap_endpoints(U: np.ndarray, r: np.ndarray, *,
                        x_thresh: float = 2.0
                        ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """The two global minima, one with ``r_A`` large (B bonded) and one with ``r_B`` large."""
    eps = 1e-12 * max(1.0, float(np.nanmax(np.abs(U))))
    Umin = float(np.nanmin(U))
    mask = np.isclose(U, Umin, rtol=0.0, atol=eps)

    maskA = mask & (r[:, None] >= x_thresh)
    maskB = mask & (r[None, :] >= x_thresh)
    if not (maskA.any() and maskB.any()):
        return None

    A = np.argwhere(maskA)
    B = np.argwhere(maskB)
    start = (int(A[0, 0]), int(A[0, 1]))
    goal = (int(B[0, 0]), int(B[0, 1]))

    if start == goal:
        for k in range(1, B.shape[0]):
            cand = (int(B[k, 0]), int(B[k, 1]))
            if cand != start:
                goal = cand
                break
        else:
            for k in range(1, A.shape[0]):
                cand = (int(A[k, 0]), int(A[k, 1]))
                if cand != goal:
                    start = cand
                    break
        if start == goal:
            return None
    return start, goal


def channel_metrics(U: np.ndarray, r_grid: np.ndarray,
                    start: Tuple[int, int], goal: Tuple[int, int],
                    E_a: float, eps_abs: float = 0.01):
    """Return ``(barrier_rel, barrier_rel/|E_a|, channel_width, well_depth)``."""
    barrier_abs = minimax_barrier(U, start, goal)
    U_start, U_goal = float(U[start]), float(U[goal])
    barrier_rel = barrier_abs - max(U_start, U_goal)
    dr = r_grid[1] - r_grid[0]
    width = float(np.sum(U <= barrier_abs + eps_abs)) * dr * dr
    well = min(U_start, U_goal)
    return barrier_rel, barrier_rel / abs(E_a), width, well

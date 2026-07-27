# VBC potential

The valency-bounding correction (VBC) turns an ordinary short-range attraction into a
saturable one. It multiplies the attractive part of a base pair potential by a smooth
switching function that depends on how many neighbours each of the two interacting
particles already has. Once a particle reaches its prescribed number of attractive
neighbours, its valency, the gate closes, and adding further neighbours increases the potential energy of the system.

This note gives the energy definition. It covers the base potential, the smooth neighbour count, the valency gate, the corrected pair energy, and the type-resolved and single
global-density forms.

## 1. The base pair potential

The uncorrected interaction is a polynomial soft core, a finite repulsive bump plus a
short attractive well,

$$V(r)=V_r(r)+V_a(r).$$

The repulsive core is non-negative and supported on $(0,r_r)$,

```math
V_r(r)=\begin{cases}E_r\left[1-\left(\dfrac{r}{r_r}\right)^{2}\right]^{2}, & r<r_r,\\ 0, & r\ge r_r,\end{cases}
```

a soft wall of height $V_r(0)=E_r$ that vanishes smoothly at $r=r_r$. The attractive
well is non-positive and supported on $(r_r,r_a)$,

```math
V_a(r)=\begin{cases}E_a\left[1-\left(\dfrac{r-r_e}{(r_a-r_r)/2}\right)^{2}\right]^{2}, & r_r<r<r_a,\\ 0, & \text{otherwise},\end{cases}\qquad r_e=\frac{r_r+r_a}{2},
```

with its minimum $V_a(r_e)=E_a$ at the well centre $r_e$. The symbols are as follows.

- $E_r>0$, repulsion wall, the height of the soft core (standard value $20\ k_BT$).
- $r_r$, repulsion range, where the core ends and the well begins (standard $1$, the
  unit of length).
- $E_a$, attraction depth. A well requires $E_a<0$, and $|E_a|$ is the tunable binding
  strength.
- $r_a$, attraction range, the outer edge of the well (standard $1.5$).

Energies are in $k_BT$ and lengths in units of $r_r=1$.

## 2. The switching function

A single twice-differentiable step is reused throughout. It decreases smoothly from $1$ to
$0$ as its argument crosses from $x_1$ to $x_2$.

```math
\omega(x;x_1,x_2)=\begin{cases}1, & x\le x_1,\\ \left[1-\left(\dfrac{x-x_1}{x_2-x_1}\right)^{2}\right]^{2}, & x_1<x<x_2,\\ 0, & x\ge x_2.\end{cases}
```

It appears twice with different arguments, once to count neighbours and once to cap the count of neighbours.

## 3. The neighbour count (density)

The local density of particle $i$ is a smooth neighbour count. Resolved by
neighbour type $\beta$,

$$n_i^{\beta}=\sum_{j\ne i} w_{ij}\delta_{\tau_j}^{\beta},\qquad w_{ij}=\omega\left(r_{ij};r_\ell,r_h\right).$$

- $w_{ij}$, the density weight of neighbour $j$, equal to $1$ when $j$ is closer than $r_\ell$,
  a smooth fraction in the shell $r_\ell<r_{ij}<r_h$, and $0$ beyond $r_h$.
- $\tau_j$, the type of particle $j$. The Kronecker symbol $\delta_{\tau_j}^{\beta}$
  keeps only type-$\beta$ neighbours, so $n_i^{\beta}$ is the smoothed number of
  type-$\beta$ neighbours of $i$.
- $r_\ell,\ r_h$, the lower and upper density-shell radii. These are independent
  parameters, not tied to the attraction range $r_a$, with standard values $r_\ell=1.5$ and $r_h=2.0$.

The global density is the sum over types, the total neighbour count
regardless of type,

$$n_i=\sum_{\beta} n_i^{\beta}=\sum_{j\ne i} w_{ij}.$$

Both $n_i^{\beta}$ and $n_i$ are per particle and local. $n_i$ is the total in $i$'s own
neighbour shell, not a box- or system-averaged density.

## 4. The valency gate

The count feeds a gate built from the same switch, with the cap $\nu$.

$$g(n;\nu)=\omega(n;\nu,\nu+1),$$

so $g=1$ while the count is below the cap $\nu$, falls smoothly to $0$ as the count crosses
from $\nu$ to $\nu+1$, and is $0$ above. The cap $\nu$ is the prescribed valency, the
number of attractive neighbours a particle is allowed.

An $i$–$j$ attraction is gated by the product of both particles' gates,

$$\Psi_{ij}=g\left(n_i^{\tau_j};\nu^{\tau_i\tau_j}\right)g\left(n_j^{\tau_i};\nu^{\tau_j\tau_i}\right)\in[0,1].$$

- $g\left(n_i^{\tau_j};\nu^{\tau_i\tau_j}\right)$ gates $i$ on its own count of the
  partner's type, against $i$'s cap $\nu^{\tau_i\tau_j}$.
- $g\left(n_j^{\tau_i};\nu^{\tau_j\tau_i}\right)$ gates $j$ likewise, against the
  cap $\nu^{\tau_j\tau_i}$.
- The caps may be asymmetric, $\nu^{\tau_i\tau_j}\neq\nu^{\tau_j\tau_i}$, for example a
  receptor that binds many ligands while each ligand binds one receptor.

Because the two gates multiply, the attraction survives only if both particles still
have room under their caps.

## 5. The corrected potential and total energy

The correction leaves the repulsion untouched and scales only the attractive well by the
gate.

$$\hat V_{ij}=V_r(r_{ij})+\Psi_{ij}V_a(r_{ij}),\qquad \mathcal V=\sum_{i<j}\hat V_{ij}.$$

Since $\Psi_{ij}\in[0,1]$ the attraction can only be weakened. A
saturated particle gains no further energy by acquiring an extra attractive neighbour. In contrast, adding neighbours beyond the cap increases the potential energy.

## 6. The single global-density form

Replacing the type-resolved counts by the one total count $n_i$ gives the
global-density variant.

$$n_i=\sum_{j\ne i} w_{ij},\qquad \Psi_{ij}=g\left(n_i;\nu^{\tau_i\tau_j}\right)g\left(n_j;\nu^{\tau_j\tau_i}\right).$$

Each gate now reads the particle's total neighbour count,
still compared against the per-couple cap $\nu^{\tau_i\tau_j}$, which may remain type-pair
specific.

This is the `use_global_dens=1` mode. `use_global_dens=0` selects the per-type form.

## Parameters and standard values

| ESI symbol           | `params` key         | meaning                                   | standard value |
| -------------------- | -------------------- | ----------------------------------------- | -------------- |
| $E_r$                | `E_r`                | repulsion depth                           | $20\ k_BT$     |
| $r_r$                | `rep_rcut`           | repulsion range                           | $1$            |
| $E_a$                | `E_a`                | attraction depth                          | tunable        |
| $r_a$                | `attr_rcut`          | attraction range                          | $1.5$          |
| $r_\ell$             | `density_lower_rcut` | lower density-shell radius                | $1.5$          |
| $r_h$                | `density_upper_rcut` | upper density-shell radius                | $2.0$          |
| $\nu^{\tau_i\tau_j}$ | `ncut_ij`            | home valency cap                          | set per type   |
| $\nu^{\tau_j\tau_i}$ | `ncut_ji`            | mirror valency cap                        | set per type   |
| —                    | `use_global_dens`    | per-type (`0`) or global/total (`1`) density | `0` / `1`   |

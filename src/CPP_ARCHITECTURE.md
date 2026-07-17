# VBC Plugin — C++ Architecture & Physics Documentation

This document preserves the knowledge that was encoded in the inline comments and
code of the VBC (Valency-capped soft density Bonded/pair Coordination) HOOMD-blue
plugin. It is the authoritative reference **after** the inline comments are
stripped from the source. Everything below is derived directly from the source
files listed at the end.

The plugin implements a soft, density-dependent, valency-capped pair potential,
built on the same infrastructure HOOMD uses for three-body (Tersoff-style)
potentials. Despite the "three-body" scaffolding, the VBC evaluator itself is
effectively a *density-modulated pair* potential: it never computes an explicit
`ik` force term.

---

## 1. Overview / call graph

### File roles

| File | Role |
|------|------|
| `module.cc` | pybind11 module `_vbc_plugin`. Exports the two compiled classes. |
| `EvaluatorVBC.h` | The physics. `param_type` + all `eval*` methods. Templated evaluator plugged into the generic driver. |
| `PotentialValency.h` | CPU `ForceCompute` template. Two-pass (density then force) neighbor-loop driver. |
| `PotentialValencyGPU.h` | GPU `ForceCompute` subclass of `PotentialValency`. Adds an `Autotuner` and launches the CUDA/HIP kernels. |
| `PotentialValencyGPU.cuh` | GPU kernels: `gpu_compute_density_kernel` (phase 1) and `gpu_compute_valency_forces_kernel` (phase 2), plus the templated launcher. |
| `PotentialValencyVBCGPUKernel.cu` | Explicit template instantiation of `gpu_compute_valency_forces<EvaluatorVBC>` so the GPU driver links. |

### What the compiled classes are

The Python-visible class names come from explicit instantiations of the generic
templates with `EvaluatorVBC`:

```cpp
// module.cc
void export_PotentialVBC(pybind11::module& m)
    {
    export_PotentialValency<EvaluatorVBC>(m, "PotentialVBC2");        // CPU
    }

#ifdef ENABLE_HIP
void export_PotentialVBCGPU(pybind11::module& m)
    {
    export_PotentialValencyGPU<EvaluatorVBC>(m, "PotentialVBC2GPU");  // GPU
    }
#endif
```

So, exactly:

- **`PotentialVBC2`**  ==  `hoomd::md::PotentialValency<EvaluatorVBC>`
- **`PotentialVBC2GPU`** == `hoomd::md::PotentialValencyGPU<EvaluatorVBC>`

The pybind11 module is `_vbc_plugin`. The GPU export is compiled/registered only
when `ENABLE_HIP` is defined. The GPU class inherits from the CPU class in
pybind11 (`PotentialValencyGPU<T>` derived from `PotentialValency<T>`), so it
reuses all the parameter-setting Python methods.

### Runtime call graph

```
Python md.pair-like wrapper
   │  setParams((typeA,typeB), {...}) / setRCut(...)
   ▼
PotentialValency<EvaluatorVBC>            (CPU)         PotentialValencyGPU<EvaluatorVBC>   (GPU)
   │  computeForces(timestep)                              │  computeForces(timestep)
   │    ├─ m_nlist->compute()                              │    ├─ m_nlist->compute()
   │    ├─ PASS 1: fill m_phi_ab_arr density               │    ├─ Autotuner picks block_size, tpp
   │    │    (evalPhi per neighbor)                         │    └─ kernel::gpu_compute_valency_forces<EvaluatorVBC>(...)
   │    └─ PASS 2: forces                                   │           ├─ Kernel A: gpu_compute_density_kernel   (evalPhi)
   │         (evalRepulsive/evalAttractive/                 │           └─ Kernel B: gpu_compute_valency_forces_kernel
   │          evalOmega/evalOmegaji/evalPhiDiv/             │                        (evalRepulsive/evalAttractive/
   │          evalForceij)                                  │                         evalOmega*/evalPhiDiv/evalForceij)
   ▼                                                        ▼
EvaluatorVBC  (physics: param_type + eval* methods, shared by CPU and GPU)
```

The evaluator is header-only and compiled for both host and device (guarded by
`__HIPCC__` / `DEVICE` / `HOSTDEVICE` macros), so the *identical* physics runs on
CPU and GPU.

> Note: `PotentialValency.h` still uses the legacy include guard
> `__POTENTIAL_TERSOFF_H__` — a vestige of being copied from HOOMD's Tersoff
> three-body driver. It is a driver for VBC despite the name.

---

## 2. The physics model

The VBC potential is a **pair** interaction whose attractive part is gated by a
**valency cap** derived from each particle's **local coordination density**.

For a pair `(i, j)` at separation `r = rij` the per-pair energy contribution is:

```
V_pair(i,j) = 0.5 * ( V_rep(r) + W1 * W2 * V_attr(r) )
```

where:

- `V_rep(r)` is a short-range soft **repulsion** (amplitude `E_r`, cutoff `rep_rcut`).
- `V_attr(r)` is a soft **attractive well** of depth `E_a`, non-zero only in the
  shell `rep_rcut < r < attr_rcut`.
- `W1 = Omega(phi_i)` and `W2 = Omega_ji(phi_j)` are **valency gates** in `[0,1]`
  that switch the attraction off once a particle's local density exceeds its cap.
- `phi_i` / `phi_j` are the local coordination **densities** of `i` and `j`,
  computed by summing a smooth kernel `evalPhi` over neighbors inside the density
  shell `[density_lower_rcut, density_upper_rcut]`.

The force also carries an extra "back-reaction" term coming from the density
dependence of the gates (see `evalForceij` and the `s_ij` bookkeeping in §5).

### Repulsion — `evalRepulsive`

Non-zero only for `rij < rep_rcut`. A smooth quartic soft sphere:

```
V_rep(r) = E_r * ( 1 - (r / rep_rcut)^2 )^2                      for r < rep_rcut
         = 0                                                    otherwise

dV_rep/dr = 2 * E_r * ( 1 - (r/rep_rcut)^2 ) * ( -2 * r / rep_rcut^2 )
```

### Attraction — `evalAttractive`

A soft well of depth `E_a`, centered in the shell `(rep_rcut, attr_rcut)`. Let the
well center and half-width be:

```
re    = (rep_rcut + attr_rcut) / 2
sigma = (attr_rcut - rep_rcut) / 2
```

Then for `rep_rcut < rij < attr_rcut`:

```
V_attr(r) = E_a * ( 1 - ((r - re)/sigma)^2 )^2                   (peak value E_a at r = re)

dV_attr/dr = 2 * E_a * ( 1 - ((r-re)/sigma)^2 ) * ( -2 * (r-re)/sigma^2 )
```

Outside `(rep_rcut, attr_rcut)` the attraction and its derivative are 0.

### Local density kernel — `evalPhi`

Each neighbor `j` of `i` contributes a smooth "occupancy" to `i`'s local density,
non-zero inside the density shell `[density_lower_rcut, density_upper_rcut]`
(abbreviated `rl`, `ru`):

```
phi_contrib(r) = 1                                             for r < rl
               = ( 1 - ((r - rl)/(ru - rl))^2 )^2              for rl <= r <= ru
               = 0                                             for r > ru
```

`phi_i = sum over neighbors j of phi_contrib(rij)`. Its `r`-derivative is
`evalPhiDiv`:

```
dphi/dr = 0                                                    for r < rl or r > ru
        = 2 * ( 1 - ((r-rl)/(ru-rl))^2 ) * ( -2 * (r-rl)/(ru-rl)^2 )   for rl <= r <= ru
```

### Valency gate — `evalOmega` / `evalOmegaji`

The gate is a smooth step that is 1 below the cap, ramps down over a width of 1
density unit, and is 0 above `cap + 1`. `evalOmega` uses cap `ncut_ij`,
`evalOmegaji` uses cap `ncut_ji`. With `local_phi` being either the per-type
density or the global density (selected by `use_global_dens`, see below):

```
Omega(local_phi) = 1                                           for local_phi < ncut
                 = ( 1 - (local_phi - ncut)^2 )^2              for ncut <= local_phi <= ncut+1
                 = 0                                           for local_phi > ncut+1
```

Derivatives `evalOmegaDiv` / `evalOmegajiDiv` (w.r.t. `local_phi`):

```
dOmega/dphi = 0                                                outside [ncut, ncut+1]
            = 2 * ( 1 - (local_phi - ncut)^2 ) * ( -2 * (local_phi - ncut) )   inside
```

The gate width is fixed at 1 density unit (`ncut+1` in the code); it is not a
tunable parameter.

### `use_global_dens` switch (per-type vs total density)

`use_global_dens` is a `Scalar` used as a boolean. It selects which density the
valency gates and the back-reaction force term see:

- **`use_global_dens == 0` (per-type):** the gate reads the *per-type* density,
  i.e. the count of neighbors of the *other* particle's type only. On CPU:
  `evalOmega` gets `phi_ab_arr[i, typej]`, `evalOmegaji` gets
  `phi_ab_arr[j, typei]`.
- **`use_global_dens != 0` (total/global):** the gate reads the *global* density
  = sum over all neighbor types = the extra "global density" column
  (`phi_global`). This is passed as the third argument to every `evalOmega*`.

Inside the evaluator: `Scalar local_phi = (use_global_dens) ? phi_global : phi;`.
The same switch is applied to the force back-reaction term in `evalForceij` via
`local_s_ij = (use_global_dens) ? s_ij_sum : s_ij`.

---

## 3. `param_type` fields

`param_type` is a POD struct, aligned to 8 bytes for 32-bit `Scalar` and 16 bytes
for 64-bit `Scalar`. It is constructed from a Python `dict` and can round-trip
back to one via `asDict()`. Every field maps 1:1 to a key in the Python params
dict:

| Field | Python key | Meaning |
|-------|-----------|---------|
| `E_r` | `"E_r"` | Repulsion amplitude (energy scale of the soft-sphere core). |
| `E_a` | `"E_a"` | Attractive well depth (peak attractive energy at `re`). |
| `attr_rcut` | `"attr_rcut"` | Outer cutoff of the attractive shell. Attraction lives in `(rep_rcut, attr_rcut)`. |
| `rep_rcut` | `"rep_rcut"` | Repulsion cutoff, and inner edge of the attractive shell. |
| `density_upper_rcut` | `"density_upper_rcut"` | Outer cutoff `ru` of the density kernel `evalPhi`. |
| `density_lower_rcut` | `"density_lower_rcut"` | Inner cutoff `rl` of the density kernel (`phi = 1` below it). |
| `ncut_ij` | `"ncut_ij"` | Valency cap seen by particle `i` (used by `evalOmega`). |
| `ncut_ji` | `"ncut_ji"` | Valency cap seen by particle `j` (used by `evalOmegaji`). |
| `use_global_dens` | `"use_global_dens"` | 0 → per-type density gating; non-zero → total (global) density gating. Stored as `Scalar`, used as bool. |

The parsing/serialization is symmetric:

```cpp
param_type(pybind11::dict v) { E_r = v["E_r"].cast<Scalar>(); ...; use_global_dens = v["use_global_dens"].cast<Scalar>(); }
pybind11::dict asDict()      { v["E_r"] = E_r; ...; v["use_global_dens"] = use_global_dens; return v; }
```

Note that the neighbor-list cutoff (`r_cut`) is **not** a `param_type` field; it
is set separately through `setRCut` and fed to the neighbor list.

### `ncut_ij` / `ncut_ji` asymmetry and the mirror-slot swap

`setParams` / `setParamsPython` store the params for `(typ1, typ2)` and mirror
them to `(typ2, typ1)`. For asymmetric pairs (`typ1 != typ2`) the two directional
caps are **swapped** in the mirror slot:

```cpp
param_swapped.ncut_ij = param.ncut_ji;
param_swapped.ncut_ji = param.ncut_ij;
```

This guarantees that whichever way the neighbor list visits a pair, the "home"
gate `ncut_ij` always corresponds to the home particle's own cap, so the computed
force is the exact gradient of the (asymmetric-cap) energy and Newton's 3rd law
holds.

---

## 4. `EvaluatorVBC` method-by-method

Constructor: `EvaluatorVBC(Scalar _rij_sq, const param_type& _params)` caches
`rij_sq` and copies every parameter into member fields.

| Method | What it does |
|--------|--------------|
| `setRij(Scalar rsq)` | Sets `rij_sq = rsq`. Lets the driver reuse one evaluator for multiple pair distances. |
| `setRik(Scalar rsq)` | **Intended** to set the `i-k` squared distance. **BUG:** it assigns `rij_sq = rsq` (writes the `ij` member, not an `ik` member). Harmless in practice because no `ik` code path is active. See §7. |
| `hasPerParticleEnergy()` | `static`, returns `true`. Drives the **density precompute pass** (both CPU and GPU run phase-1 `evalPhi` only because this is true). It does **not** add an explicit per-particle energy term. |
| `needsChi()` | `static`, returns `false`. No per-pair `chi` accumulation. |
| `hasIkForce()` | `static`, returns `false`. No explicit `i-k` three-body force. |
| `needsAngle()` | `static`, returns `false`. No bond-angle dependence. |
| `setAngle(Scalar)` | No-op (angles unused). |
| `areInteractive()` | Returns `true`. |
| `evalChi(Scalar&)` | No-op (chi unused). |
| `evalPhi(Scalar& phi)` | Adds one neighbor's density contribution: the smooth `(1 - ((r-rl)/(ru-rl))^2)^2` kernel, clamped to 1 below `rl` and 0 above `ru`. See §2. |
| `evalPhiDiv(Scalar& phi)` | Adds the `r`-derivative of `evalPhi`; 0 outside the density shell. Used for the density gradient force term `Wrij_div`. |
| `evalOmega(Scalar& W, phi, phi_global)` | Adds the valency gate for `i` using cap `ncut_ij`. Reads `phi` or `phi_global` per `use_global_dens`. |
| `evalOmegaji(Scalar& W, phi, phi_global)` | Same gate but with cap `ncut_ji` (the `j`-side / reverse-direction gate). |
| `evalOmegaDiv(Scalar& W, phi, phi_global)` | `d(Omega)/d(local_phi)` for the `ncut_ij` gate; 0 outside `[ncut_ij, ncut_ij+1]`. |
| `evalOmegajiDiv(Scalar& W, phi, phi_global)` | `d(Omega)/d(local_phi)` for the `ncut_ji` gate. (Defined; used in the CPU density-back-reaction accumulation via `evalOmegaDiv`, its `ji` counterpart present for symmetry.) |
| `evalAttractive(Scalar& attr_eng, Scalar& attr_eng_div)` | Adds the soft attractive well of depth `E_a` and its `r`-derivative; non-zero only for `rep_rcut < r < attr_rcut`. See §2. |
| `evalRepulsive(Scalar& rep_eng, Scalar& rep_eng_div)` | Adds the soft repulsion amplitude `E_r` and its `r`-derivative; non-zero only for `r < rep_rcut`. See §2. |
| `evalForceij(...)` | Combines everything into the pair force and energy. See below. |
| `evalSelfEnergy(Scalar& energy, Scalar phi)` | **DEAD CODE — never called** anywhere (CPU or GPU). Documented in §7. |
| `evalForceik(...)` | Returns `false` unconditionally (no `i-k` force is ever added). |
| `getName()` | Returns `"vbc"` (host-only). Used in error messages. |
| `getShapeSpec()` | Host-only. Throws `runtime_error("Shape definition not supported...")` — VBC has no shape spec. |

### `evalForceij` — the assembled pair force

```cpp
void evalForceij(force_divr, potential_eng,
                 rep_eng, rep_eng_div, attr_eng, attr_eng_div,
                 s_ij, s_ij_sum, W1, W2, Wrij_div)
{
    potential_eng += 0.5 * ( rep_eng + W1 * W2 * attr_eng );
    Scalar local_s_ij = (use_global_dens) ? s_ij_sum : s_ij;
    force_divr += 0.5 * ( rep_eng_div + W1 * W2 * attr_eng_div ) + Wrij_div * local_s_ij;
}
```

- Energy: repulsion plus the **gated** attraction `W1*W2*attr_eng`, halved
  (each side of the pair receives half).
- Force `F/r`: the direct radial derivatives of repulsion and gated attraction,
  plus the **density back-reaction** `Wrij_div * local_s_ij`. Here `Wrij_div`
  is `d(phi)/dr` for this `ij` bond (`evalPhiDiv`), and `local_s_ij` is the
  accumulated `W1_div * W2 * attr_eng` over the relevant neighbors — the chain-
  rule term that accounts for how moving `j` changes `i`'s density and therefore
  the gate on all of `i`'s other attractive bonds. Per-type vs global selection
  again governed by `use_global_dens`.

### `evalSelfEnergy` — what it WOULD have computed (preserved)

`evalSelfEnergy` is present in the header but is **not called by any code path**
(neither `PotentialValency.h` nor either GPU kernel invoke it). It is dead code
slated for removal. For the record, it would have added a quadratic density
self-energy penalizing deviation of a particle's density `rho_i` from `E_r`:

```cpp
void evalSelfEnergy(Scalar& energy, Scalar phi)
    {
    Scalar rho_i = phi;
    energy = 0.5 * E_a * (rho_i - E_r) * (rho_i - E_r);   // = 0.5 * E_a * (rho - E_r)^2
    }
```

i.e. `E_self = 0.5 * E_a * (phi - E_r)^2`. This is a leftover from an earlier
density-functional formulation and does **not** contribute to the current VBC
energy or forces.

---

## 5. The two-pass structure in `PotentialValency.h` (CPU)

`computeForces(timestep)` runs in two passes over the neighbor list. It requires
a **full** neighbor list; a half list is rejected up front:

```cpp
bool third_law = m_nlist->getStorageMode() == NeighborList::half;
if (third_law) throw runtime_error("PotentialValency cannot handle a half neighborlist");
```

Reason: with density-dependent gating each particle must see *all* its neighbors
to compute its own density; a half list would only store each pair once.
Positions go through `box.minImage(...)` (periodic boundary conditions) at every
distance evaluation.

### `m_phi_ab_arr` layout

`m_phi_ab_arr` is a `GPUArray<Scalar>` sized:

```
(N + Nghosts) * (ntypes + 1)
```

For particle `i` and neighbor-type `typej`:

- `phi_ab_arr[i * (ntypes + 1) + typej]` — **per-type** density of `i`: sum of
  `evalPhi` over `i`'s neighbors of type `typej`. There are `ntypes` such
  columns (indices `0 .. ntypes-1`).
- `phi_ab_arr[i * (ntypes + 1) + ntypes]` — the **extra last column**: the
  **global density** `gsum` = sum of `evalPhi` over *all* of `i`'s neighbors
  (all types). It equals the sum of the per-type columns.

The `+1` extra column is the key design element. It is populated in pass 1 and
read back in pass 2 for **both** `i` and `j` (`phi_ab_arr[j*(ntypes+1)+ntypes]`),
which is what makes the asymmetric-`ncut` case obey Newton's 3rd law. Previously
the code passed `i`'s global density to *both* `evalOmega` and `evalOmegaji`,
which was incorrect: `evalOmegaji` (the `j`-side gate) must see `j`'s global
density.

### Pass 1 — density fill

For every home particle `i` (loop `0 .. N`, ghosts excluded as home particles),
gated on `evaluator::hasPerParticleEnergy()`:

```cpp
Scalar gsum = 0;
for each neighbor j of i:
    rij_sq = |minImage(posi - posj)|^2
    evaluator eval(rij_sq, params(typei, typej));
    Scalar phi_contrib = 0; eval.evalPhi(phi_contrib);
    phi_ab_arr[i*(ntypes+1) + typej] += phi_contrib;   // per-type slot
    gsum += phi_contrib;                                // running global sum
phi_ab_arr[i*(ntypes+1) + ntypes] = gsum;               // store global density
```

`gsum` is accumulated inline while walking neighbors, avoiding a separate
sum-over-types pass.

### Pass 2 — forces

Pass 2 has an inner pre-loop and then the main force loop, per home particle `i`:

1. **Pre-loop** over neighbors builds the density back-reaction accumulators:
   - `m_cum_dens_ab[typej] += W1_div * W2 * attr_eng` (per-type bucket)
   - `m_cum_dens_ab_sum += W1_div * W2 * attr_eng` (global bucket)
   where `W1_div = evalOmegaDiv(..., phi_ab_i_sum)` uses `i`'s density and
   `W2 = evalOmegaji(..., phi_ab_j_sum)` uses **`j`'s** global density
   (`phi_ab_arr[jj*(ntypes+1)+ntypes]`). This is the asymmetric-cap fix.
   > Subtle note from the comments: the global bucket accumulates the *per-
   > neighbour increment*, not the running per-type bucket — accumulating the
   > running bucket would triangular-overcount the global back-reaction.

2. **Main force loop** over neighbors:
   - `evalRepulsive`, `evalAttractive` for the `ij` bond.
   - `W1 = evalOmega(..., phi_ab_i_sum)`, `W2 = evalOmegaji(..., phi_ab_j_sum)`.
   - `Wrij_div = evalPhiDiv(...)` (density gradient of this bond).
   - `evalForceij(force_divr, potential_eng, rep_eng, rep_eng_div, attr_eng,
      attr_eng_div, m_cum_dens_ab[typej], m_cum_dens_ab_sum, W1, W2, Wrij_div)`
      assembles the pair energy and force (see §4).
   - Newton pair accumulation:
     ```cpp
     fi += -1 * dxij * (force_divr / rij_abs);   pei += 0.5 * potential_eng;
     fj +=      dxij * (force_divr / rij_abs);    pej += 0.5 * potential_eng;
     h_force[jj] += (fj, pej);   // per neighbor
     ...
     h_force[i]  += (fi, pei);   // after the loop
     ```

> Comment flag preserved: the `-1.0` sign on `fi` carries an inline TODO
> ("sign of force might be different TODO: check in simulation") — the author was
> not fully certain of the sign convention; it should be validated against a
> reference simulation.

Force and virial arrays and the density array are zeroed with `memset` at the top
of each `computeForces`; the density `memset` correctly spans `(ntypes + 1)`
columns. Virial accumulation is present in the array plumbing but the per-pair
virial contribution code is commented out (virial is effectively not filled).

There is a large **commented-out `k`-loop** in pass 2 (the original three-body
`evalForceijk` path). It is inert dead scaffolding kept from the Tersoff origin
and is not part of the VBC physics.

MPI (`ENABLE_MPI`): `getRequestedCommFlags` requests reverse net-force
communication plus tags, so per-ghost force contributions are summed back to the
owning rank.

---

## 6. GPU specifics (`PotentialValencyGPU.h` / `.cuh` / `.cu`)

`PotentialValencyGPU<evaluator>` derives from `PotentialValency<evaluator>` and
overrides `computeForces`. It adds:

- **Autotuner** `m_tuner` of type `Autotuner<2>` named `"pair_valency"`, tuning
  two parameters: block size (`makeBlockSizeRange`) and threads-per-particle
  (`getTppListPow2`, powers of two up to the warp width). Registered in
  `m_autotuners`. The GPU constructor throws if no CUDA/HIP device is present.
- **Half-list rejection** identical to the CPU path.
- Device `ArrayHandle`s for neighbor list, positions, params, `rcutsq`, force,
  virial, and `m_phi_ab_arr`, then a single call to the templated driver
  `kernel::gpu_compute_valency_forces<evaluator>(valency_args_t(...), d_params)`
  wrapped in `m_tuner->begin()/end()` and a CUDA error check.

`valency_args_t` (in the `.cuh`) bundles all kernel arguments, including
`d_phi_ab_arr`, `N`, `Nghosts`, `ntypes`, `block_size`, `tpp`, `box`, and device
properties.

### Two-kernel design (matches the CPU two passes)

The driver `gpu_compute_valency_forces` dispatches (via the recursive
`ValencyComputeKernel<evaluator, compute_virial, tpp>` template that peels `tpp`
down by halves until it matches the tuner's `tpp`, terminating at the `tpp = 0`
no-op specialization) into a launcher that:

1. Zeros `d_force`, `d_virial`, and `d_phi_ab_arr` (the latter with stride
   `ntypes + 1` per particle — the extra global-density column).
2. **Kernel A — `gpu_compute_density_kernel<evaluator, tpp>`:** phase-1 density
   fill. Each particle is handled by a group of `tpp` threads striding over its
   neighbors. Per neighbor: `evalPhi` → `myAtomicAdd` into the per-type slot
   `d_phi_ab_arr[idx*(ntypes+1) + typej]` (atomic because multiple threads of the
   group may hit the same type slot) **and** accumulate into a per-thread local
   `phi_partial`. After the loop the `tpp` partials are `WarpReduce`-summed and
   thread 0 writes the total into the global-density column
   `d_phi_ab_arr[idx*(ntypes+1) + ntypes]`.
3. **Kernel B — `gpu_compute_valency_forces_kernel<evaluator, compute_virial,
   tpp>`:** phase-2 forces. Mirrors the CPU pass-2 pre-loop + main loop using
   shared memory `s_phi_ab[threadIdx.x*ntypes + typej]` as the per-type
   back-reaction accumulator (`W1_div * W2 * attr_eng`), warp-reduced/broadcast
   across the `tpp` group into `s_phi_ab_sum`. It reads the precomputed global
   densities for both `idx` (`phi_ab_idx_sum`) and `cur_j` (`phi_ab_jdx_sum`),
   applies `evalOmega`/`evalOmegaji`/`evalPhiDiv`/`evalForceij`, and writes forces
   with `myAtomicAdd` to both `d_force[cur_j]` and `d_force[idx]`. Pair energy is
   halved onto each side (`forcei.w`/`forcej.w += 0.5 * potential_eng`).

**Grid-wide barrier:** the separation into two kernels is deliberate — the kernel
**launch boundary between Kernel A and Kernel B is the grid-wide barrier**. It
guarantees every particle's per-type slots *and* global-density column are fully
written (even for particles owned by other blocks) before any thread in Kernel B
reads `cur_j`'s global density. A single fused kernel could not safely read a
neighbor's global density that is produced in another block.

`myAtomicAdd` provides `atomicAdd` for `float`, plus a CAS-loop `double`
implementation for pre-`sm_60` (`__CUDA_ARCH__ < 600`) and a HIP/HCC `float`
workaround.

**Shared memory:** Kernel B allocates `s_params + s_rcutsq` (per type-pair) plus
`ntypes * block_size` scalars for `s_phi_ab`. The launcher shrinks
`run_block_size` by `warpSize` until the shared-memory request fits
`sharedMemPerBlock`, and throws if it still cannot fit. Kernel A needs less shared
memory (no `s_phi_ab`) but reuses the same (over-)allocation for safety.

**Full neighbor list required (GPU too):** the GPU path throws on a half list,
same as CPU — every particle must see all neighbors to build its density.

Commented-out code in Kernel B (the `chi`/`k`-loop three-body path and the virial
accumulation) is inert Tersoff-origin scaffolding, kept but unused.

### Explicit instantiation (`.cu`)

`PotentialValencyVBCGPUKernel.cu` forces the driver to be compiled for
`EvaluatorVBC`:

```cpp
#define EVALUATOR_CLASS EvaluatorVBC
template __attribute__((visibility("default"))) hipError_t
gpu_compute_valency_forces<EvaluatorVBC>(const valency_args_t& pair_args,
                                         const typename EvaluatorVBC::param_type* d_params);
```

Without this explicit instantiation the host-side call in `PotentialValencyGPU.h`
would have no device object to link against (the `#else` branch in the `.cuh`
provides only a declaration for non-device translation units).

---

## 7. Known quirks / TODO

- **`evalSelfEnergy` is dead code.** It is defined in `EvaluatorVBC.h` but never
  invoked by the CPU driver or either GPU kernel. It would have computed a density
  self-energy `0.5 * E_a * (phi - E_r)^2` (see §4). Safe to remove; documented here to
  preserve the intent.
- **`setRik` copy-paste bug.** `setRik(Scalar rsq)` writes `rij_sq = rsq` instead
  of an `ik` distance member. It is harmless only because no active code path
  calls it (the `chi` / `ik` three-body path is entirely commented out). If the
  `ik` path is ever revived, this must be fixed to store a separate `rik_sq`.
- **`hasPerParticleEnergy()` is a control flag, not an energy.** It returns `true`
  purely to enable the phase-1 density precompute (both CPU and GPU gate the
  density loop on it). It does **not** add a per-particle energy term to the
  output — the actual energy comes only from `evalForceij`'s
  `0.5*(rep_eng + W1*W2*attr_eng)`.
- **Force sign TODO (CPU).** The `-1.0` factor on the `i` force in
  `PotentialValency.h` pass 2 carries an author TODO to verify the sign
  convention against a real simulation.
- **Virial not computed.** Virial arrays are allocated and zeroed, and the kernel
  carries `compute_virial` plumbing, but all per-pair virial accumulation is
  commented out on both CPU and GPU. Pressure-tensor output from this potential is
  effectively zero.
- **Legacy naming.** `PotentialValency.h`'s include guard is
  `__POTENTIAL_TERSOFF_H__`, and both the CPU driver and GPU kernels retain large
  commented-out Tersoff three-body (`evalForceijk`, `chi`, `k`-loop) blocks. These
  are inert and not part of VBC.

---

## Source files

All in `/groups/goloborodko/projects/vladimir/hoomd_from_source/vbc_plugin/src/`:

- `module.cc`
- `EvaluatorVBC.h`
- `PotentialValency.h`
- `PotentialValencyGPU.h`
- `PotentialValencyGPU.cuh`
- `PotentialValencyVBCGPUKernel.cu`

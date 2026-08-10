# GPU notes: why the sweeps look the way they do

This note documents the 2026 GPU overhaul of the plugin (the two PRs
"structure" and "behaviour"): six findings from a kernel audit, what was
changed for each, and the measured results. It assumes familiarity with the
physics (docs/potential.md) but not with GPU programming.

Headline, RTX 4090, N = 100k, rho = 0.5, 6 types, Tree(buffer=0.4), Langevin:
**7.10 -> 0.58 ms/step (12x); the valency force alone 6.57 -> 0.05 ms
(~130x)**; forces within ~3e-7 (normalised) of an independent float64
reference; pressures now actually include this force; GPU forces bitwise
reproducible.

## Background: how this force runs on a GPU

The computational shape first: HOOMD keeps, for every particle, a list of its
neighbours within r_cut (plus a small safety margin, the nlist "buffer", so
the list need not be rebuilt every step). The plugin computes the force in
sweeps over these lists — the density sweep, the S sweep, and the force sweep.
Four facts cover everything below:

**Fact 1 — a GPU runs a function as thousands of threads.** A GPU function (a
"kernel") is launched once and executed simultaneously by tens of thousands of
threads: lightweight workers, each assigned a small slice of the job — here, a
share of one particle's neighbour list. A thread keeps its few local variables
in *registers*, which are effectively free to access; anything shared between
threads lives in the GPU's main memory, hundreds of clock cycles away. (Each
kernel *launch* also has a fixed cost of a few microseconds no matter how
little work it contains — at small N the whole step is launch-bound and the
speedups above shrink.)

**Fact 2 — gaming GPUs nearly lack double-precision hardware.** An RTX 4090
executes 64 single-precision (float) operations in the time of one
double-precision operation; a data-center A100 executes 2. Double sqrt and
division are worse still, being built from many such operations, and doubles
move twice the bytes. Code that computes in `Scalar` (= double) therefore uses
a few percent of a consumer card's speed.

**Fact 3 — simultaneous additions to the same variable need atomics, and
atomics queue.** When two threads execute `x += a` on the same variable at the
same moment, an update can be lost: both read the old value, and one write
overwrites the other. `atomicAdd` fixes this by having the memory system apply
the additions strictly one at a time. That safety has a price: additions
converging on the *same* variable form a single-file queue — thousands of
parallel threads, processed sequentially, in main memory.

**Fact 4 — HOOMD gives each particle a small *team* of threads.** To spread a
long neighbour loop, HOOMD assigns every particle up to 32 threads (the
*threads-per-particle* parameter, `tpp`), which split that particle's
neighbour list between them. The hardware runs threads in groups of 32 (a
"warp") that march in lockstep and can hand values to one another directly,
register to register, without touching memory. A team can therefore sum its
members' partial results in a few clock cycles (a *warp reduction*) and have
one designated member — the team leader — commit the total with a single
ordinary write. Teams are what make atomics avoidable: tally privately,
combine within the warp, write once.

## The findings and what changed

### 1. All the math ran in double precision

Everything derived from the `Scalar4` positions — distances, windows, gates,
energies — stayed `Scalar` (double) all the way down. Per fact 2 that used a
few percent of a consumer card.

*Now:* positions are still loaded as double and the subtraction + minimum
image still run in double (as everywhere in HOOMD), then the separation vector
is narrowed to float **once** — and everything downstream is float, with
per-particle totals widened back to Scalar at their single store. A separation
is at most r_cut, which float resolves to ~1e-7 x r_cut, so nothing that
matters is lost: measured worst case vs the float64 reference is 1.8e-7
(energies), 3.1e-7 (forces), 5.7e-7 (virials), orders of magnitude below the
Langevin thermostat's per-step noise. Integration and stored positions stay
double.

### 2. The density sweep queued every neighbour on the same memory slot

Each neighbour's smooth weight was added to `phi[i][type_j]` with one global
`atomicAdd` — and all of team i targets the same few slots, the single-file
queue of fact 3, in double on top (fact 2).

*Now:* the pattern from fact 4 — each thread tallies its share of neighbours
per type in its own slab slots (nothing shared until the end), one warp
reduction per type, and the team leader writes the finished row. n_neigh
atomics per particle became ntypes+1 plain writes.

### 3. The force sweep pushed forces to partners through atomics

Every pair (i, j) is processed twice — once by each particle's team. Each
visit computed half the pair force, plus its own particle's share of the
many-body term, and added the result to *both* particles: four atomicAdds per
neighbour into the partner's slot, because the many-body sum s_i lived in
shared-memory scratch the partner's team cannot see.

*Now:* the sums are a stored per-particle array S (same row layout as phi,
filled by its own sweep — literally the old kernel's first loop with the
accumulator redirected to global memory). Any team can then read any
particle's S, so each visit computes the complete derivative

    dE/dr_ij = Vr' + W1 W2 Va' + w'(r_ij) (S_i + S_j)

for its own particle only and writes nothing anywhere else. Zero atomics
remain in the plugin, which also makes forces bitwise reproducible run to run
(asserted in the test suite). The CPU path runs the identical three sweeps, so
the backends can be compared term by term (also asserted).

### 4. Each pair paid for sqrt up to five times, and divided by constants

Every evaluator method re-derived `r` from `r^2` itself, normalised by the
window widths with per-pair divisions although the widths are fixed at
`setParams` time, and left `pow(x, 2)` for the compiler to lower.

*Now:* one `rsqrt` per pair at evaluator construction (`rsqrt` computes
1/sqrt(x) as a single hardware op — cheaper than sqrt followed by a division),
host-precomputed reciprocals in `param_type`, explicit squares, and an
early-out on the pair r_cut (which was staged into fast memory but never
consulted). Bonus from the guarded rsqrt: exactly overlapping particles give
zero force instead of NaN.

### 5. Bug: pressure never included this force

The virial — the per-pair r (x) F bookkeeping HOOMD sums into the pressure —
was allocated and zeroed but never written, on either backend. **Every
pressure measured with VBC active was the pressure without the VBC force**,
and NPT would have held the box at the wrong pressure.

*Now:* every term of the force, the many-body one included, acts along the
pair separation, so the textbook pairwise formula 0.5 dx_a dx_b (F/r) applies
term by term. Validated against the float64 reference; costs ~1% and only on
steps where pressure is requested.

### 6. Bug: multi-rank (MPI) runs got wrong forces at domain boundaries

Under domain decomposition each rank owns a region of the box plus read-only
copies ("ghosts") of nearby particles. The density sweep fills phi for owned
particles only, so the force sweep read ghost neighbours with phi = 0 — and a
zero density means a fully open valency gate: bonds crossing a domain boundary
were never capped, silently.

*Now:* the plugin refuses decomposed runs with an error explaining why.
Single-GPU runs have no ghosts and are unaffected. The real fix — 
communicating ghost densities between sweeps — is left for when multi-rank
runs are actually needed.

## Measured results

RTX 4090, stock HOOMD 5.1.0, N = 100k, rho = 0.5, 6 types (caps 1..6, global
mode), r_cut = 2.0, Tree(buffer=0.4), Langevin dt = 0.005. "Force only"
subtracts a Gaussian-pair baseline on the same neighbour list.

| after | whole step ms | force-only ms | cumulative |
|---|---|---|---|
| original plugin | 7.100 | 6.571 | 1.0x |
| finding 4 (evaluator, still double) | 1.375 | 0.847 | 5.2x |
| finding 2 (density, still double) | 1.393 | 0.864 | ~ |
| finding 3 (one-sided, still double) | 1.146 | 0.617 | 6.2x |
| findings 5+6 (virial off / on) | 1.149 / 1.230 | 0.620 | 6.2x |
| finding 1 (float32) | **0.579** | **0.050** | **12.3x / ~130x force-only** |

Two honest footnotes. First, the whole step is now dominated by stock HOOMD's
own double-precision integrator and neighbour list (the 0.529 ms baseline),
not by this plugin. Second, most of the pre-float gain came from finding 4 —
on a 4090 the repeated double sqrt/divisions were even more expensive than the
atomics at this density; finding 2 was near-neutral at this scale and matters
more at higher type counts and densities.

## Validation

`tests/` holds an independent float64 numpy implementation written directly
from `docs/potential.md` (not from the C++, so agreement validates both), and
compares per-particle energies, forces, and virials on CPU and GPU across
three parameter regimes, plus net-force, bitwise GPU determinism, NVE
conservation (0.02% drift / 10k steps), and CPU-vs-GPU Langevin statistics.
The original plugin was validated against the same reference at 1e-10 before
any change; every structural commit preserved that, and the float32 commit is
the single point where tolerances moved to the measured ~3e-7.

```bash
python -m pytest tests/ -v --gpu-id 0
python tests/benchmark_vbc.py --gpu-id 0 -n 100000 --pressure
```

*Lab-internal aside:* a fork-locked variant of these fixes (`vbcf_plugin`,
built on our patched HOOMD with a float position mirror) predates this port
and remains the fastest option inside the lab's fork environments; the numbers
above are this repository's own code on stock HOOMD.

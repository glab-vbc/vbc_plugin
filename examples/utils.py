"""
Analysis functions for the VBC diagnostics notebooks: 
read GSD frames and compute per-frame average KE per particle and centre-of-mass momentum, 
and the type-by-type contact-count matrix in mean / 0.9-quantile / 0.99-quantile versions
"""

import numpy as np
import gsd.hoomd
from scipy.spatial import cKDTree


def frames(path):
    return list(gsd.hoomd.open(path, "r"))


def potential_energy(frames):
    out = []
    for fr in frames:
        pe = np.nan
        for key in ("md/compute/ThermodynamicQuantities/potential_energy",
                    "log/md/compute/ThermodynamicQuantities/potential_energy"):
            if key in fr.log:
                pe = float(np.asarray(fr.log[key]).ravel()[0])
        out.append(pe)
    return np.array(out)


def ke_and_momentum(frames):
    ke_per_particle = []
    com_momentum = []
    for fr in frames:
        v = np.asarray(fr.particles.velocity, dtype=np.float64)
        m = np.asarray(fr.particles.mass, dtype=np.float64)
        ke_per_particle.append(0.5 * float(np.sum(m[:, None] * v * v)) / len(v))
        com_momentum.append(float(np.linalg.norm((m[:, None] * v).sum(axis=0))))
    return np.array(ke_per_particle), np.array(com_momentum)


def contact_matrix(frames, kind="mean", r_cut=1.5):
    n_types = len(frames[0].particles.types)
    accumulated = np.zeros((n_types, n_types))

    bonded = set()
    if frames[0].bonds.N:
        for a, b in np.asarray(frames[0].bonds.group):
            bonded.add((int(a), int(b)))
            bonded.add((int(b), int(a)))

    for fr in frames:
        pos = np.asarray(fr.particles.position, dtype=np.float64)
        tid = np.asarray(fr.particles.typeid)
        box = float(fr.configuration.box[0])
        wrapped = (pos + box / 2) % box
        pairs = cKDTree(wrapped, boxsize=box).query_pairs(r_cut, output_type="ndarray")
        if bonded and len(pairs):
            keep = np.array([(int(a), int(b)) not in bonded for a, b in pairs], dtype=bool)
            pairs = pairs[keep]

        nbr = np.zeros((len(pos), n_types), int)
        if len(pairs):
            np.add.at(nbr, (pairs[:, 0], tid[pairs[:, 1]]), 1)
            np.add.at(nbr, (pairs[:, 1], tid[pairs[:, 0]]), 1)

        frame_matrix = np.zeros((n_types, n_types))
        for i in range(n_types):
            members = tid == i
            if not members.any():
                continue
            counts = nbr[members]
            if kind == "mean":
                frame_matrix[i] = counts.mean(axis=0)
            elif kind == "q90":
                frame_matrix[i] = np.quantile(counts, 0.90, axis=0)
            elif kind == "q99":
                frame_matrix[i] = np.quantile(counts, 0.99, axis=0)
        accumulated += frame_matrix

    return accumulated / len(frames)

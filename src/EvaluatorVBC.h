// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __EVALUATOR_VBC__
#define __EVALUATOR_VBC__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {

//! Evaluator for the valency-bounding correction.
/*! Per-pair quantities of the VBC potential (docs/potential.md): the density
    weight w(r) and its derivative, the valency gates g(n; ncut) and their
    derivatives, and the repulsive core / attractive well of the base pair
    potential.

    All per-pair math runs in float32: on consumer GPUs double precision
    executes at 1/64 of float throughput, and the measured accuracy cost is
    ~1e-6 relative on forces - orders of magnitude below the Langevin
    thermostat's per-step noise (see docs/gpu-notes.md, finding 1). The
    kernels narrow the pair separation to float after the double-precision
    minimum image and hand r^2 to this evaluator; positions and integration
    stay double.

    r_ij is computed once, at construction, via rsqrt (the original recomputed
    sqrt(r^2) in every eval* method — up to four times per neighbour in the
    force sweep). The window and well reciprocals are precomputed on the host
    at parameter-set time, so the per-pair code divides nowhere, and all
    pow(x, 2) chains are explicit squares. The math is algebraically identical
    to the original; results match to floating-point rounding.
*/
class EvaluatorVBC
    {
    public:
    struct param_type
        {
        float E_r;
        float E_a;
        float attr_rcut;
        float rep_rcut;
        float density_upper_rcut;
        float density_lower_rcut;
        float ncut_ij;
        float ncut_ji;
        float use_global_dens;

        // Derived constants, precomputed host-side in setDerived(). They
        // depend only on the radial windows, so the ncut_ij/ncut_ji swap that
        // PotentialValency::setParams applies for transposed type pairs
        // leaves them valid.
        float re;             // attractive-well centre, (rep_rcut + attr_rcut)/2
        float inv_well_hw;    // 1 / well half-width = 2/(attr_rcut - rep_rcut)
        float inv_rep_rcut;   // 1 / rep_rcut
        float inv_dens_width; // 1 / (density_upper_rcut - density_lower_rcut)

#ifdef ENABLE_HIP

        void set_memory_hint() const
            {

            }
#endif

#ifndef __HIPCC__
        param_type()
            : E_r(0), E_a(0), attr_rcut(0), rep_rcut(0), density_upper_rcut(0),
              density_lower_rcut(0), ncut_ij(0), ncut_ji(0), use_global_dens(0)
            {
            setDerived();
            }

        param_type(pybind11::dict v)
            {
            E_r = float(v["E_r"].cast<Scalar>());
            E_a = float(v["E_a"].cast<Scalar>());
            attr_rcut = float(v["attr_rcut"].cast<Scalar>());
            rep_rcut = float(v["rep_rcut"].cast<Scalar>());
            density_upper_rcut = float(v["density_upper_rcut"].cast<Scalar>());
            density_lower_rcut = float(v["density_lower_rcut"].cast<Scalar>());
            ncut_ij = float(v["ncut_ij"].cast<Scalar>());
            ncut_ji = float(v["ncut_ji"].cast<Scalar>());
            use_global_dens = float(v["use_global_dens"].cast<Scalar>());
            setDerived();
            }

        //! Precompute the reciprocals so the per-pair code divides nowhere.
        /*! Degenerate windows get a 0 reciprocal; the corresponding branch is
            then unreachable (the window is empty), matching the original
            evaluator's behaviour. */
        void setDerived()
            {
            re = 0.5f * (rep_rcut + attr_rcut);
            inv_well_hw = (attr_rcut > rep_rcut) ? 2.0f / (attr_rcut - rep_rcut) : 0.0f;
            inv_rep_rcut = (rep_rcut > 0.0f) ? 1.0f / rep_rcut : 0.0f;
            inv_dens_width = (density_upper_rcut > density_lower_rcut)
                                 ? 1.0f / (density_upper_rcut - density_lower_rcut)
                                 : 0.0f;
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["E_r"] = Scalar(E_r);
            v["E_a"] = Scalar(E_a);
            v["attr_rcut"] = Scalar(attr_rcut);
            v["rep_rcut"] = Scalar(rep_rcut);
            v["density_upper_rcut"] = Scalar(density_upper_rcut);
            v["density_lower_rcut"] = Scalar(density_lower_rcut);
            v["ncut_ij"] = Scalar(ncut_ij);
            v["ncut_ji"] = Scalar(ncut_ji);
            v["use_global_dens"] = Scalar(use_global_dens);
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Construct for one pair. Computes r_ij (and 1/r_ij) once.
    /*! The r_sq == 0 guard returns zero force instead of the NaN the original
        produced for exactly overlapping particles. */
    DEVICE EvaluatorVBC(
        float _rij_sq,
        const param_type& _params
        ):
        p(_params)
        {
        m_rinv = (_rij_sq > 0.0f) ? fast::rsqrt(_rij_sq) : 0.0f;
        m_rij = _rij_sq * m_rinv;
        }

    DEVICE static float sq(float x)
        {
        return x * x;
        }

    //! 1/r_ij for the kernels' force normalisation (0 for coincident pairs).
    DEVICE float rinv() const
        {
        return m_rinv;
        }

    //! Density weight w(r): 1 below the window, quartic ease to 0 across it.
    DEVICE void evalPhi(float& phi)
        {
        if (m_rij <= p.density_lower_rcut)
            {
            phi += 1.0f;
            }
        else if (m_rij >= p.density_upper_rcut)
            {
            phi += 0.0f;
            }
        else
            {
            phi += sq(1.0f
                      - sq((m_rij - p.density_lower_rcut) * p.inv_dens_width));
            }
        }

    //! dw/dr, nonzero only inside the density window.
    DEVICE void evalPhiDiv(float& phi)
        {
        if (m_rij > p.density_lower_rcut && m_rij < p.density_upper_rcut)
            {
            float t = (m_rij - p.density_lower_rcut) * p.inv_dens_width;
            phi += -4.0f * t * (1.0f - t * t) * p.inv_dens_width;
            }
        }

    //! Valency gate g(n; ncut): 1 below the cap, quartic ease to 0 on [ncut, ncut+1].
    DEVICE static float gate(float n, float ncut)
        {
        float t = n - ncut;
        if (t <= 0.0f)
            return 1.0f;
        if (t >= 1.0f)
            return 0.0f;
        return sq(1.0f - t * t);
        }

    //! dg/dn, nonzero only inside the gate ramp.
    DEVICE static float gateDeriv(float n, float ncut)
        {
        float t = n - ncut;
        if (t <= 0.0f || t >= 1.0f)
            return 0.0f;
        return -4.0f * t * (1.0f - t * t);
        }

    //! Density argument selection: per-type or total, per the pair's use_global_dens.
    DEVICE float pickPhi(float phi_typed, float phi_total) const
        {
        return (p.use_global_dens != 0.0f) ? phi_total : phi_typed;
        }

    DEVICE void evalOmega(float& W, float phi, float phi_global)
        {
        W += gate(pickPhi(phi, phi_global), p.ncut_ij);
        }

    DEVICE void evalOmegaji(float& W, float phi, float phi_global)
        {
        W += gate(pickPhi(phi, phi_global), p.ncut_ji);
        }

    DEVICE void evalOmegaDiv(float& W, float phi, float phi_global)
        {
        W += gateDeriv(pickPhi(phi, phi_global), p.ncut_ij);
        }

    //! Attractive well V_a and dV_a/dr; both zero outside (rep_rcut, attr_rcut).
    DEVICE void evalAttractive(float& attr_eng, float& attr_eng_div)
        {
        if (m_rij > p.rep_rcut && m_rij < p.attr_rcut)
            {
            float t = (m_rij - p.re) * p.inv_well_hw;
            float u = 1.0f - t * t;
            attr_eng += p.E_a * u * u;
            attr_eng_div += -4.0f * p.E_a * t * u * p.inv_well_hw;
            }
        }

    //! Repulsive core V_r and dV_r/dr; both zero at and beyond rep_rcut.
    DEVICE void evalRepulsive(float& rep_eng, float& rep_eng_div)
        {
        if (m_rij < p.rep_rcut)
            {
            float t = m_rij * p.inv_rep_rcut;
            float u = 1.0f - t * t;
            rep_eng += p.E_r * u * u;
            rep_eng_div += -4.0f * p.E_r * t * u * p.inv_rep_rcut;
            }
        }

#ifndef __HIPCC__

    static std::string getName()
        {
        return std::string("vbc");
        }
#endif

    protected:
    const param_type& p;
    float m_rij;
    float m_rinv;
    };

    }
    }

#endif

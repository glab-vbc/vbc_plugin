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
        Scalar E_r;
        Scalar E_a;
        Scalar attr_rcut;
        Scalar rep_rcut;
        Scalar density_upper_rcut;
        Scalar density_lower_rcut;
        Scalar ncut_ij;
        Scalar ncut_ji;
        Scalar use_global_dens;

        // Derived constants, precomputed host-side in setDerived(). They
        // depend only on the radial windows, so the ncut_ij/ncut_ji swap that
        // PotentialValency::setParams applies for transposed type pairs
        // leaves them valid.
        Scalar re;             // attractive-well centre, (rep_rcut + attr_rcut)/2
        Scalar inv_well_hw;    // 1 / well half-width = 2/(attr_rcut - rep_rcut)
        Scalar inv_rep_rcut;   // 1 / rep_rcut
        Scalar inv_dens_width; // 1 / (density_upper_rcut - density_lower_rcut)

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
            E_r = v["E_r"].cast<Scalar>();
            E_a = v["E_a"].cast<Scalar>();
            attr_rcut = v["attr_rcut"].cast<Scalar>();
            rep_rcut = v["rep_rcut"].cast<Scalar>();
            density_upper_rcut = v["density_upper_rcut"].cast<Scalar>();
            density_lower_rcut = v["density_lower_rcut"].cast<Scalar>();
            ncut_ij = v["ncut_ij"].cast<Scalar>();
            ncut_ji = v["ncut_ji"].cast<Scalar>();
            use_global_dens = v["use_global_dens"].cast<Scalar>();
            setDerived();
            }

        //! Precompute the reciprocals so the per-pair code divides nowhere.
        /*! Degenerate windows get a 0 reciprocal; the corresponding branch is
            then unreachable (the window is empty), matching the original
            evaluator's behaviour. */
        void setDerived()
            {
            re = Scalar(0.5) * (rep_rcut + attr_rcut);
            inv_well_hw = (attr_rcut > rep_rcut) ? Scalar(2.0) / (attr_rcut - rep_rcut)
                                                 : Scalar(0.0);
            inv_rep_rcut
                = (rep_rcut > Scalar(0.0)) ? Scalar(1.0) / rep_rcut : Scalar(0.0);
            inv_dens_width = (density_upper_rcut > density_lower_rcut)
                                 ? Scalar(1.0) / (density_upper_rcut - density_lower_rcut)
                                 : Scalar(0.0);
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["E_r"] = E_r;
            v["E_a"] = E_a;
            v["attr_rcut"] = attr_rcut;
            v["rep_rcut"] = rep_rcut;
            v["density_upper_rcut"] = density_upper_rcut;
            v["density_lower_rcut"] = density_lower_rcut;
            v["ncut_ij"] = ncut_ij;
            v["ncut_ji"] = ncut_ji;
            v["use_global_dens"] = use_global_dens;
            return v;
            }
#endif
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    //! Construct for one pair. Computes r_ij (and 1/r_ij) once.
    /*! The r_sq == 0 guard returns zero force instead of the NaN the original
        produced for exactly overlapping particles. */
    DEVICE EvaluatorVBC(
        Scalar _rij_sq,
        const param_type& _params
        ):
        p(_params)
        {
        m_rinv = (_rij_sq > Scalar(0.0)) ? fast::rsqrt(_rij_sq) : Scalar(0.0);
        m_rij = _rij_sq * m_rinv;
        }

    DEVICE static Scalar sq(Scalar x)
        {
        return x * x;
        }

    //! 1/r_ij for the kernels' force normalisation (0 for coincident pairs).
    DEVICE Scalar rinv() const
        {
        return m_rinv;
        }

    DEVICE static bool hasPerParticleEnergy()
        {
        return true;
        }

    //! Density weight w(r): 1 below the window, quartic ease to 0 across it.
    DEVICE void evalPhi(Scalar& phi)
        {
        if (m_rij <= p.density_lower_rcut)
            {
            phi += Scalar(1.0);
            }
        else if (m_rij >= p.density_upper_rcut)
            {
            phi += Scalar(0.0);
            }
        else
            {
            phi += sq(Scalar(1.0)
                      - sq((m_rij - p.density_lower_rcut) * p.inv_dens_width));
            }
        }

    //! dw/dr, nonzero only inside the density window.
    DEVICE void evalPhiDiv(Scalar& phi)
        {
        if (m_rij > p.density_lower_rcut && m_rij < p.density_upper_rcut)
            {
            Scalar t = (m_rij - p.density_lower_rcut) * p.inv_dens_width;
            phi += Scalar(-4.0) * t * (Scalar(1.0) - t * t) * p.inv_dens_width;
            }
        }

    //! Valency gate g(n; ncut): 1 below the cap, quartic ease to 0 on [ncut, ncut+1].
    DEVICE static Scalar gate(Scalar n, Scalar ncut)
        {
        Scalar t = n - ncut;
        if (t <= Scalar(0.0))
            return Scalar(1.0);
        if (t >= Scalar(1.0))
            return Scalar(0.0);
        return sq(Scalar(1.0) - t * t);
        }

    //! dg/dn, nonzero only inside the gate ramp.
    DEVICE static Scalar gateDeriv(Scalar n, Scalar ncut)
        {
        Scalar t = n - ncut;
        if (t <= Scalar(0.0) || t >= Scalar(1.0))
            return Scalar(0.0);
        return Scalar(-4.0) * t * (Scalar(1.0) - t * t);
        }

    //! Density argument selection: per-type or total, per the pair's use_global_dens.
    DEVICE Scalar pickPhi(Scalar phi_typed, Scalar phi_total) const
        {
        return (p.use_global_dens != Scalar(0.0)) ? phi_total : phi_typed;
        }

    DEVICE void evalOmega(Scalar& W, Scalar phi, Scalar phi_global)
        {
        W += gate(pickPhi(phi, phi_global), p.ncut_ij);
        }

    DEVICE void evalOmegaji(Scalar& W, Scalar phi, Scalar phi_global)
        {
        W += gate(pickPhi(phi, phi_global), p.ncut_ji);
        }

    DEVICE void evalOmegaDiv(Scalar& W, Scalar phi, Scalar phi_global)
        {
        W += gateDeriv(pickPhi(phi, phi_global), p.ncut_ij);
        }

    //! Attractive well V_a and dV_a/dr; both zero outside (rep_rcut, attr_rcut).
    DEVICE void evalAttractive(Scalar& attr_eng, Scalar& attr_eng_div)
        {
        if (m_rij > p.rep_rcut && m_rij < p.attr_rcut)
            {
            Scalar t = (m_rij - p.re) * p.inv_well_hw;
            Scalar u = Scalar(1.0) - t * t;
            attr_eng += p.E_a * u * u;
            attr_eng_div += Scalar(-4.0) * p.E_a * t * u * p.inv_well_hw;
            }
        }

    //! Repulsive core V_r and dV_r/dr; both zero at and beyond rep_rcut.
    DEVICE void evalRepulsive(Scalar& rep_eng, Scalar& rep_eng_div)
        {
        if (m_rij < p.rep_rcut)
            {
            Scalar t = m_rij * p.inv_rep_rcut;
            Scalar u = Scalar(1.0) - t * t;
            rep_eng += p.E_r * u * u;
            rep_eng_div += Scalar(-4.0) * p.E_r * t * u * p.inv_rep_rcut;
            }
        }

    DEVICE void evalForceij(
        Scalar& force_divr,
        Scalar& potential_eng,
        Scalar rep_eng,
        Scalar rep_eng_div,
        Scalar attr_eng,
        Scalar attr_eng_div,
        Scalar s_ij,
        Scalar s_ij_sum,
        Scalar W1,
        Scalar W2,
        Scalar Wrij_div)
        {
        potential_eng += Scalar(0.5) * (rep_eng + W1 * W2 * attr_eng);
        Scalar local_s_ij = pickPhi(s_ij, s_ij_sum);
        force_divr += Scalar(0.5) * (rep_eng_div + W1 * W2 * attr_eng_div)
                      + Wrij_div * local_s_ij;
        }

#ifndef __HIPCC__

    static std::string getName()
        {
        return std::string("vbc");
        }
#endif

    protected:
    const param_type& p;
    Scalar m_rij;
    Scalar m_rinv;
    };

    }
    }

#endif

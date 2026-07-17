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

#ifdef ENABLE_HIP

        void set_memory_hint() const
            {

            }
#endif

#ifndef __HIPCC__
        param_type() : E_r(0), E_a(0), attr_rcut(0), rep_rcut(0), density_upper_rcut(0), density_lower_rcut(0), ncut_ij(0), ncut_ji(0), use_global_dens(0) { }

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

    DEVICE EvaluatorVBC(
        Scalar _rij_sq,
        const param_type& _params
        ):
        rij_sq(_rij_sq),
        attr_rcut(_params.attr_rcut),
        rep_rcut(_params.rep_rcut),
        density_upper_rcut(_params.density_upper_rcut),
        density_lower_rcut(_params.density_lower_rcut),
        ncut_ij(_params.ncut_ij),
        ncut_ji(_params.ncut_ji),
        E_r(_params.E_r),
        E_a(_params.E_a),
        use_global_dens(_params.use_global_dens)
        {
        }

    DEVICE void setRij(Scalar rsq)
        {
        rij_sq = rsq;
        }

    DEVICE void setRik(Scalar rsq)
        {
        rij_sq = rsq;
        }

    DEVICE static bool hasPerParticleEnergy()
        {
        return true;
        }

    DEVICE static bool needsChi()
        {
        return false;
        }

    DEVICE static bool hasIkForce()
        {
        return false;
        }

    DEVICE static bool needsAngle()
        {
        return false;
        }

    DEVICE void setAngle(Scalar _cos_th) { }

    DEVICE bool areInteractive()
        {
        return true;
        }

    DEVICE void evalChi(Scalar& chi) { }

    DEVICE void evalPhi(Scalar& phi)
        {

        Scalar rij = fast::sqrt(rij_sq);
        if (rij < density_lower_rcut)
            {
            phi += Scalar(1.0);
            }
        else if (rij > density_upper_rcut)
            {
            phi += Scalar(0.0);
            }
        else
        {
            phi += pow(Scalar(1.0)-pow((rij-density_lower_rcut)/(density_upper_rcut-density_lower_rcut),2),2);
        }
        }

    DEVICE void evalPhiDiv(Scalar& phi)
        {

        Scalar rij = fast::sqrt(rij_sq);
        if (rij < density_lower_rcut)
            {
            phi += Scalar(0.0);
            }
        else if (rij > density_upper_rcut)
            {
            phi += Scalar(0.0);
            }
        else
        {
            phi+=Scalar(2.0) * (Scalar(1.0) - pow((rij-density_lower_rcut)/(density_upper_rcut-density_lower_rcut),2)) * (Scalar(-2.0) * (rij-density_lower_rcut)/(pow((density_upper_rcut-density_lower_rcut),2)));
        }
        }

    DEVICE void evalOmega(Scalar& W, Scalar phi, Scalar phi_global)
        {

        Scalar local_phi = (use_global_dens) ? phi_global : phi;
        if (local_phi < ncut_ij)
            {
            W += Scalar(1.0);
            }
        else if (local_phi > (ncut_ij+Scalar(1.0)))
            {
            W += Scalar(0.0);
            }
        else
        {
            W += pow(Scalar(1.0)-pow((local_phi-ncut_ij),2),2);
        }
        }

    DEVICE void evalOmegaji(Scalar& W, Scalar phi, Scalar phi_global)
        {

        Scalar local_phi = (use_global_dens) ? phi_global : phi;
        if (local_phi < ncut_ji)
            {
            W += Scalar(1.0);
            }
        else if (local_phi > (ncut_ji+Scalar(1.0)))
            {
            W += Scalar(0.0);
            }
        else
        {
            W += pow(Scalar(1.0)-pow((local_phi-ncut_ji),2),2);
        }
        }

    DEVICE void evalOmegaDiv(Scalar& W, Scalar phi, Scalar phi_global)
        {
            Scalar local_phi = (use_global_dens) ? phi_global : phi;
            if (local_phi < ncut_ij)
            {
                W += Scalar(0.0);
            }
            else if (local_phi > ncut_ij+Scalar(1.0))
            {
                W += Scalar(0.0);
            }
            else
            {
                W += Scalar(2.0) * (Scalar(1.0) - pow((local_phi-ncut_ij),2)) * (Scalar(-2.0) * (local_phi-ncut_ij));
            }
        }
    DEVICE void evalOmegajiDiv(Scalar& W, Scalar phi, Scalar phi_global)
        {
            Scalar local_phi = (use_global_dens) ? phi_global : phi;
            if (local_phi < ncut_ji)
            {
                W += Scalar(0.0);
            }
            else if (local_phi > ncut_ji+1)
            {
                W += Scalar(0.0);
            }
            else
            {
                W+= Scalar(2.0) * (Scalar(1.0) - pow((local_phi-ncut_ji),2)) * (Scalar(-2.0) * (local_phi-ncut_ji));
            }
        }

    DEVICE void evalAttractive(Scalar& attr_eng, Scalar& attr_eng_div)
    {
        Scalar rij = fast::sqrt(rij_sq);
        if (rij < attr_rcut && rij > rep_rcut)
        {
            Scalar re = (rep_rcut + attr_rcut)/Scalar(2.0);
            attr_eng += E_a * pow(Scalar(1.0)-pow((rij-re)/((attr_rcut-rep_rcut)/Scalar(2.0)),2),2);
            attr_eng_div += Scalar(2.0) * E_a * (Scalar(1.0)-pow((rij-re)/((attr_rcut-rep_rcut)/Scalar(2.0)),2)) * (Scalar(-2.0) * (rij-re)/pow((attr_rcut-rep_rcut)/Scalar(2.0),2));
        }
    }
    DEVICE void evalRepulsive(Scalar& rep_eng, Scalar& rep_eng_div)
    {
        Scalar rij = fast::sqrt(rij_sq);
        if (rij < rep_rcut)
        {
            rep_eng += E_r * pow(Scalar(1.0)-pow(rij/rep_rcut,2),2);
            rep_eng_div += Scalar(2.0) * E_r * (Scalar(1.0)-pow(rij/rep_rcut,2)) * (Scalar(-2.0) * rij/pow(rep_rcut,2));
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
        Scalar local_s_ij = (use_global_dens) ? s_ij_sum : s_ij;
        force_divr += Scalar(0.5) * (rep_eng_div + W1 * W2 * attr_eng_div) + Wrij_div * local_s_ij;
    }

    DEVICE bool evalForceik(Scalar fR,
                            Scalar fA,
                            Scalar chi,
                            Scalar bij,
                            Scalar3& force_divr_ij,
                            Scalar3& force_divr_ik)
        {
        return false;
        }

#ifndef __HIPCC__

    static std::string getName()
        {
        return std::string("vbc");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rij_sq;
    Scalar attr_rcut;
    Scalar rep_rcut;
    Scalar density_upper_rcut;
    Scalar density_lower_rcut;
    Scalar ncut_ij;
    Scalar ncut_ji;
    Scalar E_r;
    Scalar E_a;
    Scalar use_global_dens;
    };

    }
    }

#endif

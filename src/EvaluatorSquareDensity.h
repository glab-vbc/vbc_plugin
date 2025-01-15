// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __EVALUATOR_SQUARE_DENSITY__
#define __EVALUATOR_SQUARE_DENSITY__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"
// #include <iostream>
/*! \file EvaluatorSquareDensity.h
    \brief Defines the evaluator class for a three-body square density potential

   see P. Warren, "Vapor-liquid coexistence in many-bod dissipative particle dynamics"
   Phys. Rev. E 68, p. 066702 (2003)
*/

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
//! Class for evaluating the SquareDensity three-body potential
class EvaluatorSquareDensity
    {
    public:
    struct param_type
        {
        Scalar A;
        Scalar B;
        Scalar attr_rcut;
        Scalar rep_rcut;
        Scalar density_rcut;
        Scalar ncut_ij;
        Scalar ncut_ji;

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

#ifndef __HIPCC__
        param_type() : A(0), B(0), attr_rcut(0), rep_rcut(0), density_rcut(0), ncut_ij(0), ncut_ji(0) { }

        param_type(pybind11::dict v)
            {
            A = v["A"].cast<Scalar>();
            B = v["B"].cast<Scalar>();
            attr_rcut = v["attr_rcut"].cast<Scalar>();
            rep_rcut = v["rep_rcut"].cast<Scalar>();
            density_rcut = v["density_rcut"].cast<Scalar>();
            ncut_ij = v["ncut_ij"].cast<Scalar>();
            ncut_ji = v["ncut_ji"].cast<Scalar>();
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["A"] = A;
            v["B"] = B;
            v["attr_rcut"] = attr_rcut;
            v["rep_rcut"] = rep_rcut;
            v["density_rcut"] = density_rcut;
            v["ncut_ij"] = ncut_ij;
            v["ncut_ji"] = ncut_ji;
            return v;
            }
#endif
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    //! Constructs the evaluator
    /*! \param _rij Squared distance between particles i and j
        \param _rcutsq Squared distance at which the potential goes to zero
        \param _params Per type-pair parameters for this potential
    */
    DEVICE EvaluatorSquareDensity(
        Scalar _rij_sq, 
        const param_type& _params
        ): 
        rij_sq(_rij_sq), 
        attr_rcut(_params.attr_rcut),
        rep_rcut(_params.rep_rcut),
        density_rcut(_params.density_rcut),
        ncut_ij(_params.ncut_ij),
        ncut_ji(_params.ncut_ji),       
        A(_params.A), 
        B(_params.B)
        {
        }

    //! Set the square distance between particles i and j
    DEVICE void setRij(Scalar rsq)
        {
        rij_sq = rsq;
        }

    //! Set the square distance between particles i and k
    DEVICE void setRik(Scalar rsq)
        {
        rij_sq = rsq;
        }

    //! We have a per-particl excess free energy
    DEVICE static bool hasPerParticleEnergy()
        {
        return true;
        }

    //! We don't need per-particle-pair chi
    DEVICE static bool needsChi()
        {
        return false;
        }

    //! We don't have ik-forces
    DEVICE static bool hasIkForce()
        {
        return false;
        }

    //! The SquareDensity potential needs the bond angle
    DEVICE static bool needsAngle()
        {
        return false;
        }

    //! Set the bond angle value
    //! \param _cos_th Cosine of the angle between ij and ik
    DEVICE void setAngle(Scalar _cos_th) { }

    //! Check whether a pair of particles is interactive
    DEVICE bool areInteractive()
        {
        return true;
        }

    //! Evaluate the repulsive and attractive terms of the force
    // DEVICE bool evalRepulsiveAndAttractive(Scalar& fR, Scalar& fA)
    //     {
    //     // this method does nothing except checking if we're inside the cut-off
    //     return (rij_sq < rcutsq);
    //     }

    //! Evaluate chi (the scalar ik contribution) for this triplet
    DEVICE void evalChi(Scalar& chi) { }

    //! Evaluate chi (the scalar ij contribution) for this triplet
////////////////////////////////////////////// NEWLY ADDED FUNCTION
    // DEVICE void evalPhi(Scalar& phi)
    //     {
    //     // add up density n_i
    //     if (rij_sq < rcutsq)
    //         {
    //         Scalar norm(15.0 / (2.0 * M_PI));
    //         Scalar rcut = fast::sqrt(rcutsq);
    //         norm /= rcutsq * rcut;

    //         Scalar rij = fast::sqrt(rij_sq);
    //         Scalar fac = Scalar(1.0) - rij / rcut;
    //         phi += fac * fac * norm;
    //         }
    //     }
    DEVICE void evalPhi(Scalar& phi)
        {
        // add up density n_i
        Scalar rij = fast::sqrt(rij_sq);
        if (rij < attr_rcut)
            {
            phi += Scalar(1.0);
            }
        else if (rij > density_rcut)
            {
            phi+=Scalar(0.0);
            }
        else 
        {
            phi+=pow(Scalar(1.0)-pow((rij-attr_rcut)/(density_rcut-attr_rcut),2),2);
        }
        }

    DEVICE void evalPhiDiv(Scalar& phi)
        {
        // add up density n_i
        Scalar rij = fast::sqrt(rij_sq);
        if (rij < attr_rcut)
            {
            phi += Scalar(0.0);
            }
        else if (rij > density_rcut)
            {
            phi+=Scalar(0.0);
            }
        else 
        {
            phi+=pow(Scalar(1.0)-pow((rij-attr_rcut)/(density_rcut-attr_rcut),2),2);
            phi+=Scalar(2.0) * (Scalar(1.0) - pow((rij-attr_rcut)/(density_rcut-attr_rcut),2)) * (Scalar(-2.0) * (rij-attr_rcut)/pow((density_rcut-attr_rcut),2));
        }
        }

    DEVICE void evalOmega(Scalar& W, Scalar phi)
        {
        // add up density n_i
        if (phi < ncut_ij)
            {
            W += Scalar(1.0);
            }
        else if (phi > (ncut_ij+Scalar(1.0)))
            {
            phi+=Scalar(0.0);
            }
        else 
        {
            phi+=pow(Scalar(1.0)-pow((phi-ncut_ij),2),2);
        }
        }

    DEVICE void evalOmegaji(Scalar& W, Scalar phi)
        {
        // add up density n_i
        if (phi < ncut_ji)
            {
            W += Scalar(1.0);
            }
        else if (phi > (ncut_ji+Scalar(1.0)))
            {
            phi+=Scalar(0.0);
            }
        else 
        {
            phi+=pow(Scalar(1.0)-pow((phi-ncut_ji),2),2);
        }
        }

    DEVICE void evalOmegaDiv(Scalar& W, Scalar phi)
        {
            if (phi < ncut_ij)
            {
                W += Scalar(0.0);
            }
            else if (phi > ncut_ij+1)
            {
                W += Scalar(0.0);
            }
            else 
            {
                W+= Scalar(2.0) * (Scalar(1.0) - pow((phi-ncut_ij),2)) * (Scalar(-2.0) * (phi-ncut_ij));
            }
        }
    DEVICE void evalOmegajiDiv(Scalar& W, Scalar phi)
        {
            if (phi < ncut_ji)
            {
                W += Scalar(0.0);
            }
            else if (phi > ncut_ji+1)
            {
                W += Scalar(0.0);
            }
            else 
            {
                W+= Scalar(2.0) * (Scalar(1.0) - pow((phi-ncut_ji),2)) * (Scalar(-2.0) * (phi-ncut_ji));
            }
        }
    // DEVICE void evalOmega(Scalar& W, Scalar x, Scalar x1, Scalar x2)
    //     {
    //         if (x < x1)
    //         {
    //             W += Scalar(1.0);
    //         }
    //         else if (x > x2)
    //         {
    //             W += Scalar(0.0);
    //         }
    //         else 
    //         {
    //             W+=pow(Scalar(1.0) - pow((x-x1)/(x2-x1),2),2);
    //         }
    //     }
    // DEVICE void evalOmegaDiv(Scalar& W, Scalar x, Scalar x1, Scalar x2)
    //     {
    //         if (x < x1)
    //         {
    //             W += Scalar(0.0);
    //         }
    //         else if (x > x2)
    //         {
    //             W += Scalar(0.0);
    //         }
    //         else 
    //         {
    //             W+= Scalar(2.0) * (Scalar(1.0) - pow((x-x1)/(x2-x1),2)) * (Scalar(-2.0) * (x-x1)/pow((x2-x1),2));
    //         }
    //     }
    DEVICE void evalAttractive(Scalar& attr_eng, Scalar& attr_eng_div) 
    { 
        Scalar rij = fast::sqrt(rij_sq);
        if (rij < attr_rcut && rij > rep_rcut)
        {
            Scalar re = (rep_rcut + attr_rcut)/Scalar(2.0);
            attr_eng += B * pow(Scalar(1.0)-pow((rij-re)/(attr_rcut-rep_rcut)/Scalar(2.0),2),2);
            attr_eng_div += Scalar(2.0) * B * (Scalar(1.0)-pow((rij-re)/(attr_rcut-rep_rcut)/Scalar(2.0),2)) * (Scalar(-2.0) * (rij-re)/pow((attr_rcut-rep_rcut)/Scalar(2.0),2));
        }
    }
    DEVICE void evalRepulsive(Scalar& rep_eng, Scalar& rep_eng_div) 
    { 
        Scalar rij = fast::sqrt(rij_sq);
        if (rij < rep_rcut)
        {
            rep_eng += A * pow(Scalar(1.0)-pow(rij/rep_rcut,2),2);
            rep_eng_div += Scalar(2.0) * A * (Scalar(1.0)-pow(rij/rep_rcut,2)) * (Scalar(-2.0) * pow(rij/rep_rcut,2));
        }
    }
    
    DEVICE void evalForceijk(
        Scalar& force_divr, 
        Scalar& potential_eng,
        Scalar rij_sq_ext,
        unsigned int idx_j, 
        unsigned int idx_k, 
        Scalar rep_eng,
        Scalar rep_eng_div,
        Scalar attr_eng,
        Scalar attr_eng_div,
        Scalar W1,
        Scalar W1_div,
        Scalar W2,
        Scalar W2_div,
        Scalar Wrij_div)
    {
        Scalar r = fast::sqrt(rij_sq_ext);
        Scalar C1 = Scalar(1.0);
        if (idx_j == idx_k)
        {
            C1 = Scalar(0.5);
            potential_eng += C1 * (rep_eng + W1 * W2 * attr_eng);
            force_divr += C1 * (rep_eng_div + W1 * W2 * attr_eng_div)/r; //+ attr_eng * Wrij_div * (W1 * W2_div + W2 * W1_div)) / r;
        }
        force_divr += Scalar(2.0) * (Wrij_div * W1_div * W2 * attr_eng) / r;
    }


////////////////////////////////////////////// NEWLY ADDED FUNCTION


    DEVICE void evalSelfEnergy(Scalar& energy, Scalar phi)
        {
        Scalar rho_i = phi;

        // *excess* free energy of a vdW fluid (subtract ideal gas contribution)
        energy = Scalar(0.5) * B * (rho_i - A) * (rho_i - A);
        }

    //! Evaluate the forces due to ijk interactions
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
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("squared_density");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rij_sq; //!< Stored rij_sq from the constructor
    Scalar attr_rcut;
    Scalar rep_rcut;
    Scalar density_rcut;
    Scalar ncut_ij;
    Scalar ncut_ji;
    Scalar A;      //!< attraction parameter
    Scalar B;      //!< repulsion parameter

    };

    } // end namespace md
    } // end namespace hoomd

#endif

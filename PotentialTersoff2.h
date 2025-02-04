// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __POTENTIAL_TERSOFF_H__
#define __POTENTIAL_TERSOFF_H__

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "hoomd/md/NeighborList.h"
#include "hoomd/SystemDefinition.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

/*! \file PotentialTersoff.h
    \brief Defines the template class for standard three-body potentials
    \details The heart of the code that computes three-body potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Template class for computing three-body potentials
/*! <b>Overview:</b>
    PotentialTersoff computes standard three-body potentials and forces between all particles in the
    simulation.  It employs the use of a neighbor list to limit the number of computations done to
    only those particles within the cutoff radius of each other.  the computation of the actual
    potential is not performed directly by this class, but by an evaluator class (e.g.
    EvaluatorTersoff) which is passed in as a template parameter so the computations are performed
    as efficiently as possible.

    PotentialTersoff handles most of the internal details common to all standard three-body
   potentials.
     - A cutoff radius to be specified per particle type-pair
     - Per type-pair parameters are stored and a set method is provided
     - All the details about looping through the particles, computing dr, computing the virial, etc.
   are handled

    <b>Implementation details</b>

    Unlike the pair potentials, the three-body potentials offer two force directions: ij and ik.
    In addition, some three-body potentials (such as the Tersoff potential) compute unique forces on
    each of the three particles involved.  Three-body evaluators must thus return six force
   magnitudes: two for each particle.  These values are returned in the Scalar3 values \a
   force_divr_ij and \a force_divr_ik.  The x components refer to particle i, y to particle j, and z
   to particle k. If your particular three-body potential does not compute one of these forces, then
   the evaluator can simply return 0 for that force.  In addition, the potential energy is stored in
   the w component of force_divr_ij.

    rcutsq, ronsq, and the params are stored per particle type-pair. It wastes a little bit of
   space, but benchmarks show that storing the symmetric type pairs and indexing with Index2D is
   faster than not storing redundant pairs and indexing with Index2DUpperTriangular. All of these
   values are stored in GPUArray for easy access on the GPU by a derived class. The type of the
   parameters is defined by \a param_type in the potential evaluator class passed in. See the
   appropriate documentation for the evaluator for the definition of each element of the parameters.

    \sa export_PotentialTersoff()
*/
template<class evaluator> class PYBIND11_EXPORT PotentialTersoff2 : public hoomd::ForceCompute
    {
    public:
    //! Param type from evaluator
    typedef typename evaluator::param_type param_type;

    //! Construct the potential
    PotentialTersoff2(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~PotentialTersoff2();

    //! Set the pair parameters for a single type pair
    virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type& param);
    virtual void setParamsPython(pybind11::tuple typ, pybind11::dict params);
    /// Get params for a single type pair using a tuple of strings
    virtual pybind11::dict getParams(pybind11::tuple typ);
    //! Set the rcut for a single type pair
    virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);
    //! Get the rcut for a single type pair
    Scalar getRCut(pybind11::tuple types);
    /// Set the rcut for a single type pair using a tuple of strings
    virtual void setRCutPython(pybind11::tuple types, Scalar r_cut);

    /// Validate that types are within Ntypes
    virtual void validateTypes(unsigned int typ1, unsigned int typ2, std::string action);

    virtual void notifyDetach()
        {
        if (m_attached)
            {
            m_nlist->removeRCutMatrix(m_r_cut_nlist);
            }
        m_attached = false;
        }

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    virtual CommFlags getRequestedCommFlags(uint64_t timestep);
#endif

    /// Start autotuning kernel launch parameters
    virtual void startAutotuning()
        {
        hoomd::ForceCompute::startAutotuning();

        // Start autotuning the neighbor list.
        m_nlist->startAutotuning();
        }

    /// Check if autotuning is complete.
    virtual bool isAutotuningComplete()
        {
        bool result = hoomd::ForceCompute::isAutotuningComplete();
        return result && m_nlist->isAutotuningComplete();
        }

    protected:
    std::shared_ptr<NeighborList> m_nlist; //!< The neighborlist to use for the computation
    Index2D m_typpair_idx;                 //!< Helper class for indexing per type pair arrays
    GPUArray<Scalar> m_rcutsq;             //!< Cutoff radius squared per type pair
    GPUArray<param_type> m_params;         //!< Pair parameters per type pair

    // track whether we are attached to the simulation
    bool m_attached = true;

    // r_cut (not squared) given to the neighborlist
    std::shared_ptr<GlobalArray<Scalar>> m_r_cut_nlist;

    // scratch pad memory per type
////////////////////////////////////////////// NEWLY ADDED CODE
    // std::vector<Scalar> m_phi_ab;
    std::vector<std::vector<Scalar>> m_phi_ab_arr;
////////////////////////////////////////////// NEWLY ADDED CODE

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
*/


////////////////////////////////////////////// NEWLY ADDED INITIALIZATION
// template<class evaluator>
// PotentialTersoff2<evaluator>::PotentialTersoff2(std::shared_ptr<SystemDefinition> sysdef,
//                                               std::shared_ptr<NeighborList> nlist)
//     : hoomd::ForceCompute(sysdef), m_nlist(nlist), m_typpair_idx(m_pdata->getNTypes()),
//       m_phi_ab(m_pdata->getNTypes())
//     {
template<class evaluator>
PotentialTersoff2<evaluator>::PotentialTersoff2(std::shared_ptr<SystemDefinition> sysdef,
                                              std::shared_ptr<NeighborList> nlist)
    : hoomd::ForceCompute(sysdef), m_nlist(nlist), m_typpair_idx(m_pdata->getNTypes()),
      m_phi_ab_arr(m_pdata->getN(), std::vector<Scalar>(m_pdata->getNTypes()))
    {
////////////////////////////////////////////// NEWLY ADDED INITIALIZATION
    this->m_exec_conf->msg->notice(5) << "Constructing PotentialTersoff2" << std::endl;

    assert(m_pdata);
    assert(m_nlist);

    GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_rcutsq.swap(rcutsq);
    GPUArray<param_type> params(m_typpair_idx.getNumElements(), m_exec_conf);
    m_params.swap(params);

    m_r_cut_nlist
        = std::make_shared<GlobalArray<Scalar>>(m_typpair_idx.getNumElements(), m_exec_conf);
    nlist->addRCutMatrix(m_r_cut_nlist);
    }

template<class evaluator> PotentialTersoff2<evaluator>::~PotentialTersoff2()
    {
    this->m_exec_conf->msg->notice(5) << "Destroying PotentialTersoff2" << std::endl;
    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param param Parameter to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
template<class evaluator>
void PotentialTersoff2<evaluator>::setParams(unsigned int typ1,
                                            unsigned int typ2,
                                            const param_type& param)
    {
    validateTypes(typ1, typ2, "set params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[m_typpair_idx(typ1, typ2)] = param;
    h_params.data[m_typpair_idx(typ2, typ1)] = param;
    }
// I guess here we nned to change the the typ1 and typ2 parameters to make it asymetric
template<class evaluator>
void PotentialTersoff2<evaluator>::setParamsPython(pybind11::tuple typ, pybind11::dict params)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    validateTypes(typ1, typ2, "set params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[m_typpair_idx(typ1, typ2)] = param_type(params);
    h_params.data[m_typpair_idx(typ2, typ1)] = param_type(params);
    }

template<class evaluator> pybind11::dict PotentialTersoff2<evaluator>::getParams(pybind11::tuple typ)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    validateTypes(typ1, typ2, "get params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    return h_params.data[m_typpair_idx(typ1, typ2)].asDict();
    }

/*! \param typ1 First type index in the pair
    \param typ2 Second type index in the pair
    \param rcut Cutoff radius to set
    \note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   automatically set.
*/
template<class evaluator>
void PotentialTersoff2<evaluator>::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    validateTypes(typ1, typ2, "set r_cut");
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
    h_rcutsq.data[m_typpair_idx(typ1, typ2)] = rcut * rcut;
    h_rcutsq.data[m_typpair_idx(typ2, typ1)] = rcut * rcut;

    ArrayHandle<Scalar> h_r_cut_nlist(*m_r_cut_nlist,
                                      access_location::host,
                                      access_mode::readwrite);
    h_r_cut_nlist.data[m_typpair_idx(typ1, typ2)] = rcut;
    h_r_cut_nlist.data[m_typpair_idx(typ2, typ1)] = rcut;
    // std::cout << "setRcut called for types: " << typ1 << ", " << typ2 << " with rcut = " << rcut << std::endl;
    m_nlist->notifyRCutMatrixChange();
    }

template<class evaluator>
void PotentialTersoff2<evaluator>::setRCutPython(pybind11::tuple types, Scalar r_cut)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "set r_cut");
    setRcut(typ1, typ2, r_cut);
    }

template<class evaluator>
void PotentialTersoff2<evaluator>::validateTypes(unsigned int typ1,
                                                unsigned int typ2,
                                                std::string action)
    {
    // TODO change logic to just throw an exception
    auto n_types = this->m_pdata->getNTypes();
    if (typ1 >= n_types || typ2 >= n_types)
        {
        this->m_exec_conf->msg->error()
            << "pair." << evaluator::getName() << ": Trying to " << action
            << " for a non existent type! " << typ1 << "," << typ2 << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialTersoff2");
        }
    }

template<class evaluator> Scalar PotentialTersoff2<evaluator>::getRCut(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "get rcut.");
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    return sqrt(h_rcutsq.data[m_typpair_idx(typ1, typ2)]);
    }

/*! \post The forces are computed for the given timestep. The neighborlist's compute method is
   called to ensure that it is up to date before proceeding.

    \param timestep specifies the current time step of the simulation
*/
template<class evaluator> void PotentialTersoff2<evaluator>::computeForces(uint64_t timestep)
    {
    // ****** Tersoff or SquareDensity potential
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // The three-body potentials can't handle a half neighbor list, so check now.
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        m_exec_conf->msg->error()
            << std::endl
            << "PotentialTersoff2 cannot handle a half neighborlist" << std::endl;
        throw std::runtime_error("Error computing forces in PotentialTersoff2");
        }

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);

    // force and virial arrays
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    const BoxDim box = m_pdata->getBox();
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

    // need to start from a zero force, energy
    memset(h_force.data, 0, sizeof(Scalar4) * (m_pdata->getN() + m_pdata->getNGhosts()));
    memset(h_virial.data, 0, sizeof(Scalar) * 6 * m_virial_pitch);

    unsigned int ntypes = m_pdata->getNTypes();

    
////////////////////////////////////////////// NEWLY ADDED CODE

    // looping over all particles to save the phi in an array
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 posi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const size_t head_i = h_head_list.data[i];
        // sanity check
        assert(typei < m_pdata->getNTypes());
        
        // setting phi for each particle type to 0
        for (unsigned int typ_b = 0; typ_b < ntypes; ++typ_b)
            {
            m_phi_ab_arr[i][typ_b] = Scalar(0.0);
            }
        
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        if (evaluator::hasPerParticleEnergy())
            {
            for (unsigned int j = 0; j < size; j++)
                {
                // access the index of neighbor j (MEM TRANSFER: 1 scalar)
                unsigned int jj = h_nlist.data[head_i + j];
                assert(jj < m_pdata->getN() + m_pdata->getNGhosts());

                // access the position and type of particle j
                Scalar3 posj
                    = make_scalar3(h_pos.data[jj].x, h_pos.data[jj].y, h_pos.data[jj].z);
                unsigned int typej = __scalar_as_int(h_pos.data[jj].w);
                assert(typej < m_pdata->getNTypes());

                // calculate dr_ij (MEM TRANSFER: 3 scalars / FLOPS: 3)
                Scalar3 dxij = posi - posj;

                // apply periodic boundary conditions
                dxij = box.minImage(dxij);

                // compute rij_sq (FLOPS: 5)
                Scalar rij_sq = dot(dxij, dxij);

                // get parameters for this type pair
                unsigned int typpair_idx = m_typpair_idx(typei, typej);
                const param_type& param = h_params.data[typpair_idx];
                Scalar rcutsq = h_rcutsq.data[typpair_idx];

                // evaluate the scalar per-neighbor contribution
                evaluator eval(rij_sq, param);
                eval.evalPhi(m_phi_ab_arr[i][typej]);

                // std::cout << m_phi_ab_arr[i][typej] << " ";
                }
            }
        // std::cout << std::endl;
        // std::cout << i << " ";
        // for (unsigned int typ_b = 0; typ_b < ntypes; ++typ_b)
        //     {
        //         std::cout << m_phi_ab_arr[i][typ_b] << " ";
        //     }
        //     std::cout << std::endl;

        }

    
    // 2nd loop for each particle
    // NOTE: here also we need to calculate the phi again but using the phi of both i and j
    //
    /*
    for (int i = 0; i < (int)m_pdata->getN(); i++){
        std::cout << i << " ";
        for (unsigned int typ_b = 0; typ_b < ntypes; ++typ_b)
            {
            std::cout << "Density" << m_phi_ab_arr[i][typ_b] << " ";
            
            }
        const size_t head_i = h_head_list.data[i];
        std::cout << "Nbrs" << std::endl;
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int j = 0; j < size; j++)
                {
                // access the index of neighbor j (MEM TRANSFER: 1 scalar)
                unsigned int jj = h_nlist.data[head_i + j];
                Scalar3 posi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
                unsigned int typei = __scalar_as_int(h_pos.data[i].w);
                Scalar3 posj = make_scalar3(h_pos.data[jj].x, h_pos.data[jj].y, h_pos.data[jj].z);
                unsigned int typej = __scalar_as_int(h_pos.data[jj].w);
      
                // calculate dr_ij (MEM TRANSFER: 3 scalars / FLOPS: 3)
                Scalar3 dxij = posi - posj;

                // apply periodic boundary conditions
                dxij = box.minImage(dxij);

                // compute rij_sq (FLOPS: 5)
                Scalar rij_sq = dot(dxij, dxij);
                Scalar rij_abs = fast::sqrt(rij_sq);
                std::cout << jj << " - " << rij_abs << " "; 
            
        }
            
        std::cout << std::endl;
    }
    */
    
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        ///std::cout << i <<std::endl;
        Scalar3 posi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const size_t head_i = h_head_list.data[i];

        Scalar3 fi = make_scalar3(0.0, 0.0, 0.0);
        Scalar pei = 0.0;     

        const unsigned int size = (unsigned int)h_n_neigh.data[i];
////////////////////////////////////////////// NEWLY ADDED CODE
        // loop over all of the neighbors of this particle
        for (unsigned int j = 0; j < size; j++)
            {
            // access the index of neighbor j (MEM TRANSFER: 1 scalar)
            unsigned int jj = h_nlist.data[head_i + j];
            assert(jj < m_pdata->getN() + m_pdata->getNGhosts());
            
            


            
            // access the position and type of particle j
            Scalar3 posj = make_scalar3(h_pos.data[jj].x, h_pos.data[jj].y, h_pos.data[jj].z);
            unsigned int typej = __scalar_as_int(h_pos.data[jj].w);
            assert(typej < m_pdata->getNTypes());

            // initialize the current force and potential energy of particle j to 0
            Scalar3 fj = make_scalar3(0.0, 0.0, 0.0);
            Scalar pej = 0.0;

            // calculate dr_ij (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar3 dxij = posi - posj;

            // apply periodic boundary conditions
            dxij = box.minImage(dxij);

            // compute rij_sq (FLOPS: 5)
            Scalar rij_sq = dot(dxij, dxij);
            Scalar rij_abs = fast::sqrt(rij_sq);
            // get parameters for this type pair
            unsigned int typpair_idx = m_typpair_idx(typei, typej);
            const param_type& param = h_params.data[typpair_idx];
            Scalar rcutsq = h_rcutsq.data[typpair_idx];

            Scalar force_divr = Scalar(0.0);
            Scalar potential_eng = Scalar(0.0);
            
            
            
            

            

            for (unsigned int k = 0; k < size; k++)
                {
                
  
                
                unsigned int kk = h_nlist.data[head_i + k];
                assert(kk < m_pdata->getN() + m_pdata->getNGhosts());
                

                
                // access the position and type of particle k
                Scalar3 posk = make_scalar3(h_pos.data[kk].x, h_pos.data[kk].y, h_pos.data[kk].z);
                unsigned int typek = __scalar_as_int(h_pos.data[kk].w);
                assert(typek < m_pdata->getNTypes());

                Scalar3 dxik = posi - posk;

                // apply periodic boundary conditions
                dxik = box.minImage(dxik);

                // compute rik_sq (FLOPS: 5)
                Scalar rik_sq = dot(dxik, dxik);

                // get parameters for this type pair
                unsigned int typpair_idx_ik = m_typpair_idx(typei, typek);
                const param_type& param_ik = h_params.data[typpair_idx_ik];
                Scalar rcutsq_ik = h_rcutsq.data[typpair_idx_ik];
                
                // evaluator eval(rij_sq, rcutsq, param);
                evaluator evalij(rij_sq, param);
                evaluator evalik(rik_sq, param_ik);                

                if (typej == typek)
                    {
                    Scalar rep_eng = Scalar(0.0);
                    Scalar rep_eng_div = Scalar(0.0);
                    Scalar attr_eng = Scalar(0.0);
                    Scalar attr_eng_div = Scalar(0.0);
                    evalik.evalRepulsive(rep_eng, rep_eng_div);
                    evalik.evalAttractive(attr_eng, attr_eng_div);

                    Scalar W1 = Scalar(0.0);
                    Scalar W1_div = Scalar(0.0);
                    Scalar W2 = Scalar(0.0);
                    Scalar W2_div = Scalar(0.0);
                    evalik.evalOmega(W1, m_phi_ab_arr[i][typek]);
                    evalik.evalOmegaDiv(W1_div, m_phi_ab_arr[i][typek]);
                    evalik.evalOmegaji(W2, m_phi_ab_arr[kk][typei]);
                    evalik.evalOmegajiDiv(W2_div, m_phi_ab_arr[kk][typei]);

                    Scalar Wrij_div = Scalar(0.0);
                    evalij.evalPhiDiv(Wrij_div);
                    
                    // print all the parameters of forceijk
                    
                    // std::cout << i << " " << jj << " " << kk << std::endl;
                    // std::cout << i << " density " << m_phi_ab_arr[i][typek] << std::endl;
                    // std::cout << kk << " density " << m_phi_ab_arr[k][typek] << std::endl;
                    // std::cout << "force_divr: " << force_divr << std::endl;
                    // std::cout << "potential_eng: " << potential_eng << std::endl;
                    // std::cout << "rij_sq: " << rij_sq << std::endl;
                    // std::cout << "jj: " << jj << std::endl;
                    // std::cout << "kk: " << kk << std::endl;
                    // std::cout << "rep_eng: " << rep_eng << std::endl;
                    // std::cout << "rep_eng_div: " << rep_eng_div << std::endl;
                    // std::cout << "attr_eng: " << attr_eng << std::endl;
                    // std::cout << "attr_eng_div: " << attr_eng_div << std::endl;
                    // std::cout << "W1: " << W1 << std::endl;
                    // std::cout << "W1_div: " << W1_div << std::endl;
                    // std::cout << "W2: " << W2 << std::endl;
                    // std::cout << "W2_div: " << W2_div << std::endl;
                    // std::cout << "Wrij_div: " << Wrij_div << std::endl;
                    // std::cout << std::endl;
                    // std::cout << "New set" << i << " " << jj << " " << kk << std::endl;

                    evalik.evalForceijk(force_divr, 
                                        potential_eng,
                                        jj,
                                        kk,
                                        rep_eng,
                                        rep_eng_div,
                                        attr_eng,
                                        attr_eng_div,
                                        W1,
                                        W1_div,
                                        W2,
                                        W2_div,
                                        Wrij_div);
                    }
                }

                fi += Scalar(-1.0) * dxij * (force_divr / rij_abs); // sign of force might be different TODO: check in simulation !!!!!!!!!!Scalar(-1.0) * 
                pei += potential_eng;

                fj += dxij * (force_divr / rij_abs);
                pej += potential_eng;

                // increment the force and potential energy for particle j
                unsigned int mem_idx = jj;
                h_force.data[mem_idx].x += fj.x;
                h_force.data[mem_idx].y += fj.y;
                h_force.data[mem_idx].z += fj.z;
                h_force.data[mem_idx].w += pej;
            }

            // finally, increment the force and potential energy for particle i
            unsigned int mem_idx = i;
            h_force.data[mem_idx].x += fi.x;
            h_force.data[mem_idx].y += fi.y;
            h_force.data[mem_idx].z += fi.z;
            h_force.data[mem_idx].w += pei;
        }

    }


#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template<class evaluator>
CommFlags PotentialTersoff2<evaluator>::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = CommFlags(0);

    flags |= hoomd::ForceCompute::getRequestedCommFlags(timestep);

    // enable reverse communication of forces
    flags[comm_flag::reverse_net_force] = 1;

    // reverse net force requires tags
    flags[comm_flag::tag] = 1;

    return flags;
    }
#endif

// namespace detail
    // {
//! Export this triplet potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialTersoff2(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialTersoff2<T>, hoomd::ForceCompute, std::shared_ptr<PotentialTersoff2<T>>>(
        m,
        name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("setParams", &PotentialTersoff2<T>::setParamsPython)
        .def("getParams", &PotentialTersoff2<T>::getParams)
        .def("setRCut", &PotentialTersoff2<T>::setRCutPython)
        .def("getRCut", &PotentialTersoff2<T>::getRCut);
    }

    // } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif

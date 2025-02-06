// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __POTENTIAL_VALENCY_GPU_H__
#define __POTENTIAL_VALENCY_GPU_H__

##ifdef ENABLE_HIP

#include <memory>

#include "PotentialValency.h"
#include "PotentialValencyGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file PotentialValencyGPU.h
    \brief Defines the template class computing certain three-body forces on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {
//! Template class for computing three-body potentials and forces on the GPU
/*! Derived from PotentialValency, this class provides exactly the same interface for computing
    the three-body potentials and forces.  In the same way as PotentialValency, this class serves
    as a shell dealing with all the details of looping while the evaluator actually computes the
    potential and forces.

    \tparam evaluator Evaluator class used to evaluate V(r) and F(r)/r
    \tparam gpu_cgpf Driver function that calls gpu_compute_valency_forces<evaluator>()

    \sa export_PotentialValencyGPU()
*/
template<class evaluator> class PotentialValencyGPU : public PotentialValency<evaluator>
    {
    public:
    //! Construct the potential
    PotentialValencyGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~PotentialValencyGPU();

    protected:
    std::shared_ptr<Autotuner<2>> m_tuner; //!< Autotuner for block size

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

template<class evaluator>
PotentialValencyGPU<evaluator>::PotentialValencyGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                    std::shared_ptr<NeighborList> nlist)
    : PotentialValency<evaluator>(sysdef, nlist)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing PotentialValencyGPU" << std::endl;

    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "***Error! Creating a PotentialValencyGPU with no GPU in the execution configuration"
            << std::endl;
        throw std::runtime_error("Error initializing PotentialValencyGPU");
        }

    // Initialize autotuner.
    m_tuner.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf),
                                    AutotunerBase::getTppListPow2(this->m_exec_conf)},
                                   this->m_exec_conf,
                                   "pair_Valency"));
    this->m_autotuners.push_back(m_tuner);
    }

template<class evaluator> PotentialValencyGPU<evaluator>::~PotentialValencyGPU()
    {
    this->m_exec_conf->msg->notice(5) << "Destroying PotentialValencyGPU" << std::endl;
    }

template<class evaluator> void PotentialValencyGPU<evaluator>::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error()
            << "***Error! PotentialValencyGPU cannot handle a half neighborlist" << std::endl;
        throw std::runtime_error("Error computing forces in PotentialValencyGPU");
        }

    // access the neighbor list
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(),
                                        access_location::device,
                                        access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(),
                                      access_location::device,
                                      access_mode::read);
    ArrayHandle<size_t> d_head_list(this->m_nlist->getHeadList(),
                                    access_location::device,
                                    access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);

    BoxDim box = this->m_pdata->getBox();

    // access parameters
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params,
                                                         access_location::device,
                                                         access_mode::read);

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    m_tuner->begin();
    auto param = m_tuner->getParam();
    unsigned int block_size = param[0];
    unsigned int threads_per_particle = param[1];

    kernel::gpu_compute_triplet_forces<evaluator>(
        kernel::valency_args_t(d_force.data,
                               this->m_pdata->getN(),
                               this->m_pdata->getNGhosts(),
                               d_virial.data,
                               this->m_virial_pitch,
                               compute_virial,
                               d_pos.data,
                               box,
                               d_n_neigh.data,
                               d_nlist.data,
                               d_head_list.data,
                               d_rcutsq.data,
                               this->m_nlist->getNListArray().getPitch(),
                               this->m_pdata->getNTypes(),
                               block_size,
                               threads_per_particle,
                               this->m_exec_conf->dev_prop),
        d_params.data);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner->end();
    }

namespace detail
    {
//! Export this three-body potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialValencyGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialValencyGPU<T>,
                     PotentialValency<T>,
                     std::shared_ptr<PotentialValencyGPU<T>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif

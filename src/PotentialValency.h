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

#include "HoomdCompat.h"

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {

template<class evaluator> class PYBIND11_EXPORT PotentialValency : public hoomd::ForceCompute
    {
    public:

    typedef typename evaluator::param_type param_type;

    PotentialValency(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<NeighborList> nlist);

    virtual ~PotentialValency();

    virtual void setParams(unsigned int typ1, unsigned int typ2, const param_type& param);
    virtual void setParamsPython(pybind11::tuple typ, pybind11::dict params);

    virtual pybind11::dict getParams(pybind11::tuple typ);

    virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);

    Scalar getRCut(pybind11::tuple types);

    virtual void setRCutPython(pybind11::tuple types, Scalar r_cut);

    virtual void validateTypes(unsigned int typ1, unsigned int typ2, std::string action);

    virtual void notifyDetach()
        {
        if (m_attached)
            {
            m_nlist->removeRCutMatrix(m_r_cut_nlist);
            }
        m_attached = false;
        }

    //! Reject configurations whose forces would be silently wrong.
    /*! Under MPI domain decomposition the density sweep fills phi for owned
        particles only; ghost neighbours would be read with phi = 0, and a zero
        density means a fully open valency gate - bonds crossing a domain
        boundary would never be capped. Refuse instead of miscomputing. */
    void checkSupported()
        {
        if (m_pdata->getNGhosts() != 0)
            {
            throw std::runtime_error(
                "vbc_plugin does not support MPI domain decomposition: ghost "
                "particle densities are not communicated between ranks.");
            }
        }

#ifdef ENABLE_MPI

    virtual CommFlags getRequestedCommFlags(uint64_t timestep);
#endif

    virtual void startAutotuning()
        {
        hoomd::ForceCompute::startAutotuning();

        m_nlist->startAutotuning();
        }

    virtual bool isAutotuningComplete()
        {
        bool result = hoomd::ForceCompute::isAutotuningComplete();
        return result && m_nlist->isAutotuningComplete();
        }

    protected:
    std::shared_ptr<NeighborList> m_nlist;
    Index2D m_typpair_idx;
    GPUArray<Scalar> m_rcutsq;
    GPUArray<param_type> m_params;

    bool m_attached = true;

    std::shared_ptr<VBCRCutArray<Scalar>> m_r_cut_nlist;

    GPUArray<Scalar> m_phi_ab_arr;

    //! Many-body chain-rule sums S, same per-particle row layout as m_phi_ab_arr.
    GPUArray<Scalar> m_s_ab_arr;

    virtual void computeForces(uint64_t timestep);
    };

template<class evaluator>
PotentialValency<evaluator>::PotentialValency(std::shared_ptr<SystemDefinition> sysdef,
                                              std::shared_ptr<NeighborList> nlist)
    : hoomd::ForceCompute(sysdef), m_nlist(nlist), m_typpair_idx(m_pdata->getNTypes())
    {

    this->m_exec_conf->msg->notice(5) << "Constructing PotentialValency" << std::endl;

    assert(m_pdata);
    assert(m_nlist);

    GPUArray<Scalar> rcutsq(m_typpair_idx.getNumElements(), m_exec_conf);
    m_rcutsq.swap(rcutsq);
    GPUArray<param_type> params(m_typpair_idx.getNumElements(), m_exec_conf);
    m_params.swap(params);

    m_r_cut_nlist
        = std::make_shared<VBCRCutArray<Scalar>>(m_typpair_idx.getNumElements(), m_exec_conf);
    nlist->addRCutMatrix(m_r_cut_nlist);

    GPUArray<Scalar> phi_ab_arr((m_pdata->getN()+m_pdata->getNGhosts()) * (m_pdata->getNTypes() + 1), this->m_exec_conf);
    m_phi_ab_arr.swap(phi_ab_arr);
    GPUArray<Scalar> s_ab_arr((m_pdata->getN()+m_pdata->getNGhosts()) * (m_pdata->getNTypes() + 1), this->m_exec_conf);
    m_s_ab_arr.swap(s_ab_arr);

    }

template<class evaluator> PotentialValency<evaluator>::~PotentialValency()
    {
    this->m_exec_conf->msg->notice(5) << "Destroying PotentialValency" << std::endl;
    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    }

template<class evaluator>
void PotentialValency<evaluator>::setParams(unsigned int typ1,
                                            unsigned int typ2,
                                            const param_type& param)
    {
    validateTypes(typ1, typ2, "set params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[m_typpair_idx(typ1, typ2)] = param;

    if (typ1 != typ2)
        {
        param_type param_swapped = param;
        Scalar tmp = param_swapped.ncut_ij;
        param_swapped.ncut_ij = param_swapped.ncut_ji;
        param_swapped.ncut_ji = tmp;
        h_params.data[m_typpair_idx(typ2, typ1)] = param_swapped;
        }
    else
        {
        h_params.data[m_typpair_idx(typ2, typ1)] = param;
        }
    }
template<class evaluator>
void PotentialValency<evaluator>::setParamsPython(pybind11::tuple typ, pybind11::dict params)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    validateTypes(typ1, typ2, "set params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    param_type param_in(params);
    h_params.data[m_typpair_idx(typ1, typ2)] = param_in;

    if (typ1 != typ2)
        {
        param_type param_swapped = param_in;
        Scalar tmp = param_swapped.ncut_ij;
        param_swapped.ncut_ij = param_swapped.ncut_ji;
        param_swapped.ncut_ji = tmp;
        h_params.data[m_typpair_idx(typ2, typ1)] = param_swapped;
        }
    else
        {
        h_params.data[m_typpair_idx(typ2, typ1)] = param_in;
        }
    }

template<class evaluator> pybind11::dict PotentialValency<evaluator>::getParams(pybind11::tuple typ)
    {
    auto typ1 = m_pdata->getTypeByName(typ[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(typ[1].cast<std::string>());
    validateTypes(typ1, typ2, "get params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    return h_params.data[m_typpair_idx(typ1, typ2)].asDict();
    }

template<class evaluator>
void PotentialValency<evaluator>::setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
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

    m_nlist->notifyRCutMatrixChange();
    }

template<class evaluator>
void PotentialValency<evaluator>::setRCutPython(pybind11::tuple types, Scalar r_cut)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "set r_cut");
    setRcut(typ1, typ2, r_cut);
    }

template<class evaluator>
void PotentialValency<evaluator>::validateTypes(unsigned int typ1,
                                                unsigned int typ2,
                                                std::string action)
    {

    auto n_types = this->m_pdata->getNTypes();
    if (typ1 >= n_types || typ2 >= n_types)
        {
        this->m_exec_conf->msg->error()
            << "pair." << evaluator::getName() << ": Trying to " << action
            << " for a non existent type! " << typ1 << "," << typ2 << std::endl;
        throw std::runtime_error("Error setting parameters in PotentialValency");
        }
    }

template<class evaluator> Scalar PotentialValency<evaluator>::getRCut(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateTypes(typ1, typ2, "get rcut.");
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    return sqrt(h_rcutsq.data[m_typpair_idx(typ1, typ2)]);
    }

/*! CPU path, structured identically to the GPU kernels: a density sweep, an S
    sweep, then a one-sided force sweep that reads the phi and S rows of both
    pair partners - so the two backends can be compared term by term. */
template<class evaluator> void PotentialValency<evaluator>::computeForces(uint64_t timestep)
    {

    m_nlist->compute(timestep);
    checkSupported();

    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        m_exec_conf->msg->error()
            << std::endl
            << "PotentialValency cannot handle a half neighborlist" << std::endl;
        throw std::runtime_error("Error computing forces in PotentialValency");
        }

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

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_phi(m_phi_ab_arr, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_s(m_s_ab_arr, access_location::host, access_mode::overwrite);

    const BoxDim box = m_pdata->getBox();
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

    PDataFlags flags = this->m_pdata->getFlags();
    const bool compute_virial = flags[pdata_flag::pressure_tensor];

    // Forces, phi, and S are overwritten below; the virial rows are written
    // only on steps where pressure is requested, so zero them up front.
    memset(h_virial.data, 0, sizeof(Scalar) * 6 * m_virial_pitch);

    const unsigned int N = m_pdata->getN();
    const unsigned int ntypes = m_pdata->getNTypes();
    const unsigned int row = ntypes + 1;

    // Sweep 1: per-type smooth neighbour counts (densities).
    for (unsigned int i = 0; i < N; i++)
        {
        Scalar3 posi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        const unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const size_t head_i = h_head_list.data[i];

        for (unsigned int b = 0; b < row; ++b)
            h_phi.data[i * row + b] = Scalar(0.0);

        Scalar total = Scalar(0.0);
        for (unsigned int j = 0; j < h_n_neigh.data[i]; j++)
            {
            const unsigned int jj = h_nlist.data[head_i + j];
            assert(jj < m_pdata->getN() + m_pdata->getNGhosts());

            Scalar3 posj = make_scalar3(h_pos.data[jj].x, h_pos.data[jj].y, h_pos.data[jj].z);
            const unsigned int typej = __scalar_as_int(h_pos.data[jj].w);
            const unsigned int typpair = m_typpair_idx(typei, typej);

            Scalar3 dxij = box.minImage(posi - posj);
            Scalar rij_sq = dot(dxij, dxij);
            if (rij_sq >= h_rcutsq.data[typpair])
                continue;

            evaluator eval(rij_sq, h_params.data[typpair]);
            Scalar local_phi = Scalar(0.0);
            eval.evalPhi(local_phi);
            h_phi.data[i * row + typej] += local_phi;
            total += local_phi;
            }
        h_phi.data[i * row + ntypes] = total;
        }

    // Sweep 2: many-body chain-rule sums S (the old per-particle scratch
    // m_cum_dens_ab, now stored for every particle so sweep 3 can read the
    // partner's row).
    for (unsigned int i = 0; i < N; i++)
        {
        Scalar3 posi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        const unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const size_t head_i = h_head_list.data[i];
        const Scalar phi_i_total = h_phi.data[i * row + ntypes];

        for (unsigned int b = 0; b < row; ++b)
            h_s.data[i * row + b] = Scalar(0.0);

        Scalar total = Scalar(0.0);
        for (unsigned int j = 0; j < h_n_neigh.data[i]; j++)
            {
            const unsigned int jj = h_nlist.data[head_i + j];
            Scalar3 posj = make_scalar3(h_pos.data[jj].x, h_pos.data[jj].y, h_pos.data[jj].z);
            const unsigned int typej = __scalar_as_int(h_pos.data[jj].w);
            const unsigned int typpair = m_typpair_idx(typei, typej);

            Scalar3 dxij = box.minImage(posi - posj);
            Scalar rij_sq = dot(dxij, dxij);
            if (rij_sq >= h_rcutsq.data[typpair])
                continue;

            evaluator evalij(rij_sq, h_params.data[typpair]);

            Scalar W1_div = Scalar(0.0);
            evalij.evalOmegaDiv(W1_div, h_phi.data[i * row + typej], phi_i_total);
            if (W1_div == Scalar(0.0))
                continue;

            Scalar attr_eng = Scalar(0.0);
            Scalar attr_eng_div = Scalar(0.0);
            evalij.evalAttractive(attr_eng, attr_eng_div);
            if (attr_eng == Scalar(0.0))
                continue;

            Scalar W2 = Scalar(0.0);
            evalij.evalOmegaji(W2, h_phi.data[jj * row + typei],
                               h_phi.data[jj * row + ntypes]);

            const Scalar contrib = W1_div * W2 * attr_eng;
            h_s.data[i * row + typej] += contrib;
            total += contrib;
            }
        h_s.data[i * row + ntypes] = total;
        }

    // Sweep 3: one-sided forces and per-particle energies. Each particle
    // accumulates its complete force; nothing is scattered to the partner.
    for (unsigned int i = 0; i < N; i++)
        {
        Scalar3 posi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        const unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        const size_t head_i = h_head_list.data[i];

        const Scalar phi_i_total = h_phi.data[i * row + ntypes];
        const Scalar s_i_total = h_s.data[i * row + ntypes];

        Scalar3 fi = make_scalar3(0.0, 0.0, 0.0);
        Scalar pei = Scalar(0.0);
        Scalar vxx = 0, vxy = 0, vxz = 0, vyy = 0, vyz = 0, vzz = 0;

        for (unsigned int j = 0; j < h_n_neigh.data[i]; j++)
            {
            const unsigned int jj = h_nlist.data[head_i + j];
            Scalar3 posj = make_scalar3(h_pos.data[jj].x, h_pos.data[jj].y, h_pos.data[jj].z);
            const unsigned int typej = __scalar_as_int(h_pos.data[jj].w);
            const unsigned int typpair = m_typpair_idx(typei, typej);

            Scalar3 dxij = box.minImage(posi - posj);
            Scalar rij_sq = dot(dxij, dxij);
            if (rij_sq >= h_rcutsq.data[typpair])
                continue;

            evaluator evalij(rij_sq, h_params.data[typpair]);

            Scalar rep_eng = Scalar(0.0);
            Scalar rep_eng_div = Scalar(0.0);
            Scalar attr_eng = Scalar(0.0);
            Scalar attr_eng_div = Scalar(0.0);
            evalij.evalRepulsive(rep_eng, rep_eng_div);
            evalij.evalAttractive(attr_eng, attr_eng_div);

            Scalar W1W2 = Scalar(0.0);
            if (attr_eng != Scalar(0.0) || attr_eng_div != Scalar(0.0))
                {
                Scalar W1 = Scalar(0.0);
                Scalar W2 = Scalar(0.0);
                evalij.evalOmega(W1, h_phi.data[i * row + typej], phi_i_total);
                evalij.evalOmegaji(W2, h_phi.data[jj * row + typei],
                                   h_phi.data[jj * row + ntypes]);
                W1W2 = W1 * W2;
                }

            Scalar dedr = rep_eng_div + W1W2 * attr_eng_div;
            Scalar Wrij_div = Scalar(0.0);
            evalij.evalPhiDiv(Wrij_div);
            if (Wrij_div != Scalar(0.0))
                {
                const Scalar s_i = evalij.pickPhi(h_s.data[i * row + typej], s_i_total);
                const Scalar s_j = evalij.pickPhi(h_s.data[jj * row + typei],
                                                  h_s.data[jj * row + ntypes]);
                dedr += Wrij_div * (s_i + s_j);
                }

            const Scalar force_divr = -dedr * evalij.rinv();

            fi.x += dxij.x * force_divr;
            fi.y += dxij.y * force_divr;
            fi.z += dxij.z * force_divr;
            pei += Scalar(0.5) * (rep_eng + W1W2 * attr_eng);

            // Half the pairwise virial per endpoint; every force term is
            // central (parallel to dxij), the many-body one included.
            if (compute_virial)
                {
                vxx += Scalar(0.5) * dxij.x * dxij.x * force_divr;
                vxy += Scalar(0.5) * dxij.x * dxij.y * force_divr;
                vxz += Scalar(0.5) * dxij.x * dxij.z * force_divr;
                vyy += Scalar(0.5) * dxij.y * dxij.y * force_divr;
                vyz += Scalar(0.5) * dxij.y * dxij.z * force_divr;
                vzz += Scalar(0.5) * dxij.z * dxij.z * force_divr;
                }
            }

        h_force.data[i].x = fi.x;
        h_force.data[i].y = fi.y;
        h_force.data[i].z = fi.z;
        h_force.data[i].w = pei;

        if (compute_virial)
            {
            h_virial.data[0 * m_virial_pitch + i] = vxx;
            h_virial.data[1 * m_virial_pitch + i] = vxy;
            h_virial.data[2 * m_virial_pitch + i] = vxz;
            h_virial.data[3 * m_virial_pitch + i] = vyy;
            h_virial.data[4 * m_virial_pitch + i] = vyz;
            h_virial.data[5 * m_virial_pitch + i] = vzz;
            }
        }

    }

#ifdef ENABLE_MPI

template<class evaluator>
CommFlags PotentialValency<evaluator>::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = CommFlags(0);

    flags |= hoomd::ForceCompute::getRequestedCommFlags(timestep);

    flags[comm_flag::tag] = 1;

    return flags;
    }
#endif

namespace detail
    {

template<class T> void export_PotentialValency(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialValency<T>, hoomd::ForceCompute, std::shared_ptr<PotentialValency<T>>>(
        m,
        name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("setParams", &PotentialValency<T>::setParamsPython)
        .def("getParams", &PotentialValency<T>::getParams)
        .def("setRCut", &PotentialValency<T>::setRCutPython)
        .def("getRCut", &PotentialValency<T>::getRCut);
    }

    }
    }
    }

#endif

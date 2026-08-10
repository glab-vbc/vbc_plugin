// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/TextureTools.h"
#ifdef __HIPCC__
#include "hoomd/WarpTools.cuh"
#endif
#include <assert.h>
#include <climits>

#ifndef __POTENTIAL_VALENCY_GPU_CUH__
#define __POTENTIAL_VALENCY_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

#if defined(__HIP_PLATFORM_NVCC__)
const int gpu_valency_max_tpp = 32;
#elif defined(__HIP_PLATFORM_HCC__)
const int gpu_valency_max_tpp = 64;
#endif

struct valency_args_t
    {

    valency_args_t(Scalar4* _d_force,
                   const unsigned int _N,
                   const unsigned int _Nghosts,
                   Scalar* _d_virial,
                   size_t _virial_pitch,
                   bool _compute_virial,
                   const Scalar4* _d_pos,
                   float* _d_phi_ab_arr,
                   float* _d_s_ab_arr,
                   const BoxDim& _box,
                   const unsigned int* _d_n_neigh,
                   const unsigned int* _d_nlist,
                   const size_t* _d_head_list,
                   const Scalar* _d_rcutsq,
                   const size_t _size_nlist,
                   const unsigned int _ntypes,
                   const unsigned int _block_size,
                   const unsigned int _tpp,
                   const hipDeviceProp_t& _devprop)
        : d_force(_d_force), N(_N), Nghosts(_Nghosts), d_virial(_d_virial),
          virial_pitch(_virial_pitch), compute_virial(_compute_virial), d_pos(_d_pos),
          d_phi_ab_arr(_d_phi_ab_arr), d_s_ab_arr(_d_s_ab_arr), box(_box),
          d_n_neigh(_d_n_neigh), d_nlist(_d_nlist), d_head_list(_d_head_list), d_rcutsq(_d_rcutsq),
          size_nlist(_size_nlist), ntypes(_ntypes), block_size(_block_size), tpp(_tpp),
          devprop(_devprop) {};

    Scalar4* d_force;
    const unsigned int N;
    const unsigned int Nghosts;
    Scalar* d_virial;
    const size_t virial_pitch;
    bool compute_virial;
    const Scalar4* d_pos;
    float* d_phi_ab_arr;
    float* d_s_ab_arr;
    const BoxDim box;
    const unsigned int*
        d_n_neigh;
    const unsigned int* d_nlist;
    const size_t* d_head_list;
    const Scalar* d_rcutsq;
    const size_t size_nlist;
    const unsigned int ntypes;
    const unsigned int block_size;
    const unsigned int tpp;
    const hipDeviceProp_t& devprop;
    };

#ifdef __HIPCC__

//! Cooperatively stage the per-type-pair params and rcutsq into shared memory.
/*! Must be executed by every thread of the block (call before any early
    return) and followed by __syncthreads() at the call site. */
template<class evaluator>
__device__ inline void stage_params(const typename evaluator::param_type* d_params,
                                    const Scalar* d_rcutsq,
                                    typename evaluator::param_type* s_params,
                                    float* s_rcutsq,
                                    const unsigned int num_typ_parameters)
    {
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            s_rcutsq[cur_offset + threadIdx.x] = float(d_rcutsq[cur_offset + threadIdx.x]);
            s_params[cur_offset + threadIdx.x] = d_params[cur_offset + threadIdx.x];
            }
        }
    }

//! Warp-reduce the per-type tallies of one particle's thread team and store its row.
/*! Each thread owns slab slots [threadIdx.x*ntypes, (threadIdx.x+1)*ntypes);
    the tpp threads of a particle's team hold partial per-type sums. Each type
    is summed across the team with register shuffles (no shared-memory traffic,
    no atomics) and the team leader writes the row
    d_out[idx*(ntypes+1) + typ_b] plus the type-summed total in the last slot. */
template<int tpp>
__device__ inline void reduce_and_store_row(const float* s_slab,
                                            float* d_out,
                                            const unsigned int idx,
                                            const unsigned int ntypes)
    {
    float total = 0.0f;
    for (unsigned int typ_b = 0; typ_b < ntypes; ++typ_b)
        {
        float v = hoomd::detail::WarpReduce<float, tpp>().Sum(
            s_slab[threadIdx.x * ntypes + typ_b]);
        if ((threadIdx.x % tpp) == 0)
            {
            d_out[idx * (ntypes + 1) + typ_b] = v;
            total += v;
            }
        }
    if ((threadIdx.x % tpp) == 0)
        d_out[idx * (ntypes + 1) + ntypes] = total;
    }

template<class evaluator, int tpp>
__global__ void gpu_compute_density_kernel(const unsigned int N,
                                           const Scalar4* d_pos,
                                           float* d_phi_ab_arr,
                                           const BoxDim box,
                                           const unsigned int* d_n_neigh,
                                           const unsigned int* d_nlist,
                                           const size_t* d_head_list,
                                           const typename evaluator::param_type* d_params,
                                           const Scalar* d_rcutsq,
                                           const unsigned int ntypes)
    {
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    HIP_DYNAMIC_SHARED(char, s_data)
    typename evaluator::param_type* s_params
        = (typename evaluator::param_type*)(&s_data[0]);
    float* s_rcutsq
        = (float*)(&s_data[num_typ_parameters * sizeof(typename evaluator::param_type)]);
    float* s_slab = s_rcutsq + num_typ_parameters;

    stage_params<evaluator>(d_params, d_rcutsq, s_params, s_rcutsq, num_typ_parameters);
    __syncthreads();

    unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    if (idx >= N)
        return;

    // Per-thread, per-type tally. Nothing is shared between threads until the
    // reduction at the end, so no atomics are needed anywhere: the original
    // issued one global atomicAdd per neighbour, with every thread of a
    // particle's team serialising on the same few addresses.
    for (unsigned int typ_b = 0; typ_b < ntypes; ++typ_b)
        s_slab[threadIdx.x * ntypes + typ_b] = 0.0f;

    unsigned int n_neigh = d_n_neigh[idx];
    Scalar4 postypei = __ldg(d_pos + idx);
    Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);
    const size_t head_idx = d_head_list[idx];

    for (int neigh_idx = threadIdx.x % tpp; neigh_idx < n_neigh; neigh_idx += tpp)
        {
        unsigned int cur_j = __ldg(d_nlist + head_idx + neigh_idx);

        Scalar4 postypej = __ldg(d_pos + cur_j);
        Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

        // Subtraction and minimum image in double (as everywhere in HOOMD),
        // then narrow the separation once; everything downstream is float.
        Scalar3 dxd = box.minImage(posi - posj);
        const float rij_sq = float(dot(dxd, dxd));

        unsigned int typpair = typpair_idx(__scalar_as_int(postypei.w),
                                           __scalar_as_int(postypej.w));
        // Skip neighbours beyond the pair r_cut (the staged rcutsq was never
        // consulted before, so pairs in the nlist buffer zone were evaluated;
        // all contributions are zero there as long as every window is inside
        // r_cut, which every standard parameter set satisfies).
        if (rij_sq >= s_rcutsq[typpair])
            continue;

        evaluator eval(rij_sq, s_params[typpair]);
        float local_phi = 0.0f;
        eval.evalPhi(local_phi);
        s_slab[threadIdx.x * ntypes + __scalar_as_int(postypej.w)] += local_phi;
        }

    reduce_and_store_row<tpp>(s_slab, d_phi_ab_arr, idx, ntypes);
    }

//! Many-body chain-rule sums S[i][b] = sum over type-b neighbours j of
//! g'(n_i) g(n_j) V_a(r_ij), plus the type total in the last slot.
/*! This is the first internal loop of the old force kernel moved into its own
    launch, with the accumulator redirected from block-shared scratch to a
    per-particle global array laid out exactly like the densities. Once S is
    globally readable, any particle's team can evaluate the complete many-body
    force term for a pair by itself - which is what makes the one-sided,
    atomic-free force kernel below possible. */
template<class evaluator, int tpp>
__global__ void gpu_compute_valency_s_kernel(const unsigned int N,
                                             const Scalar4* d_pos,
                                             const float* d_phi_ab_arr,
                                             float* d_s_ab_arr,
                                             const BoxDim box,
                                             const unsigned int* d_n_neigh,
                                             const unsigned int* d_nlist,
                                             const size_t* d_head_list,
                                             const typename evaluator::param_type* d_params,
                                             const Scalar* d_rcutsq,
                                             const unsigned int ntypes)
    {
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    HIP_DYNAMIC_SHARED(char, s_data)
    typename evaluator::param_type* s_params = (typename evaluator::param_type*)(&s_data[0]);
    float* s_rcutsq
        = (float*)(&s_data[num_typ_parameters * sizeof(typename evaluator::param_type)]);
    float* s_slab = s_rcutsq + num_typ_parameters;

    stage_params<evaluator>(d_params, d_rcutsq, s_params, s_rcutsq, num_typ_parameters);
    __syncthreads();

    unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    if (idx >= N)
        return;

    for (unsigned int typ_b = 0; typ_b < ntypes; ++typ_b)
        s_slab[threadIdx.x * ntypes + typ_b] = 0.0f;

    unsigned int n_neigh = d_n_neigh[idx];
    Scalar4 postypei = __ldg(d_pos + idx);
    Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);
    const unsigned int typei = __scalar_as_int(postypei.w);
    const size_t head_idx = d_head_list[idx];

    const float phi_i_total = d_phi_ab_arr[idx * (ntypes + 1) + ntypes];

    for (int neigh_idx = threadIdx.x % tpp; neigh_idx < n_neigh; neigh_idx += tpp)
        {
        unsigned int cur_j = __ldg(d_nlist + head_idx + neigh_idx);

        Scalar4 postypej = __ldg(d_pos + cur_j);
        Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

        Scalar3 dxd = box.minImage(posi - posj);
        const float rij_sq = float(dot(dxd, dxd));

        const unsigned int typej = __scalar_as_int(postypej.w);
        const unsigned int typpair = typpair_idx(typei, typej);
        if (rij_sq >= s_rcutsq[typpair])
            continue;

        evaluator evalij(rij_sq, s_params[typpair]);

        // Most pairs sit outside the unit-wide gate ramp, where g' is exactly
        // zero; skip the rest of the work there.
        float W1_div = 0.0f;
        evalij.evalOmegaDiv(W1_div, d_phi_ab_arr[idx * (ntypes + 1) + typej],
                            phi_i_total);
        if (W1_div == 0.0f)
            continue;

        float attr_eng = 0.0f;
        float attr_eng_div = 0.0f;
        evalij.evalAttractive(attr_eng, attr_eng_div);
        if (attr_eng == 0.0f)
            continue;

        float W2 = 0.0f;
        evalij.evalOmegaji(W2, d_phi_ab_arr[cur_j * (ntypes + 1) + typei],
                           d_phi_ab_arr[cur_j * (ntypes + 1) + ntypes]);

        s_slab[threadIdx.x * ntypes + typej] += W1_div * W2 * attr_eng;
        }

    reduce_and_store_row<tpp>(s_slab, d_s_ab_arr, idx, ntypes);
    }

//! One-sided force sweep: each particle's team computes its complete force.
/*! With a full neighbour list every pair (i, j) is processed twice, once by
    each particle's team. The old kernel had each visit compute half the pair
    force plus its own particle's share of the many-body term and push the
    result into BOTH particles' force slots - four atomicAdds per neighbour to
    the partner, because the partner's team cannot see this team's s_i scratch.
    With S stored per particle, each visit now computes the full derivative

        dE/dr_ij = Vr' + W1 W2 Va' + w'(r_ij) (S_i + S_j)

    for its own particle only and never writes anywhere else: the team's
    partial forces are summed with warp reductions and the team leader issues
    the single store. No atomics remain, so forces are also bitwise
    reproducible run to run. */
template<class evaluator, unsigned char compute_virial, int tpp>
__global__ void gpu_compute_valency_forces_kernel(Scalar4* d_force,
                                                  const unsigned int N,
                                                  Scalar* d_virial,
                                                  size_t virial_pitch,
                                                  const Scalar4* d_pos,
                                                  const float* d_phi_ab_arr,
                                                  const float* d_s_ab_arr,
                                                  const BoxDim box,
                                                  const unsigned int* d_n_neigh,
                                                  const unsigned int* d_nlist,
                                                  const size_t* d_head_list,
                                                  const typename evaluator::param_type* d_params,
                                                  const Scalar* d_rcutsq,
                                                  const unsigned int ntypes)
    {
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    HIP_DYNAMIC_SHARED(char, s_data)
    typename evaluator::param_type* s_params = (typename evaluator::param_type*)(&s_data[0]);
    float* s_rcutsq
        = (float*)(&s_data[num_typ_parameters * sizeof(typename evaluator::param_type)]);

    stage_params<evaluator>(d_params, d_rcutsq, s_params, s_rcutsq, num_typ_parameters);
    __syncthreads();

    unsigned int idx = blockIdx.x * (blockDim.x / tpp) + threadIdx.x / tpp;
    if (idx >= N)
        return;

    unsigned int n_neigh = d_n_neigh[idx];
    Scalar4 postypei = __ldg(d_pos + idx);
    Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);
    const unsigned int typei = __scalar_as_int(postypei.w);
    const size_t head_idx = d_head_list[idx];

    const float phi_i_total = d_phi_ab_arr[idx * (ntypes + 1) + ntypes];
    const float s_i_total = d_s_ab_arr[idx * (ntypes + 1) + ntypes];

    float fx = 0.0f, fy = 0.0f, fz = 0.0f, pe = 0.0f;
    float virialxx = 0.0f;
    float virialxy = 0.0f;
    float virialxz = 0.0f;
    float virialyy = 0.0f;
    float virialyz = 0.0f;
    float virialzz = 0.0f;

    unsigned int cur_j = 0;
    unsigned int next_j
        = threadIdx.x % tpp < n_neigh ? __ldg(d_nlist + head_idx + threadIdx.x % tpp) : 0;

    for (int neigh_idx = threadIdx.x % tpp; neigh_idx < n_neigh; neigh_idx += tpp)
        {
        cur_j = next_j;
        if (neigh_idx + tpp < n_neigh)
            {
            next_j = __ldg(d_nlist + head_idx + neigh_idx + tpp);
            }

        Scalar4 postypej = __ldg(d_pos + cur_j);
        Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

        // Subtraction and minimum image in double (as everywhere in HOOMD),
        // then narrow the separation once; everything downstream is float.
        Scalar3 dxd = box.minImage(posi - posj);
        const float dx = float(dxd.x), dy = float(dxd.y), dz = float(dxd.z);
        const float rij_sq = dx * dx + dy * dy + dz * dz;

        const unsigned int typej = __scalar_as_int(postypej.w);
        const unsigned int typpair = typpair_idx(typei, typej);
        if (rij_sq >= s_rcutsq[typpair])
            continue;

        evaluator evalij(rij_sq, s_params[typpair]);

        float rep_eng = 0.0f;
        float rep_eng_div = 0.0f;
        float attr_eng = 0.0f;
        float attr_eng_div = 0.0f;
        evalij.evalRepulsive(rep_eng, rep_eng_div);
        evalij.evalAttractive(attr_eng, attr_eng_div);

        float W1W2 = 0.0f;
        if (attr_eng != 0.0f || attr_eng_div != 0.0f)
            {
            float W1 = 0.0f;
            float W2 = 0.0f;
            evalij.evalOmega(W1, d_phi_ab_arr[idx * (ntypes + 1) + typej], phi_i_total);
            evalij.evalOmegaji(W2, d_phi_ab_arr[cur_j * (ntypes + 1) + typei],
                               d_phi_ab_arr[cur_j * (ntypes + 1) + ntypes]);
            W1W2 = W1 * W2;
            }

        // dE/dr_ij: the pair part plus the many-body chain-rule part, the
        // latter only inside the density window where w' is nonzero.
        float dedr = rep_eng_div + W1W2 * attr_eng_div;
        float Wrij_div = 0.0f;
        evalij.evalPhiDiv(Wrij_div);
        if (Wrij_div != 0.0f)
            {
            const float s_i
                = evalij.pickPhi(d_s_ab_arr[idx * (ntypes + 1) + typej], s_i_total);
            const float s_j
                = evalij.pickPhi(d_s_ab_arr[cur_j * (ntypes + 1) + typei],
                                 d_s_ab_arr[cur_j * (ntypes + 1) + ntypes]);
            dedr += Wrij_div * (s_i + s_j);
            }

        const float force_divr = -dedr * evalij.rinv();

        fx += dx * force_divr;
        fy += dy * force_divr;
        fz += dz * force_divr;
        pe += 0.5f * (rep_eng + W1W2 * attr_eng);

        // Half the pairwise virial per endpoint; every force term, the
        // many-body one included, is central (parallel to the separation), so
        // the textbook pairwise formula applies term by term.
        if (compute_virial)
            {
            virialxx += 0.5f * dx * dx * force_divr;
            virialxy += 0.5f * dx * dy * force_divr;
            virialxz += 0.5f * dx * dz * force_divr;
            virialyy += 0.5f * dy * dy * force_divr;
            virialyz += 0.5f * dy * dz * force_divr;
            virialzz += 0.5f * dz * dz * force_divr;
            }
        }

    // Combine the team's partials with register shuffles; one write, no atomics.
    fx = hoomd::detail::WarpReduce<float, tpp>().Sum(fx);
    fy = hoomd::detail::WarpReduce<float, tpp>().Sum(fy);
    fz = hoomd::detail::WarpReduce<float, tpp>().Sum(fz);
    pe = hoomd::detail::WarpReduce<float, tpp>().Sum(pe);

    if (compute_virial)
        {
        virialxx = hoomd::detail::WarpReduce<float, tpp>().Sum(virialxx);
        virialxy = hoomd::detail::WarpReduce<float, tpp>().Sum(virialxy);
        virialxz = hoomd::detail::WarpReduce<float, tpp>().Sum(virialxz);
        virialyy = hoomd::detail::WarpReduce<float, tpp>().Sum(virialyy);
        virialyz = hoomd::detail::WarpReduce<float, tpp>().Sum(virialyz);
        virialzz = hoomd::detail::WarpReduce<float, tpp>().Sum(virialzz);
        }

    // Widen back to Scalar at the single per-particle store.
    if ((threadIdx.x % tpp) == 0)
        {
        d_force[idx] = make_scalar4(Scalar(fx), Scalar(fy), Scalar(fz), Scalar(pe));
        if (compute_virial)
            {
            d_virial[0 * virial_pitch + idx] = Scalar(virialxx);
            d_virial[1 * virial_pitch + idx] = Scalar(virialxy);
            d_virial[2 * virial_pitch + idx] = Scalar(virialxz);
            d_virial[3 * virial_pitch + idx] = Scalar(virialyy);
            d_virial[4 * virial_pitch + idx] = Scalar(virialyz);
            d_virial[5 * virial_pitch + idx] = Scalar(virialzz);
            }
        }
    }

template<typename T>
void get_max_block_size(T func,
                        const valency_args_t& pair_args,
                        unsigned int& max_block_size,
                        unsigned int& kernel_shared_bytes)
    {
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)func);

    max_block_size = attr.maxThreadsPerBlock;
    max_block_size &= ~(pair_args.devprop.warpSize - 1);

    kernel_shared_bytes = (unsigned int)(attr.sharedSizeBytes);
    }

template<class evaluator, unsigned int compute_virial, int tpp> struct ValencyComputeKernel
    {

    static void launch(const valency_args_t& pair_args,
                       const typename evaluator::param_type* d_params)
        {
        if (tpp == pair_args.tpp)
            {
            unsigned int max_block_size = UINT_MAX;
            unsigned int kernel_shared_bytes = 0;
            unsigned int mbs, ksb;
            get_max_block_size(gpu_compute_density_kernel<evaluator, tpp>, pair_args, mbs, ksb);
            max_block_size = min(max_block_size, mbs);
            kernel_shared_bytes = max(kernel_shared_bytes, ksb);
            get_max_block_size(gpu_compute_valency_s_kernel<evaluator, tpp>, pair_args, mbs, ksb);
            max_block_size = min(max_block_size, mbs);
            kernel_shared_bytes = max(kernel_shared_bytes, ksb);
            get_max_block_size(gpu_compute_valency_forces_kernel<evaluator, compute_virial, tpp>,
                               pair_args,
                               mbs,
                               ksb);
            max_block_size = min(max_block_size, mbs);
            kernel_shared_bytes = max(kernel_shared_bytes, ksb);

            int run_block_size = min(pair_args.block_size, max_block_size);

            Index2D typpair_idx(pair_args.ntypes);
            const size_t base_shared
                = (sizeof(float) + sizeof(typename evaluator::param_type))
                  * typpair_idx.getNumElements();

            // The density and S kernels add the per-thread, per-type slab on
            // top of the staged parameters; the force kernel needs only the
            // parameters.
            size_t slab_shared = pair_args.ntypes * run_block_size * sizeof(float);
            while (base_shared + slab_shared + kernel_shared_bytes
                   >= pair_args.devprop.sharedMemPerBlock)
                {
                run_block_size -= pair_args.devprop.warpSize;
                if (run_block_size <= 0)
                    {
                    throw std::runtime_error(
                        "Valency potential parameters exceed the available shared "
                        "memory per block.");
                    }
                slab_shared = pair_args.ntypes * run_block_size * sizeof(float);
                }

            dim3 grid(pair_args.N / (run_block_size / pair_args.tpp) + 1, 1, 1);
            dim3 threads(run_block_size, 1, 1);

            hipLaunchKernelGGL((gpu_compute_density_kernel<evaluator, tpp>),
                               dim3(grid),
                               dim3(threads),
                               base_shared + slab_shared,
                               0,
                               pair_args.N,
                               pair_args.d_pos,
                               pair_args.d_phi_ab_arr,
                               pair_args.box,
                               pair_args.d_n_neigh,
                               pair_args.d_nlist,
                               pair_args.d_head_list,
                               d_params,
                               pair_args.d_rcutsq,
                               pair_args.ntypes);

            hipLaunchKernelGGL((gpu_compute_valency_s_kernel<evaluator, tpp>),
                               dim3(grid),
                               dim3(threads),
                               base_shared + slab_shared,
                               0,
                               pair_args.N,
                               pair_args.d_pos,
                               pair_args.d_phi_ab_arr,
                               pair_args.d_s_ab_arr,
                               pair_args.box,
                               pair_args.d_n_neigh,
                               pair_args.d_nlist,
                               pair_args.d_head_list,
                               d_params,
                               pair_args.d_rcutsq,
                               pair_args.ntypes);

            hipLaunchKernelGGL((gpu_compute_valency_forces_kernel<evaluator, compute_virial, tpp>),
                               dim3(grid),
                               dim3(threads),
                               base_shared,
                               0,
                               pair_args.d_force,
                               pair_args.N,
                               pair_args.d_virial,
                               pair_args.virial_pitch,
                               pair_args.d_pos,
                               pair_args.d_phi_ab_arr,
                               pair_args.d_s_ab_arr,
                               pair_args.box,
                               pair_args.d_n_neigh,
                               pair_args.d_nlist,
                               pair_args.d_head_list,
                               d_params,
                               pair_args.d_rcutsq,
                               pair_args.ntypes);

            }
        else
            {
            ValencyComputeKernel<evaluator, compute_virial, tpp / 2>::launch(pair_args, d_params);

            }
        }
    };

template<class evaluator, unsigned int compute_virial>
struct ValencyComputeKernel<evaluator, compute_virial, 0>
    {
    static void launch(const valency_args_t& pair_args,
                       const typename evaluator::param_type* d_params)
        {

        }
    };

template<class evaluator>
__attribute__((visibility("default"))) hipError_t
gpu_compute_valency_forces(const valency_args_t& pair_args,
                           const typename evaluator::param_type* d_params)
    {
    assert(d_params);
    assert(pair_args.d_rcutsq);
    assert(pair_args.ntypes > 0);

    if (!pair_args.compute_virial)
        {
        ValencyComputeKernel<evaluator, 0, gpu_valency_max_tpp>::launch(pair_args, d_params);
        }
    else
        {
        ValencyComputeKernel<evaluator, 1, gpu_valency_max_tpp>::launch(pair_args, d_params);
        }
    return hipSuccess;
    }
#else
template<class evaluator>
__attribute__((visibility("default"))) hipError_t
gpu_compute_valency_forces(const valency_args_t& pair_args,
                           const typename evaluator::param_type* d_params);
#endif

    }
    }
    }

#endif

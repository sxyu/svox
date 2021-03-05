#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <utility>
#include "data_spec_packed.cuh"

namespace {
namespace device {

template <typename scalar_t>
__device__ __inline__ void clamp_coord(scalar_t* __restrict__ q) {
    for (int i = 0; i < 3; ++i) {
        q[i] = max(scalar_t(0.0), min(scalar_t(1.0) - 1e-6, q[i]));
    }
}

template <typename scalar_t>
__device__ __inline__ void transform_coord(scalar_t* __restrict__ q,
                                           const scalar_t* __restrict__ offset,
                                           const scalar_t* __restrict__ scaling) {
    for (int i = 0; i < 3; ++i) {
        q[i] = offset[i] + scaling[i] * q[i];
    }
}

template <typename scalar_t>
__device__ __inline__ scalar_t* query_single_from_root(
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
       data,
    PackedTreeSpec<scalar_t>& __restrict__ tree,
    scalar_t* __restrict__ xyz_inout,
    scalar_t* __restrict__ cube_sz_out,
    int32_t* __restrict__ node_id_out,
    scalar_t* __restrict__ out_ptr=nullptr) {
    const scalar_t N = tree.child.size(1);
    clamp_coord<scalar_t>(xyz_inout);

    int32_t node_id = 0;
    int32_t u, v, w;
    *cube_sz_out = N;
    while (true) {
        xyz_inout[0] *= N;
        xyz_inout[1] *= N;
        xyz_inout[2] *= N;
        u = floor(xyz_inout[0]);
        v = floor(xyz_inout[1]);
        w = floor(xyz_inout[2]);
        xyz_inout[0] -= u;
        xyz_inout[1] -= v;
        xyz_inout[2] -= w;

        const int32_t skip = tree.child[node_id][u][v][w];
        if (skip == 0) {
            *node_id_out = node_id * N * N * N + u * N * N + v * N + w;
            if (tree.quant_colors.size(0)) {
                // This condition makes things slow
                // Decode from codebook
                const auto n_col = tree.quant_colors.size(0);
                for (int i = 0; i < n_col; ++i) {
                    uint16_t color_idx = tree.quant_color_map[node_id][u][v][w][i];
                    torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t>
                        color_ptr = tree.quant_colors[i][color_idx];
                    for (int j = 0; j < 3; ++j) {
                        out_ptr[j * n_col + i] = color_ptr[j];
                    }
                }
                out_ptr[tree.data_dim - 1] = data[node_id][u][v][w][0];
                return out_ptr;
            } else {
                scalar_t* leaf = &data[node_id][u][v][w][0];
                if (out_ptr != nullptr) {
                    // Copy
                    for (int i = 0; i < tree.data_dim; ++i) {
                        out_ptr[i] = leaf[i];
                    }
                    return out_ptr;
                }
                return leaf;
            }
        }
        *cube_sz_out *= N;
        node_id += skip;
    }
    return nullptr;
}

}  // namespace device
}  // namespace

#define CUDA_GET_THREAD_ID(tid, Q) const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
                      if (tid >= Q) return
#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)
#define CUDA_CHECK_ERRORS \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
            printf("Error in svox.%s : %s\n", __FUNCTION__, cudaGetErrorString(err))

namespace {
// Get approx number of CUDA cores
__host__ int get_sp_cores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            break;
        default:
            break;
    }
    return cores;
}
}  // namespace

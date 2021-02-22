#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits>
        child,
    scalar_t* __restrict__ xyz_inout,
    scalar_t* __restrict__ cube_sz_out) {
    const scalar_t N = child.size(1);
    clamp_coord<scalar_t>(xyz_inout);

    int node_id = 0;
    int u, v, w;
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

        const int32_t skip = child[node_id][u][v][w];
        if (skip == 0) {
            return &data[node_id][u][v][w][0];
        }
        *cube_sz_out *= N;
        node_id += skip;
    }
    return nullptr;
}

}  // namespace device
}  // namespace

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

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

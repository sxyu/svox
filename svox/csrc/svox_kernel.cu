#include <ATen/ATen.h>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.hpp"

namespace {
namespace device {

#define CUDA_GET_THREAD_ID(tid, Q) const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
                      if (tid >= Q) return
#define N_THREADS 1024
#define N_BLOCKS_NEEDED(Q) ((Q - 1) / N_THREADS + 1)
#define CUDA_CHECK_ERRORS \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
            printf("Error in svox.%s : %s\n", __FUNCTION__, cudaGetErrorString(err))

template <typename scalar_t>
__device__ __inline__ void clamp_coord(scalar_t* __restrict__ q) {
    for (int i = 0; i < 3; ++i) {
        q[i] = max(scalar_t(0.0),
               min(scalar_t(1.0) - std::numeric_limits<scalar_t>::epsilon(),
                   q[i]));
    }
}

template <typename scalar_t>
__device__ __inline__ bool outside_grid(const scalar_t* __restrict__ q) {
    for (int i = 0; i < 3; ++i) {
        if (q[i] < 0.0 || q[i] >= 1.0)
            return true;
    }
    return false;
}

template <typename scalar_t>
__device__ __inline__ void transform_coord(scalar_t* __restrict__ q,
                                           const scalar_t* __restrict__ offset,
                                           const scalar_t* __restrict__ invradius) {
    for (int i = 0; i < 3; ++i) {
        q[i] = offset[i] + invradius[0] * q[i];
    }
}

template <typename scalar_t, int padding_mode>
__global__ void query_single_vertical(const scalar_t* data,
                                      const int32_t* child,
                                      const scalar_t* __restrict__ indices,
                                      const scalar_t* __restrict__ offset,
                                      const scalar_t* __restrict__ invradius,
                                      scalar_t* __restrict__ result,
                                      const int N,
                                      const int Q, const int K) {
    CUDA_GET_THREAD_ID(tid, Q);
    const scalar_t* q_ = indices + 3 * tid;
    scalar_t* out = result + K * tid;

    scalar_t q[3] = {q_[0], q_[1], q_[2]};
    transform_coord<scalar_t>(q, offset, invradius);
    if (padding_mode == PADDING_MODE_BORDER) {
        clamp_coord<scalar_t>(q);
    } else {
        if (outside_grid<scalar_t>(q))
            return;
    }
    while (true) {
        // Find index of query point, in {0, ... N^3}
        int32_t index = 0;
        for (int i = 0; i < 3; ++i) {
            float idx_dimi = floorf(q[i] * N);
            index = index * N + (int32_t) idx_dimi;
            q[i] = q[i] * N - idx_dimi;
        }

        // Find child offset
        int32_t skip = child[index];

        // Add to output
        const scalar_t* val = data + index * K;
        for (int i = 0; i < K; ++i) out[i] += val[i];
        if (skip == 0) break;

        child += skip * N * N * N;
        data += skip * N * N * N * K;
    }
}

template <typename scalar_t, int padding_mode>
__global__ void query_single_vertical_backward(
                                      const int32_t* child,
                                      const scalar_t* __restrict__ indices,
                                      const scalar_t* __restrict__ grad_output,
                                      const scalar_t* __restrict__ offset,
                                      const scalar_t* __restrict__ invradius,
                                      scalar_t* grad_data,
                                      const int N,
                                      const int Q, const int K) {
    CUDA_GET_THREAD_ID(tid, Q);
    const scalar_t* q_ = indices + 3 * tid;
    const scalar_t* grad_out_q = grad_output + K * tid;

    scalar_t q[3] = {q_[0], q_[1], q_[2]};
    transform_coord<scalar_t>(q, offset, invradius);
    if (padding_mode == PADDING_MODE_BORDER) {
        clamp_coord<scalar_t>(q);
    } else {
        if (outside_grid<scalar_t>(q))
            return;
    }
    while (true) {
        // Find index of query point, in {0, ... N^3}
        int32_t index = 0;
        for (int i = 0; i < 3; ++i) {
            float idx_dimi = floorf(q[i] * N);
            index = index * N + (int32_t) idx_dimi;
            q[i] = q[i] * N - idx_dimi;
        }

        // Find child offset
        int32_t skip = child[index];

        // Add to output
        scalar_t* grad_data_q = grad_data + index * K;
        for (int i = 0; i < K; ++i)
            atomicAdd(grad_data_q + i, grad_out_q[i]);
        if (skip == 0) break;

        child += skip * N * N * N;
        grad_data += skip * N * N * N * K;
    }
}

template <typename scalar_t, int K, int padding_mode>
__global__ void assign_single_vertical(const int32_t* child,
                                      const scalar_t* __restrict__ indices,
                                      const scalar_t* __restrict__ values,
                                      const scalar_t* __restrict__ offset,
                                      const scalar_t* __restrict__ invradius,
                                      scalar_t* data,
                                      const int N,
                                      const int Q) {
    CUDA_GET_THREAD_ID(tid, Q);
    const scalar_t* q_ = indices + 3 * tid;
    const scalar_t* tgt = values + K * tid;

    scalar_t q[3] = {q_[0], q_[1], q_[2]};
    transform_coord<scalar_t>(q, offset, invradius);
    if (padding_mode == PADDING_MODE_BORDER) {
        clamp_coord<scalar_t>(q);
    } else {
        if (outside_grid<scalar_t>(q))
            return;
    }

    scalar_t tmp[K];
    for (int i = 0; i < K; ++i) {
        tmp[i] = (scalar_t) 0.0;
    }
    while (true) {
        // Find index of query point, in {0, ... N^3}
        int32_t index = 0;
        for (int i = 0; i < 3; ++i) {
            float idx_dimi = floorf(q[i] * N);
            index = index * N + (int32_t) idx_dimi;
            q[i] = q[i] * N - idx_dimi;
        }

        // Find child offset
        int32_t skip = child[index];

        // Add to output
        scalar_t* val = data + index * K;
        if (skip == 0) {
            for (int i = 0; i < K; ++i) val[i] = tgt[i] - tmp[i];
            break;
        } else {
            for (int i = 0; i < K; ++i) tmp[i] += val[i];
        }

        child += skip * N * N * N;
        data += skip * N * N * N * K;
    }
}

// template <typename scalar_t>
// __device__ void render_single_ray_sub(scalar_t* __restrict__ data,
//                                       const int32_t* __restrict__ child,
//                                       const scalar_t* __restrict__ ray,
//                                       scalar_t* __restrict__ out,
//                                       int N, int Q, int K) {
// }
//
//
// template <typename scalar_t>
// __global__ void render_single_ray(scalar_t* __restrict__ data,
//                                   const int32_t* __restrict__ child,
//                                   const scalar_t* __restrict__ rays,
//                                   const scalar_t* __restrict__ offset,
//                                   const scalar_t* __restrict__ invradius,
//                                   float scale,
//                                   scalar_t* __restrict__ result,
//                                   bool white_bkgd,
//                                   int N, int Q, int K) {
//     CUDA_GET_THREAD_ID(tid, Q);
//     scalar_t ray[8];
//     for (int i = 0; i < 8; ++i) {
//         ray[i] = ray[tid * 8 + i];
//     }
//     transform_coord<scalar_t>(ray, offset, invradius);
//
//     scalar_t* __restrict__ out = result + tid * K;
//     render_single_ray_sub<scalar_t>(data, child, ray, out, N, Q, K);
//
//     scalar_t radius = 0.5 / invradius[0];
// }

}  // namespace device
}  // namespace

at::Tensor _query_vertical_cuda(
        at::Tensor data, at::Tensor child,
        at::Tensor indices,
        at::Tensor offset, at::Tensor invradius,
        int padding_mode) {
    const auto Q = indices.size(0), N = data.size(1), K = data.size(-1);

    const int blocks = N_BLOCKS_NEEDED(Q);

    at::Tensor result = at::zeros({Q, K}, indices.options());

    AT_DISPATCH_FLOATING_TYPES(indices.type(), "_query_cuda_vertical", [&] {
#define QUERY_SINGLE_CALL(PADDING_MODE) \
        device::query_single_vertical<scalar_t, PADDING_MODE><<<blocks, N_THREADS>>>( \
                data.data<scalar_t>(), \
                child.data<int32_t>(), \
                indices.data<scalar_t>(), \
                offset.data<scalar_t>(), \
                invradius.data<scalar_t>(), \
                result.data<scalar_t>(), \
                N, Q, K)
        if (padding_mode == 0) {
            QUERY_SINGLE_CALL(0);
        } else {
            QUERY_SINGLE_CALL(1);
        }
    });
#undef QUERY_SINGLE_CALL
    CUDA_CHECK_ERRORS;
    return result;
}

void _assign_vertical_cuda(
        at::Tensor data, at::Tensor child,
        at::Tensor indices, at::Tensor values,
        at::Tensor offset, at::Tensor invradius,
        int padding_mode) {
    const auto Q = indices.size(0), N = data.size(1), K = data.size(-1);

    const int blocks = N_BLOCKS_NEEDED(Q);

    AT_DISPATCH_FLOATING_TYPES(indices.type(), "_query_cuda_vertical", [&] {
        switch(K) {
#define ASSIGN_SINGLE_CALL(K_VAL, PADDING_MODE) \
        device::assign_single_vertical<scalar_t, K_VAL, PADDING_MODE><<<blocks, N_THREADS>>>( \
                child.data<int32_t>(), \
                indices.data<scalar_t>(), \
                values.data<scalar_t>(), \
                offset.data<scalar_t>(), \
                invradius.data<scalar_t>(), \
                data.data<scalar_t>(), \
                N, Q)
#define ASSIGN_SINGLE_CASE(K_VAL) \
        case K_VAL: \
                if (padding_mode == 0) { \
                    ASSIGN_SINGLE_CALL(K_VAL, 0); \
                } else { \
                    ASSIGN_SINGLE_CALL(K_VAL, 1); \
                } \
                break;
            ASSIGN_SINGLE_CASE(1);
            ASSIGN_SINGLE_CASE(2);
            ASSIGN_SINGLE_CASE(3);
            ASSIGN_SINGLE_CASE(4);
            ASSIGN_SINGLE_CASE(5);
            ASSIGN_SINGLE_CASE(6);
            ASSIGN_SINGLE_CASE(7);
            ASSIGN_SINGLE_CASE(8);
            ASSIGN_SINGLE_CASE(9);
            ASSIGN_SINGLE_CASE(10);
        };
    });
#undef ASSIGN_SINGLE_CALL
#undef ASSIGN_SINGLE_CASE
    CUDA_CHECK_ERRORS;
}

/**
 * @param data (M, N, N, N, K)
 * @param child (M, N, N, N)
 * @param indices (Q, 3)
 * @param grad_output (Q, K)
 * @return (M, N, N, N, K)
 * */
at::Tensor _query_vertical_backward_cuda(
        at::Tensor child,
        at::Tensor indices,
        at::Tensor grad_output,
        at::Tensor offset,
        at::Tensor invradius,
        int padding_mode) {
    const auto Q = indices.size(0), N = child.size(1),
               K = grad_output.size(-1), M = child.size(0);
    const int blocks = N_BLOCKS_NEEDED(Q);

    at::Tensor grad_data = at::zeros({M, N, N, N, K}, grad_output.options());

    AT_DISPATCH_FLOATING_TYPES(indices.type(), "_query_cuda_vertical_backward", [&] {
#define QUERY_BACKWARD_SINGLE_CALL(PADDING_MODE) \
        device::query_single_vertical_backward<scalar_t, PADDING_MODE><<<blocks, N_THREADS>>>( \
                child.data<int32_t>(), \
                indices.data<scalar_t>(), \
                grad_output.data<scalar_t>(), \
                offset.data<scalar_t>(), \
                invradius.data<scalar_t>(), \
                grad_data.data<scalar_t>(), \
                N, Q, K)
        if (padding_mode == 0) {
            QUERY_BACKWARD_SINGLE_CALL(0);
        } else {
            QUERY_BACKWARD_SINGLE_CALL(1);
        }
    });
#undef QUERY_BACKWARD_SINGLE_CALL

    CUDA_CHECK_ERRORS;
    return grad_data;
}

// at::Tensor _render_cuda(
//         at::Tensor data, at::Tensor child,
//         at::Tensor rays, at::Tensor offset, at::Tensor invradius, bool white_bkgd) {
//     const auto Q = rays.size(0), N = data.size(1), K = data.size(-1);
//     const int blocks = N_BLOCKS_NEEDED(Q);
//
//     at::Tensor result = at::zeros({Q, K}, rays.options());
//
//     // rays.packed_accessor32<float, 2>();
//     AT_DISPATCH_FLOATING_TYPES(rays.type(), "_query_cuda_vertical_backward", [&] {
//         device::render_single_ray<scalar_t><<<blocks, N_THREADS>>>(
//                 data.data<scalar_t>(),
//                 child.data<int32_t>(),
//                 rays.data<scalar_t>(),
//                 offset.data<scalar_t>(),
//                 invradius.data<scalar_t>(),
//                 result.data<scalar_t>(),
//                 white_bkgd,
//                 N, Q, K);
//     });
//     return result;
// }

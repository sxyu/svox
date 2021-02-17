/*
 * Copyright Alex Yu 2021
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdint>
#include "common.cuh"

namespace {
namespace device {

template <typename scalar_t>
__device__ __inline__ void transform_coord(scalar_t* __restrict__ q,
                                           const scalar_t* __restrict__ offset,
                                           const scalar_t* __restrict__ invradius) {
    for (int i = 0; i < 3; ++i) {
        q[i] = offset[i] + invradius[0] * q[i];
    }
}

template <typename scalar_t>
__global__ void query_single_kernel(
       const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> data,
       const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const scalar_t* __restrict__ offset,
       const scalar_t* __restrict__ invradius,
       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> result) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    scalar_t xyz[3] = {indices[tid][0], indices[tid][1], indices[tid][2]};
    transform_coord<scalar_t>(xyz, offset, invradius);
    scalar_t _cube_sz;
    query_single_from_root<scalar_t>(data, child,
            xyz, &result[tid][0], &_cube_sz);
}

template <typename scalar_t>
__global__ void query_single_kernel_backward(
       const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_output,
       const scalar_t* __restrict__ offset,
       const scalar_t* __restrict__ invradius,
       torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> grad_data) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    scalar_t q[3] = {indices[tid][0], indices[tid][1], indices[tid][2]};
    transform_coord<scalar_t>(q, offset, invradius);
    clamp_coord<scalar_t>(q);
    const scalar_t N = child.size(1);
    const int data_dim = grad_output.size(1);

    int node_id = 0;
    int u, v, w;
    while (true) {
        q[0] *= N; q[1] *= N; q[2] *= N;
        u = q[0]; v = q[1]; w = q[2];
        q[0] -= u; q[1] -= v; q[2] -= w;

        const int32_t skip = child[node_id][u][v][w];
        if (skip == 0) {
            for (int i = 0; i < data_dim; ++i)
                atomicAdd(&grad_data[node_id][u][v][w][i], grad_output[tid][i]);
            break;
        }
        node_id += skip;
    }
}

template <typename scalar_t>
__global__ void assign_single_kernel(
       torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> data,
       const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
       const scalar_t* __restrict__ offset,
       const scalar_t* __restrict__ invradius) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));

    scalar_t q[3] = {indices[tid][0], indices[tid][1], indices[tid][2]};
    transform_coord<scalar_t>(q, offset, invradius);
    clamp_coord<scalar_t>(q);
    const scalar_t N = child.size(1);
    const int data_dim = data.size(4);

    int node_id = 0;
    int u, v, w;
    while (true) {
        q[0] *= N; q[1] *= N; q[2] *= N;
        u = q[0]; v = q[1]; w = q[2];
        q[0] -= u; q[1] -= v; q[2] -= w;

        const int32_t skip = child[node_id][u][v][w];
        if (skip == 0) {
            for (int i = 0; i < data_dim; ++i)
                data[node_id][u][v][w][i] = values[tid][i];
            break;
        }
        node_id += skip;
    }
}

}  // namespace device
}  // namespace

torch::Tensor _query_vertical_cuda(
        torch::Tensor data, torch::Tensor child,
        torch::Tensor indices,
        torch::Tensor offset, torch::Tensor invradius) {
    const auto Q = indices.size(0), K = data.size(4);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q);
    torch::Tensor result = torch::zeros({Q, K}, indices.options());
    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::query_single_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                data.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                offset.data<scalar_t>(),
                invradius.data<scalar_t>(),
                result.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

void _assign_vertical_cuda(
        torch::Tensor data, torch::Tensor child,
        torch::Tensor indices, torch::Tensor values,
        torch::Tensor offset, torch::Tensor invradius) {
    const int blocks = CUDA_N_BLOCKS_NEEDED(indices.size(0));
    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::assign_single_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                data.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                offset.data<scalar_t>(),
                invradius.data<scalar_t>());
    });
    CUDA_CHECK_ERRORS;
}

torch::Tensor _query_vertical_backward_cuda(
        torch::Tensor child,
        torch::Tensor indices,
        torch::Tensor grad_output,
        torch::Tensor offset,
        torch::Tensor invradius) {
    const auto Q = indices.size(0), N = child.size(1),
               K = grad_output.size(1), M = child.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q);

    torch::Tensor grad_data = torch::zeros({M, N, N, N, K}, grad_output.options());

    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::query_single_kernel_backward<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                offset.data<scalar_t>(),
                invradius.data<scalar_t>(),
                grad_data.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>());
    });

    CUDA_CHECK_ERRORS;
    return grad_data;
}

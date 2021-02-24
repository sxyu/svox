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

#define CUDA_N_THREADS 1024

namespace {
namespace device {

template <typename scalar_t, typename data_storage_t>
__device__ __inline__ data_storage_t* get_tree_leaf_ptr(
       torch::PackedTensorAccessor32<data_storage_t, 5, torch::RestrictPtrTraits> data,
       const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
       const scalar_t* __restrict__ xyz_ind,
       const scalar_t* __restrict__ offset,
       const scalar_t* __restrict__ scaling,
       int32_t* node_id) {
    scalar_t xyz[3] = {xyz_ind[0], xyz_ind[1], xyz_ind[2]};
    transform_coord<scalar_t>(xyz, offset, scaling);
    scalar_t _cube_sz;
    return query_single_from_root<scalar_t, data_storage_t>(data, child,
            xyz, &_cube_sz, node_id);
}

template <typename scalar_t>
__global__ void query_single_kernel(
       torch::PackedTensorAccessor32<torch::Half, 5, torch::RestrictPtrTraits> data,
       const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const scalar_t* __restrict__ offset,
       const scalar_t* __restrict__ scaling,
       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> result,
       torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> node_ids) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    torch::Half* data_ptr = get_tree_leaf_ptr(data, child, &indices[tid][0], offset, scaling,
            &node_ids[tid]);
    for (int i = 0; i < data.size(4); ++i)
        result[tid][i] = __half2float(data_ptr[i]);
}

template <typename scalar_t>
__global__ void query_single_kernel_backward(
       const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_output,
       const scalar_t* __restrict__ offset,
       const scalar_t* __restrict__ scaling,
       torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> grad_data) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    int32_t _node_id;
    scalar_t* data_ptr = get_tree_leaf_ptr(grad_data,
            child, &indices[tid][0], offset, scaling, &_node_id);
    for (int i = 0; i < grad_output.size(1); ++i)
        atomicAdd(&data_ptr[i], grad_output[tid][i]);
}

template <typename scalar_t>
__global__ void assign_single_kernel(
       torch::PackedTensorAccessor32<torch::Half, 5, torch::RestrictPtrTraits> data,
       const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits> child,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values,
       const scalar_t* __restrict__ offset,
       const scalar_t* __restrict__ scaling) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    int32_t _node_id;
    torch::Half* data_ptr = get_tree_leaf_ptr(data, child, &indices[tid][0], offset, scaling,
            &_node_id);
    for (int i = 0; i < values.size(1); ++i)
        data_ptr[i] = __float2half(values[tid][i]);
}

}  // namespace device
}  // namespace

std::tuple<torch::Tensor, torch::Tensor>
    _query_vertical_cuda(
        torch::Tensor data, torch::Tensor child,
        torch::Tensor indices,
        torch::Tensor offset, torch::Tensor scaling) {
    const auto Q = indices.size(0), K = data.size(4);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);
    torch::Tensor result = torch::empty({Q, K}, indices.options());
    torch::Tensor node_ids = torch::empty({Q}, child.options());
    AT_DISPATCH_FLOATING_TYPES(indices.type(), query_vertical, [&] {
        device::query_single_kernel<float><<<blocks, CUDA_N_THREADS>>>(
                data.packed_accessor32<torch::Half, 5, torch::RestrictPtrTraits>(),
                child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                offset.data<float>(),
                scaling.data<float>(),
                result.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                node_ids.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return std::make_tuple(result, node_ids);
}

void _assign_vertical_cuda(
        torch::Tensor data, torch::Tensor child,
        torch::Tensor indices, torch::Tensor values,
        torch::Tensor offset, torch::Tensor scaling) {
    const int blocks = CUDA_N_BLOCKS_NEEDED(indices.size(0), CUDA_N_THREADS);
    AT_DISPATCH_FLOATING_TYPES(indices.type(), assign_vertical, [&] {
        device::assign_single_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                data.packed_accessor32<torch::Half, 5, torch::RestrictPtrTraits>(),
                child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                offset.data<scalar_t>(),
                scaling.data<scalar_t>());
    });
    CUDA_CHECK_ERRORS;
}

torch::Tensor _query_vertical_backward_cuda(
        torch::Tensor child,
        torch::Tensor indices,
        torch::Tensor grad_output,
        torch::Tensor offset,
        torch::Tensor scaling) {
    const auto Q = indices.size(0), N = child.size(1),
               K = grad_output.size(1), M = child.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);

    torch::Tensor grad_data = torch::zeros({M, N, N, N, K}, grad_output.options());

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), query_vertical_backward, [&] {
        device::query_single_kernel_backward<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                offset.data<scalar_t>(),
                scaling.data<scalar_t>(),
                grad_data.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>());
    });

    CUDA_CHECK_ERRORS;
    return grad_data;
}

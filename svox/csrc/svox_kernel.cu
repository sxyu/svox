/*
 * Copyright 2021 PlenOctree Authors
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
#include "data_spec_packed.cuh"

#define CUDA_N_THREADS 1024

namespace {
void check_indices(torch::Tensor& indices) {
    CHECK_INPUT(indices);
    TORCH_CHECK(indices.dim() == 2);
    TORCH_CHECK(indices.is_floating_point());
}

namespace device {

template <typename scalar_t>
__device__ __inline__ scalar_t* get_tree_leaf_ptr(
       torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits>
        data,
       PackedTreeSpec<scalar_t>& __restrict__ tree,
       const scalar_t* __restrict__ xyz_ind,
       int64_t* node_id=nullptr) {
    scalar_t xyz[3] = {xyz_ind[0], xyz_ind[1], xyz_ind[2]};
    transform_coord<scalar_t>(xyz, tree.offset, tree.scaling);
    scalar_t _cube_sz;
    return query_single_from_root<scalar_t>(data, tree.child,
            xyz, &_cube_sz, node_id);
}

template <typename scalar_t>
__global__ void query_single_kernel(
        PackedTreeSpec<scalar_t> tree,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> values_out,
        torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> node_ids_out) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    scalar_t* data_ptr = get_tree_leaf_ptr(tree.data, tree, &indices[tid][0], &node_ids_out[tid]);
    for (int i = 0; i < tree.data.size(4); ++i)
        values_out[tid][i] = data_ptr[i];
}

template <typename scalar_t>
__global__ void query_single_kernel_backward(
       PackedTreeSpec<scalar_t> tree,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_output,
       torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_data_out) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    scalar_t* data_ptr = get_tree_leaf_ptr(grad_data_out, tree, &indices[tid][0]);
    for (int i = 0; i < grad_output.size(1); ++i)
        atomicAdd(&data_ptr[i], grad_output[tid][i]);
}

template <typename scalar_t>
__global__ void assign_single_kernel(
       PackedTreeSpec<scalar_t> tree,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> values) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    scalar_t* data_ptr = get_tree_leaf_ptr(tree.data, tree, &indices[tid][0]);
    for (int i = 0; i < values.size(1); ++i)
        data_ptr[i] = values[tid][i];
}

template <typename scalar_t>
__global__ void calc_corner_kernel(
       PackedTreeSpec<scalar_t> tree,
       const torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> indexer,
       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output) {
    CUDA_GET_THREAD_ID(tid, indexer.size(0));
    const int N = tree.data.size(1);
    const auto* leaf = &indexer[tid][0];
    scalar_t* result = &output[tid][0];

    int32_t curr[4] = {(int32_t) leaf[0], (int32_t) leaf[1],
                       (int32_t) leaf[2], (int32_t) leaf[3]};
    while (true) {
        for (int i = 0; i < 3; ++i) {
            result[i] += curr[i + 1];
            result[i] /= N;
        }
        if (curr[0] == 0) break;
        curr[0] = tree.parent_depth[curr[0]][0];
        for (int i = 3; i > 0; --i) {
            curr[i] = curr[0] % N;
            curr[0] /= N;
        }
    }
}

}  // namespace device
}  // namespace

QueryResult query_vertical(TreeSpec& tree, torch::Tensor indices) {
    tree.check();
    check_indices(indices);
    DEVICE_GUARD(indices);

    const auto Q = indices.size(0), K = tree.data.size(4);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);
    torch::Tensor values = torch::empty({Q, K}, indices.options());
    auto node_ids_options = at::TensorOptions()
                       .dtype(at::kLong)
                       .layout(tree.child.layout())
                       .device(tree.child.device());
    torch::Tensor node_ids = torch::empty({Q}, node_ids_options);
    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::query_single_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                tree,
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                values.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                node_ids.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return QueryResult(values, node_ids);
}

void assign_vertical(TreeSpec& tree, torch::Tensor indices, torch::Tensor values) {
    tree.check();
    check_indices(indices);
    check_indices(values);
    DEVICE_GUARD(indices);
    const int blocks = CUDA_N_BLOCKS_NEEDED(indices.size(0), CUDA_N_THREADS);
    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::assign_single_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                tree,
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                values.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
}

torch::Tensor query_vertical_backward(
        TreeSpec& tree,
        torch::Tensor indices,
        torch::Tensor grad_output) {
    tree.check();
    DEVICE_GUARD(indices);
    const auto Q = indices.size(0), N = tree.child.size(1),
               K = grad_output.size(1), M = tree.child.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);

    torch::Tensor grad_data = torch::zeros({M, N, N, N, K}, grad_output.options());

    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::query_single_kernel_backward<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                tree,
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_output.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_data.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>());
    });

    CUDA_CHECK_ERRORS;
    return grad_data;
}

torch::Tensor calc_corners(
        TreeSpec& tree,
        torch::Tensor indexer) {
    tree.check();
    DEVICE_GUARD(indexer);
    const auto Q = indexer.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);

    torch::Tensor output = torch::zeros({Q, 3}, tree.data.options());

    AT_DISPATCH_FLOATING_TYPES(tree.data.type(), __FUNCTION__, [&] {
        device::calc_corner_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                tree,
                indexer.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });

    CUDA_CHECK_ERRORS;
    return output;
}

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
#include <utility>
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
__device__ __inline__ TreeLeaf<scalar_t> get_tree_leaf_ptr(
       torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
        data,
       PackedTreeSpec<scalar_t>& __restrict__ tree,
       const scalar_t* __restrict__ xyz_ind,
       int32_t* node_id_out) {
    scalar_t xyz[3] = {xyz_ind[0], xyz_ind[1], xyz_ind[2]};
    transform_coord<scalar_t>(xyz, tree.offset, tree.scaling);
    scalar_t _cube_sz;
    return query_single_from_root<scalar_t>(data, tree,
            xyz, &_cube_sz, node_id_out);
}

template <typename scalar_t>
__global__ void query_single_kernel(
        PackedTreeSpec<scalar_t> tree,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values_out,
        torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> node_ids_out) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    TreeLeaf<scalar_t> leaf = get_tree_leaf_ptr<scalar_t>(
            tree.data, tree, &indices[tid][0], &node_ids_out[tid]);
    for (int i = 0; i < tree.data_dim - 1; ++i)
        values_out[tid][i] = leaf.rgb[i];
    values_out[tid][tree.data_dim - 1] = *leaf.sigma;
}

template <typename scalar_t>
__global__ void query_single_kernel_backward(
       PackedTreeSpec<scalar_t> tree,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_output,
       torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> grad_data_out) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    // FIXME derivatives for quantized
    int32_t _node_id;
    TreeLeaf<scalar_t> leaf = get_tree_leaf_ptr(
            grad_data_out, tree, &indices[tid][0], &_node_id);
    for (int i = 0; i < tree.data_dim - 1; ++i)
        atomicAdd(&leaf.rgb[i], grad_output[tid][i]);
    atomicAdd(leaf.sigma, grad_output[tid][tree.data_dim - 1]);
}

template <typename scalar_t>
__global__ void assign_single_kernel(
       PackedTreeSpec<scalar_t> tree,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> indices,
       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> values) {
    CUDA_GET_THREAD_ID(tid, indices.size(0));
    int32_t _node_id;
    TreeLeaf<scalar_t> leaf = get_tree_leaf_ptr(
            tree.data, tree, &indices[tid][0], &_node_id);
    for (int i = 0; i < tree.data_dim - 1; ++i)
        leaf.rgb[i] = values[tid][i];
    *leaf.sigma = values[tid][tree.data_dim - 1];
}

}  // namespace device
}  // namespace

QueryResult query_vertical(TreeSpec& tree, torch::Tensor indices) {
    tree.check();
    check_indices(indices);
    DEVICE_GUARD(indices);

    const int Q = indices.size(0);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);
    torch::Tensor values = torch::empty({Q, tree.data_dim}, indices.options());
    torch::Tensor node_ids = torch::empty({Q}, tree.child.options());
    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::query_single_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                tree,
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                node_ids.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());
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
                values.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
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
               M = tree.child.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS);

    // FIXME gradient with quantized colors (library/codebook)
    torch::Tensor grad_data = torch::zeros({M, N, N, N, tree.data_dim},
            grad_output.options());

    AT_DISPATCH_FLOATING_TYPES(indices.type(), __FUNCTION__, [&] {
        device::query_single_kernel_backward<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                tree,
                indices.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_data.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>());
    });

    CUDA_CHECK_ERRORS;
    return grad_data;
}

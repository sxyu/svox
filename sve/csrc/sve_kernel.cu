#include <ATen/ATen.h>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {
namespace device {

#define FOR_CUDA(tid) const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
                      if (tid >= Q) return

template <typename scalar_t>
__device__ __inline__ void clamp_coord(scalar_t* __restrict__ q) {
    for (int i = 0; i < 3; ++i) {
        q[i] = max(scalar_t(0.0),
               min(scalar_t(1.0) - std::numeric_limits<scalar_t>::epsilon(),
                   q[i]));
    }
}

template <typename scalar_t>
__global__ void query_single_vertical(const scalar_t* data,
                                      const int32_t* child,
                                      const scalar_t* __restrict__ indices,
                                      scalar_t* __restrict__ result,
                                      const int N,
                                      const int Q, const int K,
                                      bool vary_non_leaf
                                    ) {
    FOR_CUDA(tid);
    const scalar_t* q_ = indices + 3 * tid;
    scalar_t* out = result + K * tid;

    scalar_t q[3] = {q_[0], q_[1], q_[2]};
    clamp_coord<scalar_t>(q);
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
        if (skip == 0 || vary_non_leaf) {
            const scalar_t* val = data + index * K;
            for (int i = 0; i < K; ++i) out[i] += val[i];
            if (skip == 0) break;
        }

        child += skip * N * N * N;
        data += skip * N * N * N * K;
    }
}

template <typename scalar_t>
__global__ void query_single_vertical_backward(
                                      const int32_t* child,
                                      const scalar_t* __restrict__ indices,
                                      const scalar_t* __restrict__ grad_output,
                                      scalar_t* grad_data,
                                      const int N,
                                      const int Q, const int K,
                                      bool vary_non_leaf
                                    ) {
    FOR_CUDA(tid);

    const scalar_t* q_ = indices + 3 * tid;
    const scalar_t* grad_out_q = grad_output + K * tid;

    scalar_t q[3] = {q_[0], q_[1], q_[2]};
    clamp_coord<scalar_t>(q);
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
        if (skip == 0 || vary_non_leaf) {
            scalar_t* grad_data_q = grad_data + index * K;
            for (int i = 0; i < K; ++i)
                atomicAdd(grad_data_q + i, grad_out_q[i]);
            if (skip == 0) break;
        }

        child += skip * N * N * N;
        grad_data += skip * N * N * N * K;
    }
}

template <typename scalar_t, int K>
__global__ void assign_single_vertical(scalar_t* data,
                                      const int32_t* child,
                                      const scalar_t* __restrict__ indices,
                                      const scalar_t* __restrict__ values,
                                      const int N,
                                      const int Q,
                                      bool vary_non_leaf
                                    ) {
    FOR_CUDA(tid);
    const scalar_t* q_ = indices + 3 * tid;
    const scalar_t* tgt = values + K * tid;

    scalar_t tmp[K];
    for (int i = 0; i < K; ++i) {
        tmp[i] = (scalar_t) 0.0;
    }
    scalar_t q[3] = {q_[0], q_[1], q_[2]};
    clamp_coord<scalar_t>(q);
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
        } else if (vary_non_leaf) {
            for (int i = 0; i < K; ++i) tmp[i] += val[i];
        }

        child += skip * N * N * N;
        data += skip * N * N * N * K;
    }
}

}  // namespace device
const int N_THREADS = 1024;
}  // namespace

at::Tensor _query_vertical_cuda(
        at::Tensor data, at::Tensor child,
        at::Tensor indices, bool vary_non_leaf) {
    const auto Q = indices.size(0), N = data.size(1), K = data.size(-1);

    const int blocks = (Q - 1) / N_THREADS + 1;

    at::Tensor result = at::zeros({Q, K}, indices.options());

    AT_DISPATCH_FLOATING_TYPES(indices.type(), "_query_cuda_vertical", [&] {
        device::query_single_vertical<scalar_t><<<blocks, N_THREADS>>>(
                data.data<scalar_t>(),
                child.data<int32_t>(),
                indices.data<scalar_t>(),
                result.data<scalar_t>(),
                N,
                Q,
                K,
                vary_non_leaf);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in sve._query_vertical_cuda: %s\n", cudaGetErrorString(err));
    return result;
}

void _assign_vertical_cuda(
        at::Tensor data, at::Tensor child,
        at::Tensor indices, at::Tensor values, bool vary_non_leaf) {
    const auto Q = indices.size(0), N = data.size(1), K = data.size(-1);

    const int blocks = (Q - 1) / N_THREADS + 1;

    AT_DISPATCH_FLOATING_TYPES(indices.type(), "_query_cuda_vertical", [&] {
        switch(K) {
#define ASSIGN_SINGLE_VERTICAL_CASE(K_VAL) \
        case K_VAL: \
        device::assign_single_vertical<scalar_t, K_VAL><<<blocks, N_THREADS>>>( \
                data.data<scalar_t>(), \
                child.data<int32_t>(), \
                indices.data<scalar_t>(), \
                values.data<scalar_t>(), \
                N, \
                Q, \
                vary_non_leaf); \
                break;
            ASSIGN_SINGLE_VERTICAL_CASE(1);
            ASSIGN_SINGLE_VERTICAL_CASE(2);
            ASSIGN_SINGLE_VERTICAL_CASE(3);
            ASSIGN_SINGLE_VERTICAL_CASE(4);
            ASSIGN_SINGLE_VERTICAL_CASE(5);
            ASSIGN_SINGLE_VERTICAL_CASE(6);
            ASSIGN_SINGLE_VERTICAL_CASE(7);
            ASSIGN_SINGLE_VERTICAL_CASE(8);
            ASSIGN_SINGLE_VERTICAL_CASE(9);
            ASSIGN_SINGLE_VERTICAL_CASE(10);
        };
    });
#undef ASSIGN_SINGLE_VERTICAL_CASE

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in sve._assign_vertical_cuda: %s\n", cudaGetErrorString(err));
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
        bool vary_non_leaf) {
    const auto Q = indices.size(0), N = child.size(1),
               K = grad_output.size(-1), M = child.size(0);

    const int blocks = (Q - 1) / N_THREADS + 1;

    at::Tensor grad_data = at::zeros({M, N, N, N, K}, grad_output.options());

    AT_DISPATCH_FLOATING_TYPES(indices.type(), "_query_cuda_vertical_backward", [&] {
        device::query_single_vertical_backward<scalar_t><<<blocks, N_THREADS>>>(
                child.data<int32_t>(),
                indices.data<scalar_t>(),
                grad_output.data<scalar_t>(),
                grad_data.data<scalar_t>(),
                N,
                Q,
                K,
                vary_non_leaf);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in sve._query_vertical_backward_cuda: %s\n", cudaGetErrorString(err));
    return grad_data;
}

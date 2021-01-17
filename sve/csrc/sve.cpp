#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <vector>

namespace py = pybind11;

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

/** CUDA Interface **/
at::Tensor _query_vertical_cuda(at::Tensor data, at::Tensor child,
                                at::Tensor indices, bool vary_non_leaf);
at::Tensor _query_vertical_backward_cuda(at::Tensor child, at::Tensor indices,
                                         at::Tensor grad_output,
                                         bool vary_non_leaf);
void _assign_vertical_cuda(at::Tensor data, at::Tensor child,
                           at::Tensor indices, at::Tensor values,
                           bool vary_non_leaf);
/** END CUDA Interface **/

/**
 * @param data (M, N, N, N, K)
 * @param child (M, N, N, N)
 * @param indices (Q, 3)
 * @return (Q, K)
 * */
at::Tensor query_vertical(at::Tensor data, at::Tensor child, at::Tensor indices,
                          bool vary_non_leaf) {
    CHECK_INPUT(data);
    CHECK_INPUT(child);
    CHECK_INPUT(indices);
    TORCH_CHECK(indices.is_floating_point());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
    return _query_vertical_cuda(data, child, indices, vary_non_leaf);
}

/**
 * @param data (M, N, N, N, K)
 * @param child (M, N, N, N)
 * @param indices (Q, 3)
 * @param grad_output (Q, K)
 * @return (M, N, N, N, K)
 * */
at::Tensor query_vertical_backward(at::Tensor child, at::Tensor indices,
                                   at::Tensor grad_output, bool vary_non_leaf) {
    CHECK_INPUT(child);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(indices);
    TORCH_CHECK(indices.is_floating_point());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_output));
    return _query_vertical_backward_cuda(child, indices, grad_output,
                                         vary_non_leaf);
}

/**
 * @param data (M, N, N, N, K)
 * @param child (M, N, N, N)
 * @param indices (Q, 3)
 * @param values (Q, K)
 * */
void assign_vertical(at::Tensor data, at::Tensor child, at::Tensor indices,
                     at::Tensor values, bool vary_non_leaf) {
    CHECK_INPUT(data);
    CHECK_INPUT(child);
    CHECK_INPUT(indices);
    CHECK_INPUT(values);
    TORCH_CHECK(indices.is_floating_point());
    TORCH_CHECK(values.is_floating_point());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
    _assign_vertical_cuda(data, child, indices, values, vary_non_leaf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("query_vertical", &query_vertical, "Query tree at coords [0, 1)",
          py::arg("data"), py::arg("child"), py::arg("indices"),
          py::arg("vary_non_leaf"));
    m.def("query_vertical_backward", &query_vertical_backward,
          "Backwards pass for query_vertical", py::arg("child"),
          py::arg("indices"), py::arg("grad_output"), py::arg("vary_non_leaf"));
    m.def("assign_vertical", &assign_vertical,
          "Assign tree at given coords [0, 1)", py::arg("data"),
          py::arg("child"), py::arg("indices"), py::arg("values"),
          py::arg("vary_non_leaf"));
}

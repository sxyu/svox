#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <vector>
#include "common.hpp"

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
                                at::Tensor indices, at::Tensor offset,
                                at::Tensor invradius, int padding_mode);
at::Tensor _query_vertical_backward_cuda(at::Tensor child, at::Tensor indices,
                                         at::Tensor grad_output,
                                         at::Tensor offset,
                                         at::Tensor invradius,
                                         int padding_mode);
void _assign_vertical_cuda(at::Tensor data, at::Tensor child,
                           at::Tensor indices, at::Tensor values,
                           at::Tensor offset, at::Tensor invradius,
                           int padding_mode);
at::Tensor _render_cuda(at::Tensor data, at::Tensor child, at::Tensor rays,
                        at::Tensor offset, at::Tensor invradius,
                        bool white_bkgd);
/** END CUDA Interface **/

/**
 * @param data (M, N, N, N, K)
 * @param child (M, N, N, N)
 * @param indices (Q, 3)
 * @return (Q, K)
 * */
at::Tensor query_vertical(at::Tensor data, at::Tensor child, at::Tensor indices,
                          at::Tensor offset, at::Tensor invradius,
                          int padding_mode) {
    CHECK_INPUT(data);
    CHECK_INPUT(child);
    CHECK_INPUT(indices);
    CHECK_INPUT(offset);
    CHECK_INPUT(invradius);
    TORCH_CHECK(indices.dim() == 2);
    TORCH_CHECK(indices.is_floating_point());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
    return _query_vertical_cuda(data, child, indices, offset, invradius,
                                padding_mode);
}

/**
 * @param data (M, N, N, N, K)
 * @param child (M, N, N, N)
 * @param indices (Q, 3)
 * @param grad_output (Q, K)
 * @return (M, N, N, N, K)
 * */
at::Tensor query_vertical_backward(at::Tensor child, at::Tensor indices,
                                   at::Tensor grad_output, at::Tensor offset,
                                   at::Tensor invradius, int padding_mode) {
    CHECK_INPUT(child);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(indices);
    CHECK_INPUT(offset);
    CHECK_INPUT(invradius);
    TORCH_CHECK(indices.dim() == 2);
    TORCH_CHECK(indices.is_floating_point());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_output));
    return _query_vertical_backward_cuda(child, indices, grad_output, offset,
                                         invradius, padding_mode);
}

/**
 * @param data (M, N, N, N, K)
 * @param child (M, N, N, N)
 * @param indices (Q, 3)
 * @param values (Q, K)
 * */
void assign_vertical(at::Tensor data, at::Tensor child, at::Tensor indices,
                     at::Tensor values, at::Tensor offset, at::Tensor invradius,
                     int padding_mode) {
    CHECK_INPUT(data);
    CHECK_INPUT(child);
    CHECK_INPUT(indices);
    CHECK_INPUT(values);
    CHECK_INPUT(offset);
    CHECK_INPUT(invradius);
    TORCH_CHECK(indices.dim() == 2);
    TORCH_CHECK(values.dim() == 2);
    TORCH_CHECK(indices.is_floating_point());
    TORCH_CHECK(values.is_floating_point());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
    _assign_vertical_cuda(data, child, indices, values, offset, invradius,
                          padding_mode);
}

/**
 * @param rays [origins (3), directions(3), near(1), far(1)] (Q, 8)
 * */
// at::Tensor render(at::Tensor data, at::Tensor child, at::Tensor rays,
//                   at::Tensor offset, at::Tensor invradius, bool white_bkgd) {
//     CHECK_INPUT(data);
//     CHECK_INPUT(child);
//     CHECK_INPUT(rays);
//     CHECK_INPUT(offset);
//     CHECK_INPUT(invradius);
//     TORCH_CHECK(data.size(-1) >= 4);
//     TORCH_CHECK(rays.dim() == 2);
//     TORCH_CHECK(rays.size(1) == 8);
//     return _render_cuda(data, child, rays, offset, invradius, white_bkgd);
// }

int parse_padding_mode(const std::string& padding_mode) {
    if (padding_mode == "zeros") {
        return PADDING_MODE_ZEROS;
    } else if (padding_mode == "border") {
        return PADDING_MODE_BORDER;
    } else {
        throw std::invalid_argument("Unsupported padding mode");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("query_vertical", &query_vertical, "Query tree at coords [0, 1)",
          py::arg("data"), py::arg("child"), py::arg("indices"),
          py::arg("offset"), py::arg("invradius"), py::arg("padding_mode"));
    m.def("query_vertical_backward", &query_vertical_backward,
          "Backwards pass for query_vertical", py::arg("child"),
          py::arg("indices"), py::arg("grad_output"), py::arg("offset"),
          py::arg("invradius"), py::arg("padding_mode"));
    m.def("assign_vertical", &assign_vertical,
          "Assign tree at given coords [0, 1)", py::arg("data"),
          py::arg("child"), py::arg("indices"), py::arg("values"),
          py::arg("offset"), py::arg("invradius"), py::arg("padding_mode"));
    m.def("parse_padding_mode", &parse_padding_mode);
    // m.def("render", &render, py::arg("data"), py::arg("child"),
    // py::arg("rays"),
    //       py::arg("offset"), py::arg("invradius"), py::arg("white_bkgd"));
}

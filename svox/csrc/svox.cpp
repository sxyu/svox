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

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <vector>

namespace py = pybind11;

// Changed from x.type().is_cuda() due to deprecation
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor _query_vertical_cuda(torch::Tensor data, torch::Tensor child,
                                   torch::Tensor indices, torch::Tensor offset,
                                   torch::Tensor scaling);
torch::Tensor _query_vertical_backward_cuda(torch::Tensor child,
                                            torch::Tensor indices,
                                            torch::Tensor grad_output,
                                            torch::Tensor offset,
                                            torch::Tensor scaling);
void _assign_vertical_cuda(torch::Tensor data, torch::Tensor child,
                           torch::Tensor indices, torch::Tensor values,
                           torch::Tensor offset, torch::Tensor scaling);

torch::Tensor _volume_render_cuda(torch::Tensor data, torch::Tensor child,
                                  torch::Tensor origins, torch::Tensor dirs,
                                  torch::Tensor vdirs, torch::Tensor offset,
                                  torch::Tensor scaling, float step_size,
                                  float background_brightness, int sh_order,
                                  bool fast);

torch::Tensor _volume_render_image_cuda(
    torch::Tensor data, torch::Tensor child, torch::Tensor offset,
    torch::Tensor scaling, torch::Tensor c2w, float fx, float fy, int width,
    int height, float step_size, float background_brightness, int sh_order,
    int ndc_width, int ndc_height, float ndc_focal, bool fast);

torch::Tensor _volume_render_backward_cuda(
    torch::Tensor data, torch::Tensor child, torch::Tensor grad_output,
    torch::Tensor origins, torch::Tensor dirs, torch::Tensor vdirs,
    torch::Tensor offset, torch::Tensor scaling, float step_size,
    float background_brightness, int sh_order);

torch::Tensor _volume_render_image_backward_cuda(
    torch::Tensor data, torch::Tensor child, torch::Tensor grad_output,
    torch::Tensor offset, torch::Tensor scaling, torch::Tensor c2w, float fx,
    float fy, int width, int height, float step_size,
    float background_brightness, int sh_order, int ndc_width, int ndc_height,
    float ndc_focal);

/**
 * @param data (M, N, N, N, K)
 * @param child (M, N, N, N)
 * @param indices (Q, 3)
 * @return (Q, K)
 * */
torch::Tensor query_vertical(torch::Tensor data, torch::Tensor child,
                             torch::Tensor indices, torch::Tensor offset,
                             torch::Tensor scaling) {
    CHECK_INPUT(data);
    CHECK_INPUT(child);
    CHECK_INPUT(indices);
    CHECK_INPUT(offset);
    CHECK_INPUT(scaling);
    TORCH_CHECK(indices.dim() == 2);
    TORCH_CHECK(indices.is_floating_point());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
    return _query_vertical_cuda(data, child, indices, offset, scaling);
}

/**
 * @param data (M, N, N, N, K)
 * @param child (M, N, N, N)
 * @param indices (Q, 3)
 * @param grad_output (Q, K)
 * @return (M, N, N, N, K)
 * */
torch::Tensor query_vertical_backward(torch::Tensor child,
                                      torch::Tensor indices,
                                      torch::Tensor grad_output,
                                      torch::Tensor offset,
                                      torch::Tensor scaling) {
    CHECK_INPUT(child);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(indices);
    CHECK_INPUT(offset);
    CHECK_INPUT(scaling);
    TORCH_CHECK(indices.dim() == 2);
    TORCH_CHECK(indices.is_floating_point());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_output));
    return _query_vertical_backward_cuda(child, indices, grad_output, offset,
                                         scaling);
}

/**
 * @param data (M, N, N, N, K)
 * @param child (M, N, N, N)
 * @param indices (Q, 3)
 * @param values (Q, K)
 * */
void assign_vertical(torch::Tensor data, torch::Tensor child,
                     torch::Tensor indices, torch::Tensor values,
                     torch::Tensor offset, torch::Tensor scaling) {
    CHECK_INPUT(data);
    CHECK_INPUT(child);
    CHECK_INPUT(indices);
    CHECK_INPUT(values);
    CHECK_INPUT(offset);
    CHECK_INPUT(scaling);
    TORCH_CHECK(indices.dim() == 2);
    TORCH_CHECK(values.dim() == 2);
    TORCH_CHECK(indices.is_floating_point());
    TORCH_CHECK(values.is_floating_point());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
    _assign_vertical_cuda(data, child, indices, values, offset, scaling);
}

torch::Tensor volume_render(torch::Tensor data, torch::Tensor child,
                            torch::Tensor origins, torch::Tensor dirs,
                            torch::Tensor vdirs, torch::Tensor offset,
                            torch::Tensor scaling, float step_size,
                            float background_brightness, int sh_order,
                            bool fast) {
    CHECK_INPUT(data);
    CHECK_INPUT(child);
    CHECK_INPUT(origins);
    CHECK_INPUT(dirs);
    CHECK_INPUT(vdirs);
    CHECK_INPUT(offset);
    CHECK_INPUT(scaling);
    TORCH_CHECK(dirs.size(0) == vdirs.size(0));
    TORCH_CHECK(dirs.size(0) == origins.size(0));
    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
    return _volume_render_cuda(data, child, origins, dirs, vdirs, offset,
                               scaling, step_size, background_brightness,
                               sh_order, fast);
}

torch::Tensor volume_render_image(torch::Tensor data, torch::Tensor child,
                                  torch::Tensor offset, torch::Tensor scaling,
                                  torch::Tensor c2w, float fx, float fy,
                                  int width, int height, float step_size,
                                  float background_brightness, int sh_order,
                                  int ndc_width, int ndc_height,
                                  float ndc_focal, bool fast) {
    CHECK_INPUT(data);
    CHECK_INPUT(child);
    CHECK_INPUT(c2w);
    CHECK_INPUT(scaling);
    TORCH_CHECK(c2w.ndimension() == 2);
    TORCH_CHECK(c2w.size(1) == 4);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
    return _volume_render_image_cuda(data, child, offset, scaling, c2w, fx, fy,
                                     width, height, step_size,
                                     background_brightness, sh_order, ndc_width,
                                     ndc_height, ndc_focal, fast);
}

torch::Tensor volume_render_backward(torch::Tensor data, torch::Tensor child,
                                     torch::Tensor grad_output,
                                     torch::Tensor origins, torch::Tensor dirs,
                                     torch::Tensor vdirs, torch::Tensor offset,
                                     torch::Tensor scaling, float step_size,
                                     float background_brightness,
                                     int sh_order) {
    CHECK_INPUT(data);
    CHECK_INPUT(child);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(origins);
    CHECK_INPUT(dirs);
    CHECK_INPUT(vdirs);
    CHECK_INPUT(offset);
    CHECK_INPUT(scaling);
    TORCH_CHECK(dirs.size(0) == vdirs.size(0));
    TORCH_CHECK(dirs.size(0) == origins.size(0));
    TORCH_CHECK(grad_output.ndimension() == 2);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
    return _volume_render_backward_cuda(data, child, grad_output, origins, dirs,
                                        vdirs, offset, scaling, step_size,
                                        background_brightness, sh_order);
}

torch::Tensor volume_render_image_backward(
    torch::Tensor data, torch::Tensor child, torch::Tensor grad_output,
    torch::Tensor offset, torch::Tensor scaling, torch::Tensor c2w, float fx,
    float fy, int width, int height, float step_size,
    float background_brightness, int sh_order, int ndc_width, int ndc_height,
    float ndc_focal) {
    CHECK_INPUT(data);
    CHECK_INPUT(child);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(c2w);
    CHECK_INPUT(scaling);
    TORCH_CHECK(c2w.ndimension() == 2);
    TORCH_CHECK(c2w.size(1) == 4);
    TORCH_CHECK(grad_output.ndimension() == 3);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
    return _volume_render_image_backward_cuda(
        data, child, grad_output, offset, scaling, c2w, fx, fy, width, height,
        step_size, background_brightness, sh_order, ndc_width, ndc_height,
        ndc_focal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("query_vertical", &query_vertical, "Query tree at coords [0, 1)",
          py::arg("data"), py::arg("child"), py::arg("indices"),
          py::arg("offset"), py::arg("scaling"));
    m.def("query_vertical_backward", &query_vertical_backward,
          "Backwards pass for query_vertical", py::arg("child"),
          py::arg("indices"), py::arg("grad_output"), py::arg("offset"),
          py::arg("scaling"));
    m.def("assign_vertical", &assign_vertical,
          "Assign tree at given coords [0, 1)", py::arg("data"),
          py::arg("child"), py::arg("indices"), py::arg("values"),
          py::arg("offset"), py::arg("scaling"));

    m.def("volume_render", &volume_render, py::arg("data"), py::arg("child"),
          py::arg("origins"), py::arg("dirs"), py::arg("vdirs"),
          py::arg("offset"), py::arg("scaling"), py::arg("step_size"),
          py::arg("background_brightness"), py::arg("sh_order"),
          py::arg("fast"));
    m.def("volume_render_image", &volume_render_image, py::arg("data"),
          py::arg("child"), py::arg("offset"), py::arg("scaling"),
          py::arg("c2w"), py::arg("fx"), py::arg("fy"), py::arg("width"),
          py::arg("height"), py::arg("step_size"),
          py::arg("background_brightness"), py::arg("sh_order"),
          py::arg("ndc_width"), py::arg("ndc_height"), py::arg("ndc_focal"),
          py::arg("fast"));
    m.def("volume_render_backward", &volume_render_backward, py::arg("data"),
          py::arg("child"), py::arg("grad_output"), py::arg("origins"),
          py::arg("dirs"), py::arg("vdirs"), py::arg("offset"),
          py::arg("scaling"), py::arg("step_size"),
          py::arg("background_brightness"), py::arg("sh_order"));
    m.def("volume_render_image_backward", &volume_render_image_backward,
          py::arg("data"), py::arg("child"), py::arg("grad_output"),
          py::arg("offset"), py::arg("scaling"), py::arg("c2w"), py::arg("fx"),
          py::arg("fy"), py::arg("width"), py::arg("height"),
          py::arg("step_size"), py::arg("background_brightness"),
          py::arg("sh_order"), py::arg("ndc_width"), py::arg("ndc_height"),
          py::arg("ndc_focal"));
}

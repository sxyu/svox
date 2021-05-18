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

// This file contains only forward declarations and Python bindings

#include <torch/extension.h>
#include <cstdint>
#include <vector>

#include "data_spec.hpp"

namespace py = pybind11;
using torch::Tensor;

std::vector<torch::Tensor> grid_weight_render(torch::Tensor data,
                                              CameraSpec& cam,
                                              RenderOptions& opt,
                                              torch::Tensor offset,
                                              torch::Tensor scaling);

QueryResult query_vertical(TreeSpec&, Tensor);
Tensor query_vertical_backward(TreeSpec&, Tensor, Tensor);
void assign_vertical(TreeSpec&, Tensor, Tensor);

Tensor volume_render(TreeSpec&, RaysSpec&, RenderOptions&);
Tensor volume_render_image(TreeSpec&, CameraSpec&, RenderOptions&);
Tensor volume_render_backward(TreeSpec&, RaysSpec&, RenderOptions&, Tensor);
Tensor volume_render_image_backward(TreeSpec&, CameraSpec&, RenderOptions&,
                                    Tensor);

Tensor calc_corners(TreeSpec&, Tensor);

std::tuple<Tensor, Tensor> quantize_median_cut(Tensor data, Tensor, int32_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<RaysSpec>(m, "RaysSpec")
        .def(py::init<>())
        .def_readwrite("origins", &RaysSpec::origins)
        .def_readwrite("dirs", &RaysSpec::dirs)
        .def_readwrite("vdirs", &RaysSpec::vdirs);

    py::class_<TreeSpec>(m, "TreeSpec")
        .def(py::init<>())
        .def_readwrite("data", &TreeSpec::data)
        .def_readwrite("child", &TreeSpec::child)
        .def_readwrite("parent_depth", &TreeSpec::parent_depth)
        .def_readwrite("extra_data", &TreeSpec::extra_data)
        .def_readwrite("offset", &TreeSpec::offset)
        .def_readwrite("scaling", &TreeSpec::scaling)
        .def_readwrite("_weight_accum", &TreeSpec::_weight_accum)
        .def_readwrite("_weight_accum_max", &TreeSpec::_weight_accum_max);

    py::class_<CameraSpec>(m, "CameraSpec")
        .def(py::init<>())
        .def_readwrite("c2w", &CameraSpec::c2w)
        .def_readwrite("fx", &CameraSpec::fx)
        .def_readwrite("fy", &CameraSpec::fy)
        .def_readwrite("width", &CameraSpec::width)
        .def_readwrite("height", &CameraSpec::height);

    py::class_<RenderOptions>(m, "RenderOptions")
        .def(py::init<>())
        .def_readwrite("step_size", &RenderOptions::step_size)
        .def_readwrite("background_brightness",
                       &RenderOptions::background_brightness)
        .def_readwrite("ndc_width", &RenderOptions::ndc_width)
        .def_readwrite("ndc_height", &RenderOptions::ndc_height)
        .def_readwrite("ndc_focal", &RenderOptions::ndc_focal)
        .def_readwrite("format", &RenderOptions::format)
        .def_readwrite("basis_dim", &RenderOptions::basis_dim)
        .def_readwrite("min_comp", &RenderOptions::min_comp)
        .def_readwrite("max_comp", &RenderOptions::max_comp)
        .def_readwrite("sigma_thresh", &RenderOptions::sigma_thresh)
        .def_readwrite("stop_thresh", &RenderOptions::stop_thresh)
        .def_readwrite("density_softplus", &RenderOptions::density_softplus)
        .def_readwrite("rgb_padding", &RenderOptions::rgb_padding);

    m.def("query_vertical", &query_vertical);
    m.def("query_vertical_backward", &query_vertical_backward);
    m.def("assign_vertical", &assign_vertical);

    m.def("volume_render", &volume_render);
    m.def("volume_render_image", &volume_render_image);
    m.def("volume_render_backward", &volume_render_backward);
    m.def("volume_render_image_backward", &volume_render_image_backward);

    m.def("calc_corners", &calc_corners);

    m.def("grid_weight_render", &grid_weight_render);
    m.def("quantize_median_cut", &quantize_median_cut);
}

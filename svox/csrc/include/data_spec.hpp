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

#pragma once

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>

#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

// Changed from x.type().is_cuda() due to deprecation
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

enum DataFormat {
    FORMAT_RGBA,
    FORMAT_SH,
    FORMAT_SG,
    FORMAT_ASG,
};

struct RaysSpec {
    torch::Tensor origins;
    torch::Tensor dirs;
    torch::Tensor vdirs;

    inline void check() {
        CHECK_INPUT(origins);
        CHECK_INPUT(dirs);
        CHECK_INPUT(vdirs);
        TORCH_CHECK(origins.is_floating_point());
        TORCH_CHECK(dirs.is_floating_point());
        TORCH_CHECK(vdirs.is_floating_point());
    }
};

struct TreeSpec {
    torch::Tensor data;
    torch::Tensor child;
    torch::Tensor parent_depth;
    torch::Tensor extra_data;
    torch::Tensor offset;
    torch::Tensor scaling;
    torch::Tensor _weight_accum;
    bool _weight_accum_max;

    inline void check() {
        CHECK_INPUT(data);
        CHECK_INPUT(child);
        CHECK_INPUT(parent_depth);
        if (extra_data.numel()) {
            CHECK_INPUT(extra_data);
        }
        CHECK_INPUT(offset);
        CHECK_INPUT(scaling);
        if (_weight_accum.numel()) {
            CHECK_INPUT(_weight_accum);
        }
    }
};

struct CameraSpec {
    torch::Tensor c2w;
    float fx;
    float fy;
    int width;
    int height;

    inline void check() {
        CHECK_INPUT(c2w);
        TORCH_CHECK(c2w.is_floating_point());
        TORCH_CHECK(c2w.ndimension() == 2);
        TORCH_CHECK(c2w.size(1) == 4);
    }
};

// CUDA-ready
struct RenderOptions {
    float step_size;
    float background_brightness;

    int format;
    int basis_dim;

    int ndc_width;
    int ndc_height;
    float ndc_focal;

    int min_comp;
    int max_comp;

    float sigma_thresh;
    float stop_thresh;

    bool density_softplus;
    float rgb_padding;
};

using QueryResult = std::tuple<torch::Tensor, torch::Tensor>;

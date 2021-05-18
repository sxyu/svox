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
#include "data_spec.hpp"

template<class scalar_t>
struct SingleRaySpec {
    const scalar_t* __restrict__ origin;
    scalar_t* __restrict__ dir;
    const scalar_t* __restrict__ vdir;
};

template<class scalar_t>
struct PackedRaysSpec {
    PackedRaysSpec(RaysSpec& ray) :
        origins(ray.origins.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()),
        vdirs(ray.vdirs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()),
        dirs(ray.dirs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()) { }

    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        origins;
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        dirs;
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        vdirs;

    SingleRaySpec<scalar_t> operator[](int32_t i) {
        return SingleRaySpec<scalar_t>{&origins[i][0], &dirs[i][0], &vdirs[i][0]};
    }
};

template<class scalar_t>
struct PackedTreeSpec {
    PackedTreeSpec(TreeSpec& tree) :
        data(tree.data.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>()),
        child(tree.child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>()),
        parent_depth(tree.parent_depth.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>()),
        extra_data(tree.extra_data.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()),
        offset(tree.offset.data<scalar_t>()),
        scaling(tree.scaling.data<scalar_t>()),
        weight_accum(tree._weight_accum.numel() > 0 ? tree._weight_accum.data<scalar_t>() : nullptr),
        weight_accum_max(tree._weight_accum_max)
     { }

    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits>
        data;
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits>
        child;
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits>
        parent_depth;
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        extra_data;
    const scalar_t* __restrict__ offset;
    const scalar_t* __restrict__ scaling;
    scalar_t* __restrict__ weight_accum;
    bool weight_accum_max;
};

template<class scalar_t>
struct PackedCameraSpec {
    PackedCameraSpec(CameraSpec& cam) :
        c2w(cam.c2w.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()),
        fx(cam.fx), fy(cam.fy), width(cam.width), height(cam.height) {}
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        c2w;
    float fx;
    float fy;
    int width;
    int height;
};

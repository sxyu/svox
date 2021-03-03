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
        data(tree.data.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()),
        child(tree.child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>()),
        extra_data(tree.extra_data.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()),
        offset(tree.offset.data<scalar_t>()),
        scaling(tree.scaling.data<scalar_t>()),
        weight_accum(tree._weight_accum.numel() > 0 ? tree._weight_accum.data<scalar_t>() : nullptr) { }

    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        data;
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits>
        child;
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        extra_data;
    const scalar_t* __restrict__ offset;
    const scalar_t* __restrict__ scaling;
    scalar_t* __restrict__ weight_accum;
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

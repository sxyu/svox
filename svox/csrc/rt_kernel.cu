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
#include "common.cuh"

namespace {
namespace device {
const float C0 = 0.5 * sqrt(1.0 / M_PI);
const float C1 = sqrt(3 / (4 * M_PI));
const float C2[] = {0.5f * sqrtf(15.0f / M_PI),
    -0.5f * sqrtf(15.0f / M_PI),
    0.25f * sqrtf(5.0f / M_PI),
    -0.5f * sqrtf(15.0f / M_PI),
    0.25f * sqrtf(15.0f / M_PI)};

const float C3[] = {-0.25f * sqrtf(35 / (2 * M_PI)),
    0.5f * sqrtf(105 / M_PI),
    -0.25f * sqrtf(21/(2 * M_PI)),
    0.25f * sqrtf(7 / M_PI),
    -0.25f * sqrtf(21/(2 * M_PI)),
    0.25f * sqrtf(105 / M_PI),
    -0.25f * sqrtf(35/(2 * M_PI))
};

template <typename scalar_t>
__device__ __inline__ void _precalc_sh(
    const int order,
    const scalar_t* __restrict__ dir,
    scalar_t* __restrict__ out_mult) {

    out_mult[0] = C0;
    const scalar_t x = dir[0], y = dir[1], z = dir[2];
    out_mult[1] = - C1 * y;
    out_mult[2] = C1 * z;
    out_mult[3] = -C1 * x;
    if (order > 1) {
        const scalar_t xx = x * x, yy = y * y, zz = z * z;
        out_mult[4] = C2[0] * x * y;
        out_mult[5] = C2[1] * y * z;
        out_mult[6] = C2[2] * (2.0 * zz - xx - yy);
        out_mult[7] = C2[3] * x * z;
        out_mult[8] = C2[4] * (xx - yy);
        if (order > 2) {
            const scalar_t tmp_zzxxyy = 4 * zz - xx - yy;
            out_mult[9] = C3[0] * y * (3 * xx - yy);
            out_mult[10] = C3[1] * x * y * z;
            out_mult[11] = C3[2] * y * tmp_zzxxyy;
            out_mult[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
            out_mult[13] = C3[4] * x * tmp_zzxxyy;
            out_mult[14] = C3[5] * z * (xx - yy);
            out_mult[15] = C3[6] * x * (xx - 3 * yy);
        }
    }
}

template <typename scalar_t>
__device__ __inline__ void _dda_unit(
        const scalar_t* __restrict__ cen,
        const scalar_t* __restrict__ _invdir,
        scalar_t* __restrict__ tmin,
        scalar_t* __restrict__ tmax) {
    scalar_t t1, t2;
    *tmin = 0.0f;
    *tmax = 1e9f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * _invdir[i];
        t2 = t1 +  _invdir[i];
        *tmin = max(*tmin, min(t1, t2));
        *tmax = min(*tmax, max(t1, t2));
    }
}


template <typename scalar_t>
__device__ __inline__ void trace_ray(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
        data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits>
        child,
        const scalar_t* __restrict__ origin,
        const scalar_t* __restrict__ dir,
        const scalar_t* __restrict__ vdir,
        int sh_order,
        scalar_t step_size,
        scalar_t stop_thresh,
        scalar_t background_brightness,
        scalar_t* __restrict__ out) {

    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int tree_N = child.size(1);
    const int data_dim = data.size(4);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (dir[i] + 1e-9);
    }
    _dda_unit(origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        out[0] = out[1] = out[2] = background_brightness;
        return;
    } else {
        out[0] = out[1] = out[2] = 0.0f;
        scalar_t pos[3], tmp;
        const scalar_t* tree_val;
        scalar_t sh_mult[16];
        if (sh_order >= 0) {
            _precalc_sh(sh_order, vdir, sh_mult);
        }

        scalar_t light_intensity = 1.f;
        scalar_t t = tmin;
        const int n_coe = (sh_order + 1) * (sh_order + 1);
        scalar_t cube_sz;
        while (t < tmax) {
            for (int j = 0; j < 3; ++j) {
                pos[j] = origin[j] + t * dir[j];
            }

            ::device::query_single_from_root(data, child,
                    pos, &tree_val, &cube_sz, tree_N, data_dim);

            scalar_t att;
            scalar_t subcube_tmin, subcube_tmax;
            _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

            const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
            const scalar_t delta_t = t_subcube + step_size;
        att = expf(-delta_t * tree_val[data_dim - 1]);
        const scalar_t weight = light_intensity * (1.f - att);

        if (sh_order >= 0) {
#pragma unroll 3
            for (int t = 0; t < 3; ++ t) {
                int off = t * n_coe;
                tmp = sh_mult[0] * tree_val[off] +
                    sh_mult[1] * tree_val[off + 1] +
                    sh_mult[2] * tree_val[off + 2];
#pragma unroll 6
                for (int i = 3; i < n_coe; ++i) {
                    tmp += sh_mult[i] * tree_val[off + i];
                }
                out[t] += weight / (1.f + expf(-tmp));
            }
        } else {
            for (int j = 0; j < 3; ++j) {
                out[j] += tree_val[j] * weight;
            }
        }

        light_intensity *= att;

        if (light_intensity < stop_thresh) {
            // Almost full opacity, stop
            scalar_t scale = 1.0 / (1.0 - light_intensity);
            out[0] *= scale;
            out[1] *= scale;
            out[2] *= scale;
            return;
        }
            t += delta_t;
        }
        out[0] += light_intensity * background_brightness;
        out[1] += light_intensity * background_brightness;
        out[2] += light_intensity * background_brightness;
    }
}

template <typename scalar_t>
__global__ void render_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
        data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits>
        child,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        origin,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        dir,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        vdir,
    int sh_order,
    scalar_t step_size,
    scalar_t stop_thresh,
    scalar_t background_brightness,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        out) {
    CUDA_GET_THREAD_ID(tid, dir.size(0));

    trace_ray<scalar_t>(
        data, child,
        &origin[tid][0],
        &dir[tid][0],
        &vdir[tid][0],
        sh_order,
        step_size,
        stop_thresh,
        background_brightness,
        &out[tid][0]);
}

}  // namespace device
}  // namespace

torch::Tensor _volume_render_cuda(torch::Tensor data, torch::Tensor child,
                            torch::Tensor origins, torch::Tensor dirs,
                            torch::Tensor vdirs, torch::Tensor offset,
                            torch::Tensor invradius, float step_size,
                            float stop_thresh, float background_brightness) {
    const auto Q = origins.size(0), K = data.size(4);

    const int blocks = CUDA_N_BLOCKS_NEEDED(Q);
    torch::Tensor result = torch::zeros({Q, 3}, indices.options());
    AT_DISPATCH_FLOATING_TYPES(origins.type(), __FUNCTION__, [&] {
            device::render_kernel<scalar_t><<<blocks, CUDA_N_THREADS>>>(
                data.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                offset.data<scalar_t>(),
                invradius.data<scalar_t>(),
                result.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

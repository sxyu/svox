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
int cuda_n_threads = -1;
__host__ void auto_cuda_threads() {
    if (~cuda_n_threads) return;
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    const int n_cores = get_sp_cores(dev_prop);
    // Optimize number of CUDA threads per block
    if (n_cores < 2048) {
        cuda_n_threads = 256;
    } if (n_cores < 8192) {
        cuda_n_threads = 512;
    } else {
        cuda_n_threads = 1024;
    }
}

namespace device {
// SH Coefficients from https://github.com/google/spherical-harmonics
__device__ __constant__ const float C0 = 0.28209479177387814;
__device__ __constant__ const float C1 = 0.4886025119029199;
__device__ __constant__ const float C2[] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};

__device__ __constant__ const float C3[] = {
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
};

__device__ __constant__ const float C4[] = {
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
};

// Calculate SH basis functions of given order, for given view directions
template <typename scalar_t>
__device__ __inline__ void _precalc_sh(
    const int order,
    const scalar_t* __restrict__ dir,
    scalar_t* __restrict__ out_mult) {

    out_mult[0] = C0;
    const float x = dir[0], y = dir[1], z = dir[2];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    switch (order) {
        case 4:
            out_mult[16] = C4[0] * xy * (xx - yy);
            out_mult[17] = C4[1] * yz * (3 * xx - yy);
            out_mult[18] = C4[2] * xy * (7 * zz - 1.f);
            out_mult[19] = C4[3] * yz * (7 * zz - 3.f);
            out_mult[20] = C4[4] * (zz * (35 * zz - 30) + 3);
            out_mult[21] = C4[5] * xz * (7 * zz - 3);
            out_mult[22] = C4[6] * (xx - yy) * (7 * zz - 1.f);
            out_mult[23] = C4[7] * xz * (xx - 3 * yy);
            out_mult[24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
            [[fallthrough]];
        case 3:
            out_mult[9] = C3[0] * y * (3 * xx - yy);
            out_mult[10] = C3[1] * xy * z;
            out_mult[11] = C3[2] * y * (4 * zz - xx - yy);
            out_mult[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
            out_mult[13] = C3[4] * x * (4 * zz - xx - yy);
            out_mult[14] = C3[5] * z * (xx - yy);
            out_mult[15] = C3[6] * x * (xx - 3 * yy);
            [[fallthrough]];
        case 2:
            out_mult[4] = C2[0] * xy;
            out_mult[5] = C2[1] * yz;
            out_mult[6] = C2[2] * (2.0 * zz - xx - yy);
            out_mult[7] = C2[3] * xz;
            out_mult[8] = C2[4] * (xx - yy);
            [[fallthrough]];
        case 1:
            out_mult[1] = -C1 * y;
            out_mult[2] = C1 * z;
            out_mult[3] = -C1 * x;
    }
}

template <typename scalar_t>
__device__ __inline__ scalar_t _get_delta_scale(
    const scalar_t* __restrict__ scaling,
    const scalar_t* __restrict__ dir) {
    const scalar_t dx = dir[0] / scaling[0],
                   dy = dir[1] / scaling[1],
                   dz = dir[2] / scaling[2];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

template <typename scalar_t>
__device__ __inline__ void _dda_unit(
        const scalar_t* __restrict__ cen,
        const scalar_t* __restrict__ invdir,
        scalar_t* __restrict__ tmin,
        scalar_t* __restrict__ tmax) {
    // Perform DDA for 1 iteration on a unit cube
    scalar_t t1, t2;
    *tmin = 0.0f;
    *tmax = 1e9f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * invdir[i];
        t2 = t1 +  invdir[i];
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
        scalar_t step_size,
        scalar_t background_brightness,
        int sh_order,
        scalar_t delta_scale,
        scalar_t sigma_thresh,
        scalar_t stop_thresh,
        torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t> out) {

    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int tree_N = child.size(1);
    const int data_dim = data.size(4);
    const int out_data_dim = out.size(0);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (dir[i] + 1e-9);
    }
    _dda_unit(origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] = background_brightness;
        }
        return;
    } else {
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] = 0.f;
        }
        scalar_t pos[3];
        scalar_t sh_mult[25];
        if (sh_order >= 0) {
            _precalc_sh<scalar_t>(sh_order, vdir, sh_mult);
        }

        scalar_t light_intensity = 1.f;
        scalar_t t = tmin;
        const int n_coe = (sh_order + 1) * (sh_order + 1);
        scalar_t cube_sz;
        while (t < tmax) {
            for (int j = 0; j < 3; ++j) {
                pos[j] = origin[j] + t * dir[j];
            }

            scalar_t* tree_val = query_single_from_root<scalar_t>(data, child,
                        pos, &cube_sz);

            scalar_t att;
            scalar_t subcube_tmin, subcube_tmax;
            _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

            const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
            const scalar_t delta_t = t_subcube + step_size;
            const scalar_t sigma = tree_val[data_dim - 1];
            if (sigma > sigma_thresh) {
                att = expf(-delta_t * delta_scale * sigma);
                const scalar_t weight = light_intensity * (1.f - att);

                if (sh_order >= 0) {
                    for (int t = 0; t < out_data_dim; ++ t) {
                        int off = t * n_coe;
                        scalar_t tmp = 0.0;
                        for (int i = 0; i < n_coe; ++i) {
                            tmp += sh_mult[i] * tree_val[off + i];
                        }
                        out[t] += weight / (1.0 + expf(-tmp));
                    }
                } else {
                    for (int j = 0; j < out_data_dim; ++j) {
                        out[j] += weight / (1.0 + expf(-tree_val[j]));
                    }
                }
                light_intensity *= att;

                if (light_intensity <= stop_thresh) {
                    // Full opacity, stop
                    scalar_t scale = 1.0 / (1.0 - light_intensity);
                    out[0] *= scale; out[1] *= scale; out[2] *= scale;
                    return;
                }
            }
            t += delta_t;
        }
        for (int j = 0; j < out_data_dim; ++j) {
            out[j] += light_intensity * background_brightness;
        }
    }
}

template <typename scalar_t>
__device__ __inline__ void trace_ray_backward(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
        data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits>
        child,
    const torch::TensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, int32_t>
        grad_out,
        const scalar_t* __restrict__ origin,
        const scalar_t* __restrict__ dir,
        const scalar_t* __restrict__ vdir,
        scalar_t step_size,
        scalar_t background_brightness,
        int sh_order,
        scalar_t delta_scale,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
        grad_data) {

    scalar_t tmin, tmax;
    scalar_t invdir[3];
    const int tree_N = child.size(1);
    const int data_dim = data.size(4);
    const int out_data_dim = grad_out.size(0);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.0 / (dir[i] + 1e-9);
    }
    _dda_unit(origin, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        return;
    } else {
        scalar_t pos[3];
        scalar_t sh_mult[16];
        if (sh_order >= 0) {
            _precalc_sh<scalar_t>(sh_order, vdir, sh_mult);
        }
        const int n_coe = (sh_order + 1) * (sh_order + 1);

        scalar_t accum = 0.0;
        // PASS 1
        {
            scalar_t light_intensity = 1.f, t = tmin, cube_sz;
            while (t < tmax) {
                for (int j = 0; j < 3; ++j) pos[j] = origin[j] + t * dir[j];

                const scalar_t* tree_val = query_single_from_root<scalar_t>(data, child,
                        pos, &cube_sz);
                // Reuse offset on gradient
                const int curr_leaf_offset = tree_val - data.data();
                scalar_t* grad_tree_val = grad_data.data() + curr_leaf_offset;

                scalar_t att;
                scalar_t subcube_tmin, subcube_tmax;
                _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

                const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                const scalar_t delta_t = t_subcube + step_size;
                const scalar_t sigma = tree_val[data_dim - 1];
                if (sigma > 0.0) {
                    att = expf(-delta_t * sigma * delta_scale);
                    const scalar_t weight = light_intensity * (1.f - att);

                    scalar_t total_color = 0.f;
                    if (sh_order >= 0) {
                        for (int t = 0; t < out_data_dim; ++ t) {
                            int off = t * n_coe;
                            scalar_t tmp = 0.0;
                            for (int i = 0; i < n_coe; ++i) {
                                tmp += sh_mult[i] * tree_val[off + i];
                            }
                            const scalar_t sigmoid = 1.0 / (1.0 + expf(-tmp));
                            const scalar_t grad_sigmoid = sigmoid * (1.0 - sigmoid);
                            for (int i = 0; i < n_coe; ++i) {
                                const scalar_t toadd = weight * sh_mult[i] *
                                    grad_sigmoid * grad_out[t];
                                atomicAdd(&grad_tree_val[off + i],
                                        toadd);
                            }
                            total_color += sigmoid * grad_out[t];
                        }
                    } else {
                        for (int j = 0; j < out_data_dim; ++j) {
                            const scalar_t sigmoid = 1.0 / (1.0 + expf(-tree_val[j]));
                            const scalar_t toadd = weight * sigmoid * (1.f - sigmoid) * grad_out[j];
                            atomicAdd(&grad_tree_val[j], toadd);
                            total_color += sigmoid * grad_out[j];
                        }
                    }
                    light_intensity *= att;
                    accum += weight * total_color;
                }
                t += delta_t;
            }
            scalar_t total_grad = 0.f;
            for (int j = 0; j < out_data_dim; ++j)
                total_grad += grad_out[j];
            accum += light_intensity * background_brightness * total_grad;
        }
        // PASS 2
        {
            // scalar_t accum_lo = 0.0;
            scalar_t light_intensity = 1.f, t = tmin, cube_sz;
            while (t < tmax) {
                for (int j = 0; j < 3; ++j) pos[j] = origin[j] + t * dir[j];
                const scalar_t* tree_val = query_single_from_root<scalar_t>(data, child,
                        pos, &cube_sz);
                // Reuse offset on gradient
                const int curr_leaf_offset = tree_val - data.data();
                scalar_t* grad_tree_val = grad_data.data() + curr_leaf_offset;

                scalar_t att;
                scalar_t subcube_tmin, subcube_tmax;
                _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

                const scalar_t t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
                const scalar_t delta_t = t_subcube + step_size;
                const scalar_t sigma = tree_val[data_dim - 1];
                if (sigma > 0.0) {
                    att = expf(-delta_t * sigma * delta_scale);
                    const scalar_t weight = light_intensity * (1.f - att);

                    scalar_t total_color = 0.f;
                    if (sh_order >= 0) {
                        for (int t = 0; t < out_data_dim; ++ t) {
                            int off = t * n_coe;
                            scalar_t tmp = 0.0;
                            for (int i = 0; i < n_coe; ++i) {
                                tmp += sh_mult[i] * tree_val[off + i];
                            }
                            total_color += 1.0 / (1.0 + expf(-tmp)) * grad_out[t];
                        }
                    } else {
                        for (int j = 0; j < out_data_dim; ++j) {
                            total_color += 1.0 / (1.0 + expf(-tree_val[j])) * grad_out[j];
                        }
                    }
                    light_intensity *= att;
                    accum -= weight * total_color;
                    atomicAdd(
                            &grad_tree_val[out_data_dim],
                            delta_t * delta_scale * (
                                total_color * light_intensity - accum)
                            );
                }
                t += delta_t;
            }
        }
    }
}

template <typename scalar_t>
__global__ void render_ray_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
        data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits>
        child,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        origins,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        dirs,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        vdirs,
    scalar_t step_size,
    scalar_t background_brightness,
    int sh_order,
    scalar_t sigma_thresh,
    scalar_t stop_thresh,
    const scalar_t* __restrict__ offset,
    const scalar_t* __restrict__ scaling,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        out
        ) {
    CUDA_GET_THREAD_ID(tid, origins.size(0));
    scalar_t origin[3] = {origins[tid][0], origins[tid][1], origins[tid][2]};
    transform_coord<scalar_t>(origin, offset, scaling);
    const scalar_t delta_scale = _get_delta_scale(scaling, &dirs[tid][0]);

    trace_ray<scalar_t>(
        data, child,
        origin,
        &dirs[tid][0],
        &vdirs[tid][0],
        step_size,
        background_brightness,
        sh_order,
        delta_scale,
        sigma_thresh,
        stop_thresh,
        out[tid]);
}


template <typename scalar_t>
__global__ void render_ray_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
        data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits>
        child,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        grad_out,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        origins,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        dirs,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        vdirs,
    scalar_t step_size,
    scalar_t background_brightness,
    int sh_order,
    const scalar_t* __restrict__ offset,
    const scalar_t* __restrict__ scaling,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
        grad_data
        ) {
    CUDA_GET_THREAD_ID(tid, origins.size(0));
    scalar_t origin[3] = {origins[tid][0], origins[tid][1], origins[tid][2]};
    transform_coord<scalar_t>(origin, offset, scaling);
    const scalar_t delta_scale = _get_delta_scale(scaling, &dirs[tid][0]);
    trace_ray_backward<scalar_t>(
        data, child,
        grad_out[tid],
        origin,
        &dirs[tid][0],
        &vdirs[tid][0],
        step_size,
        background_brightness,
        sh_order,
        delta_scale,
        grad_data);
}


template <typename scalar_t>
__device__ __inline__ void cam2world_ray(
    int ix, int iy,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        c2w,
    scalar_t* dir,
    scalar_t* origin,
    scalar_t fx, scalar_t fy,
    int width, int height) {
    scalar_t x = (ix - 0.5 * width) / fx;
    scalar_t y = -(iy - 0.5 * height) / fy;
    scalar_t z = sqrtf(x * x + y * y + 1.0);
    x /= z; y /= z; z = -1.0f / z;
    dir[0] = c2w[0][0] * x + c2w[0][1] * y + c2w[0][2] * z;
    dir[1] = c2w[1][0] * x + c2w[1][1] * y + c2w[1][2] * z;
    dir[2] = c2w[2][0] * x + c2w[2][1] * y + c2w[2][2] * z;
    origin[0] = c2w[0][3]; origin[1] = c2w[1][3]; origin[2] = c2w[2][3];
}


template <typename scalar_t>
__host__ __device__ __inline__ static void world2ndc(
        int ndc_width, int ndc_height, scalar_t ndc_focal,
        scalar_t* __restrict__ dir,
        scalar_t* __restrict__ cen, scalar_t near = 1.f) {
    scalar_t t = -(near + cen[2]) / dir[2];
    for (int i = 0; i < 3; ++i) {
        cen[i] = cen[i] + t * dir[i];
    }

    dir[0] = -((2 * ndc_focal) / ndc_width) * (dir[0] / dir[2] - cen[0] / cen[2]);
    dir[1] = -((2 * ndc_focal) / ndc_height) * (dir[1] / dir[2] - cen[1] / cen[2]);
    dir[2] = -2 * near / cen[2];

    cen[0] = -((2 * ndc_focal) / ndc_width) * (cen[0] / cen[2]);
    cen[1] = -((2 * ndc_focal) / ndc_height) * (cen[1] / cen[2]);
    cen[2] = 1 + 2 * near / cen[2];

    scalar_t norm = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    dir[0] /= norm; dir[1] /= norm; dir[2] /= norm;
}


template <typename scalar_t>
__global__ void render_image_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
        data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits>
        child,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        c2w,
    scalar_t step_size,
    scalar_t background_brightness,
    int sh_order,
    scalar_t sigma_thresh,
    scalar_t stop_thresh,
    scalar_t fx,
    scalar_t fy,
    int width,
    int height,
    scalar_t ndc_focal,
    int ndc_width,
    int ndc_height,
    const scalar_t* __restrict__ offset,
    const scalar_t* __restrict__ scaling,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        out
        ) {
    CUDA_GET_THREAD_ID(tid, width * height);
    int iy = tid / width, ix = tid % width;
    scalar_t dir[3], origin[3];
    cam2world_ray(ix, iy, c2w, dir, origin, fx, fy, width, height);
    scalar_t vdir[3] = {dir[0], dir[1], dir[2]};
    if (ndc_width > 1) {
        world2ndc(ndc_width, ndc_height, ndc_focal, dir, origin);
    }

    transform_coord<scalar_t>(origin, offset, scaling);
    const scalar_t delta_scale = _get_delta_scale(scaling, dir);
    trace_ray<scalar_t>(
        data, child,
        origin,
        dir,
        vdir,
        step_size,
        background_brightness,
        sh_order,
        delta_scale,
        sigma_thresh,
        stop_thresh,
        out[iy][ix]);
}

template <typename scalar_t>
__global__ void render_image_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
        data,
    const torch::PackedTensorAccessor32<int32_t, 4, torch::RestrictPtrTraits>
        child,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        c2w,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        grad_out,
    scalar_t step_size,
    scalar_t background_brightness,
    int sh_order,
    scalar_t fx,
    scalar_t fy,
    int width,
    int height,
    scalar_t ndc_focal,
    int ndc_width,
    int ndc_height,
    const scalar_t* __restrict__ offset,
    const scalar_t* __restrict__ scaling,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
        grad_data
        ) {
    CUDA_GET_THREAD_ID(tid, width * height);
    int iy = tid / width, ix = tid % width;
    scalar_t dir[3], origin[3];
    cam2world_ray(ix, iy, c2w, dir, origin, fx, fy, width, height);
    scalar_t vdir[3] = {dir[0], dir[1], dir[2]};
    if (ndc_width > 1) {
        world2ndc(ndc_width, ndc_height, ndc_focal, dir, origin);
    }

    transform_coord<scalar_t>(origin, offset, scaling);
    const scalar_t delta_scale = _get_delta_scale(scaling, dir);
    trace_ray_backward<scalar_t>(
        data, child,
        grad_out[iy][ix],
        origin,
        dir,
        vdir,
        step_size,
        background_brightness,
        sh_order,
        delta_scale,
        grad_data);
}

}  // namespace device


// Compute RGB output dimension from input dimension & SH order
__host__ int get_out_data_dim(int sh_order, int in_data_dim) {
    int out_data_dim;
    if (sh_order >= 0) {
        const int n_coe = (sh_order + 1) * (sh_order + 1);
        out_data_dim = (in_data_dim - 1) / n_coe;
    } else {
        out_data_dim = in_data_dim - 1;
    }
    return out_data_dim;
}

}  // namespace

torch::Tensor _volume_render_cuda(torch::Tensor data, torch::Tensor child,
                            torch::Tensor origins, torch::Tensor dirs,
                            torch::Tensor vdirs, torch::Tensor offset,
                            torch::Tensor scaling, float step_size,
                            float background_brightness,
                            int sh_order, bool fast) {
    const auto Q = origins.size(0);

    const float sigma_thresh = fast ? 1e-2f : 0.f;
    const float stop_thresh = fast ? 1e-2f : 0.f;

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int out_data_dim = get_out_data_dim(sh_order, data.size(4));
    torch::Tensor result = torch::zeros({Q, out_data_dim}, origins.options());
    AT_DISPATCH_FLOATING_TYPES(origins.type(), __FUNCTION__, [&] {
            device::render_ray_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                data.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                origins.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                dirs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                vdirs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                step_size,
                background_brightness,
                sh_order,
                sigma_thresh,
                stop_thresh,
                offset.data<scalar_t>(),
                scaling.data<scalar_t>(),
                result.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

torch::Tensor _volume_render_image_cuda(
    torch::Tensor data, torch::Tensor child, torch::Tensor offset,
    torch::Tensor scaling, torch::Tensor c2w, float fx, float fy, int width,
    int height, float step_size,
    float background_brightness, int sh_order, int ndc_width, int ndc_height,
    float ndc_focal, bool fast) {
    const size_t Q = size_t(width) * height;

    const float sigma_thresh = fast ? 1e-2f : 0.f;
    const float stop_thresh = fast ? 1e-2f : 0.f;

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int out_data_dim = get_out_data_dim(sh_order, data.size(4));
    torch::Tensor result = torch::zeros({height, width, out_data_dim}, data.options());

    AT_DISPATCH_FLOATING_TYPES(data.type(), __FUNCTION__, [&] {
            device::render_image_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                data.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                c2w.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                step_size,
                background_brightness,
                sh_order,
                sigma_thresh,
                stop_thresh,
                fx,
                fy,
                width,
                height,
                ndc_focal,
                ndc_width,
                ndc_height,
                offset.data<scalar_t>(),
                scaling.data<scalar_t>(),
                result.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

torch::Tensor _volume_render_backward_cuda(
    torch::Tensor data, torch::Tensor child, torch::Tensor grad_output,
    torch::Tensor origins, torch::Tensor dirs, torch::Tensor vdirs,
    torch::Tensor offset, torch::Tensor scaling, float step_size,
    float background_brightness, int sh_order) {
    const int Q = origins.size(0);

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int out_data_dim = get_out_data_dim(sh_order, data.size(4));
    torch::Tensor result = torch::zeros_like(data);
    AT_DISPATCH_FLOATING_TYPES(origins.type(), __FUNCTION__, [&] {
            device::render_ray_backward_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                data.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                origins.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                dirs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                vdirs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                step_size,
                background_brightness,
                sh_order,
                offset.data<scalar_t>(),
                scaling.data<scalar_t>(),
                result.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

torch::Tensor _volume_render_image_backward_cuda(
    torch::Tensor data, torch::Tensor child, torch::Tensor grad_output,
    torch::Tensor offset, torch::Tensor scaling, torch::Tensor c2w, float fx,
    float fy, int width, int height, float step_size,
    float background_brightness, int sh_order, int ndc_width, int ndc_height,
    float ndc_focal) {
    const size_t Q = size_t(width) * height;

    auto_cuda_threads();
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, cuda_n_threads);
    int out_data_dim = get_out_data_dim(sh_order, data.size(4));
    torch::Tensor result = torch::zeros_like(data);

    AT_DISPATCH_FLOATING_TYPES(data.type(), __FUNCTION__, [&] {
            device::render_image_backward_kernel<scalar_t><<<blocks, cuda_n_threads>>>(
                data.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                child.packed_accessor32<int32_t, 4, torch::RestrictPtrTraits>(),
                c2w.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                step_size,
                background_brightness,
                sh_order,
                fx,
                fy,
                width,
                height,
                ndc_focal,
                ndc_width,
                ndc_height,
                offset.data<scalar_t>(),
                scaling.data<scalar_t>(),
                result.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>());
    });
    CUDA_CHECK_ERRORS;
    return result;
}

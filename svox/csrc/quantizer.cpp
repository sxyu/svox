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

#include <torch/extension.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <tuple>
#include <limits>
#include <cstdint>

namespace {

template <class scalar_t>
struct Comparer {
    Comparer(const torch::TensorAccessor<scalar_t, 2> data) : data(data) {}
    bool operator()(int64_t a, int64_t b) const {
        return data[a][dim] < data[b][dim];
    }
    const torch::TensorAccessor<scalar_t, 2> data;
    int dim;
};

template <class scalar_t>
void _quantize_median_cut_impl(
    const torch::TensorAccessor<scalar_t, 2> data,
    const torch::TensorAccessor<scalar_t, 1> weights,
    std::vector<int64_t>& tmp_rev_map,
    torch::TensorAccessor<scalar_t, 2> colors_out,
    torch::TensorAccessor<int32_t, 1> color_id_map_out, int32_t order,
    int64_t l, int64_t r, int32_t& color_idx, Comparer<scalar_t>& comp) {
    const int K = data.size(1);
    scalar_t total_weight = 0.0;
    const bool use_weights = weights.size(0) > 0;
    if (order <= 0 || r - l <= 1) {
        torch::TensorAccessor<scalar_t, 1> color = colors_out[color_idx];
        for (int i = l; i < r; ++i) {
            const int64_t ii = tmp_rev_map[i];
            for (int j = 0; j < K; ++j) {
                scalar_t entry = data[ii][j];
                if (use_weights) entry *= weights[ii];
                color[j] += entry;
            }
            if (use_weights) total_weight += weights[ii];
            color_id_map_out[ii] = color_idx;
        }
        if (!use_weights) {
            total_weight = r - l;
        }
        // std::nth_element(tmp_rev_map.data() + l, tmp_rev_map.data() + m,
        //                  tmp_rev_map.data() + r, comp);
        for (int j = 0; j < K; ++j) {
            color[j] /= total_weight;
        }
        ++color_idx;
    } else {
        const scalar_t MAX_VAL = std::numeric_limits<scalar_t>::max();
        comp.dim = 0;
        {
            std::vector<scalar_t> mins(K, MAX_VAL), maxs(K, -MAX_VAL);
            for (int i = l; i < r; ++i) {
                const int64_t ii = tmp_rev_map[i];
                if (use_weights) total_weight += weights[ii];
                for (int j = 0; j < K; ++j) {
                    const scalar_t val = data[ii][j];
                    maxs[j] = std::max(maxs[j], val);
                    mins[j] = std::min(mins[j], val);
                }
            }
            scalar_t largest_var = -1.0;
            for (int j = 0; j < K; ++j) {
                if (maxs[j] - mins[j] > largest_var) {
                    comp.dim = j;
                    largest_var = maxs[j] - mins[j];
                }
            }
        }

        int64_t m;
        if (!use_weights) {
            m = l + (r - l) / 2;
            std::nth_element(tmp_rev_map.data() + l, tmp_rev_map.data() + m,
                             tmp_rev_map.data() + r, comp);
        } else {
            std::sort(tmp_rev_map.data() + l, tmp_rev_map.data() + r, comp);
            scalar_t pfxsum = 0.0;
            for (m = l; m < r; ++m) {
                pfxsum += weights[tmp_rev_map[m]];
                if (pfxsum > total_weight * 0.5) {
                    break;
                }
            }
        }

        _quantize_median_cut_impl(data, weights, tmp_rev_map, colors_out,
                                  color_id_map_out, order - 1, l, m, color_idx,
                                  comp);
        _quantize_median_cut_impl(data, weights, tmp_rev_map, colors_out,
                                  color_id_map_out, order - 1, m, r, color_idx,
                                  comp);
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> quantize_median_cut(
    torch::Tensor data, torch::Tensor weights, int32_t order) {
    TORCH_CHECK(data.is_contiguous());
    TORCH_CHECK(weights.is_contiguous());
    TORCH_CHECK(!data.is_cuda());
    TORCH_CHECK(order < 31);
    TORCH_CHECK(data.dim() == 2);
    const int32_t N_COLORS = 1 << order;
    TORCH_CHECK(N_COLORS <= data.size(0));
    auto options = at::TensorOptions()
                       .dtype(at::kInt)
                       .layout(data.layout())
                       .device(data.device());
    torch::Tensor colors =
        torch::zeros({N_COLORS, data.size(1)}, data.options());
    torch::Tensor color_id_map = torch::zeros({data.size(0)}, options);
    std::vector<int64_t> tmp(data.size(0));
    std::iota(tmp.begin(), tmp.end(), 0);
    AT_DISPATCH_FLOATING_TYPES(data.type(), __FUNCTION__, [&] {
        int32_t color_idx = 0;
        Comparer<scalar_t> comp(data.accessor<scalar_t, 2>());
        _quantize_median_cut_impl<scalar_t>(
            comp.data, weights.accessor<scalar_t, 1>(), tmp,
            colors.accessor<scalar_t, 2>(), color_id_map.accessor<int32_t, 1>(),
            order, 0, data.size(0), color_idx, comp);
    });
    return std::tuple<torch::Tensor, torch::Tensor>(colors, color_id_map);
}

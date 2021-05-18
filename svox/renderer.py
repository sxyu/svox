#  Copyright 2021 PlenOctree Authors.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
"""
Volume rendering utilities
"""

import torch
import numpy as np
from torch import nn, autograd
from collections import namedtuple
from warnings import warn

from svox.helpers import _get_c_extension, LocalIndex, DataFormat

NDCConfig = namedtuple('NDCConfig', ["width", "height", "focal"])
Rays = namedtuple('Rays', ["origins", "dirs", "viewdirs"])

_C = _get_c_extension()

def _rays_spec_from_rays(rays):
    spec = _C.RaysSpec()
    spec.origins = rays.origins
    spec.dirs = rays.dirs
    spec.vdirs = rays.viewdirs
    return spec

def _make_camera_spec(c2w, width, height, fx, fy):
    spec = _C.CameraSpec()
    spec.c2w = c2w
    spec.width = width
    spec.height = height
    spec.fx = fx
    spec.fy = fy
    return spec

class _VolumeRenderFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, tree, rays, opt):
        out = _C.volume_render(tree, rays, opt)
        ctx.tree = tree
        ctx.rays = rays
        ctx.opt = opt
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.needs_input_grad[0]:
            return _C.volume_render_backward(
                ctx.tree, ctx.rays, ctx.opt, grad_out.contiguous()
            ), None, None, None
        return None, None, None, None

class _VolumeRenderImageFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, tree, cam, opt):
        out = _C.volume_render_image(tree, cam, opt)
        ctx.tree = tree
        ctx.cam = cam
        ctx.opt = opt
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.needs_input_grad[0]:
            return _C.volume_render_image_backward(
                ctx.tree, ctx.cam, ctx.opt, grad_out.contiguous()
            ), None, None, None
        return None, None, None, None


def convert_to_ndc(origins, directions, focal, w, h, near=1.0):
    """Convert a set of rays to NDC coordinates. (only for grad check)"""
    # Shift ray origins to near plane
    t = -(near + origins[..., 2]) / directions[..., 2]
    origins = origins + t[..., None] * directions

    dx, dy, dz = directions.unbind(-1)
    ox, oy, oz = origins.unbind(-1)

    # Projection
    o0 = -((2 * focal) / w) * (ox / oz)
    o1 = -((2 * focal) / h) * (oy / oz)
    o2 = 1 + 2 * near / oz

    d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
    d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
    d2 = -2 * near / oz

    origins = torch.stack([o0, o1, o2], -1)
    directions = torch.stack([d0, d1, d2], -1)
    return origins, directions

class VolumeRenderer(nn.Module):
    """
    Volume renderer
    """
    def __init__(self, tree,
            step_size : float=1e-3,
            background_brightness : float=1.0,
            ndc : NDCConfig=None,
            min_comp : int=0,
            max_comp : int=-1,
            density_softplus : bool=False,
            rgb_padding : float=0.0,
        ):
        """
        Construct volume renderer associated with given N^3 tree.

        The renderer traces rays with origins/dirs within the octree boundaries,
        detection ray-voxel intersections. The color and density within
        each voxel is assumed constant, and no interpolation is performed.

        For each intersection point, it queries the tree, assuming the last data dimension
        is density (sigma) and the rest of the dimensions are color,
        formatted according to tree.data_format.
        It then applies SH/SG/ASG basis functions, if any, according to viewdirs.
        Sigmoid will be applied to these colors to normalize them,
        and optionally a shifted softplus is applied to the density.

        :param tree: N3Tree instance for rendering
        :param step_size: float step size eps, added to each voxel aabb intersection step
        :param background_brightness: float background brightness, 1.0 = white
        :param ndc: NDCConfig, NDC coordinate configuration,
                    namedtuple(width, height, focal).
                    None = no NDC, use usual coordinates
        :param min_comp: minimum SH/SG component to render.
        :param max_comp: maximum SH/SG component to render, -1=last.
                         Set :code:`min_comp = max_comp` to render a particular
                         component. Default means all.
        :param density_softplus: if true, applies :math:`\\log(1 + \\exp(sigma - 1))`.
                                 **Mind the shift -1!** (from mip-NeRF).
                                 Please note softplus will NOT be compatible with volrend,
                                 please pre-apply it .
        :param rgb_padding: to avoid oversaturating the sigmoid,
                        applies :code:`* (1 + 2 * rgb_padding) - rgb_padding` to
                        colors after sigmoid (from mip-NeRF).
                        Please note the padding will NOT be compatible with volrend,
                        although most likely the effect is very small.
                        0.001 is a reasonable value to try.

        """
        super().__init__()
        self.tree = tree
        self.step_size = step_size
        self.background_brightness = background_brightness
        self.ndc_config = ndc
        self.min_comp = min_comp
        self.max_comp = max_comp
        self.density_softplus = density_softplus
        self.rgb_padding = rgb_padding
        if isinstance(tree.data_format, DataFormat):
            self._data_format = None
        else:
            warn("Legacy N3Tree (pre 0.2.18) without data_format, auto-infering SH deg")
            # Auto SH deg
            ddim = tree.data_dim
            if ddim == 4:
                self._data_format = DataFormat("")
            else:
                self._data_format = DataFormat(f"SH{(ddim - 1) // 3}")
        self.tree._weight_accum = None

    def forward(self, rays : Rays, cuda=True, fast=False):
        """
        Render a batch of rays. Differentiable.

        :param rays: namedtuple Rays of origins (B, 3), dirs (B, 3), viewdirs (B, 3)
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version. *Only True supported right now*
        :param fast: if True, enables faster evaluation, potentially leading
                     to some loss of accuracy.

        :return: :code:`(B, rgb_dim)`.
                Where *rgb_dim* is :code:`tree.data_dim - 1` if
                :code:`data_format.format == DataFormat.RGBA`
                or :code:`(tree.data_dim - 1) / tree.data_format.basis_dim` else.
        """
        if not cuda or _C is None or not self.tree.data.is_cuda:
            assert False  # Not supported in current version, use CUDA kernel
            warn("Using slow volume rendering")
            def dda_unit(cen, invdir):
                """
                voxel aabb ray tracing step

                :param cen: jnp.ndarray [B, 3] center
                :param invdir: jnp.ndarray [B, 3] 1/dir

                :return: tmin jnp.ndarray [B] at least 0;
                         tmax jnp.ndarray [B]
                """
                B = invdir.shape[0]
                tmin = torch.zeros((B,), device=cen.device)
                tmax = torch.full((B,), fill_value=1e9, device=cen.device)
                for i in range(3):
                    t1 = -cen[..., i] * invdir[..., i]
                    t2 = t1 + invdir[..., i]
                    tmin = torch.max(tmin, torch.min(t1, t2))
                    tmax = torch.min(tmax, torch.max(t1, t2))
                return tmin, tmax

            origins, dirs, viewdirs = rays.origins, rays.dirs, rays.viewdirs
            origins = self.tree.world2tree(origins)
            B = dirs.size(0)
            assert viewdirs.size(0) == B and origins.size(0) == B
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)

            sh_mult = None
            if self.data_format.format != DataFormat.RGBA:
                sh_mult = maybe_eval_basis(self.data_format.basis_dim, viewdirs)[:, None]

            invdirs = 1.0 / (dirs + 1e-9)
            t, tmax = dda_unit(origins, invdirs)
            light_intensity = torch.ones(B, device=origins.device)
            out_rgb = torch.zeros((B, 3), device=origins.device)

            good_indices = torch.arange(B, device=origins.device)
            delta_scale = 1.0 / self.tree.invradius
            while good_indices.numel() > 0:
                pos = origins + t[:, None] * dirs
                treeview = self.tree[LocalIndex(pos)]
                rgba = treeview.values
                cube_sz = treeview.lengths_local
                pos_t = (pos - treeview.corners_local) / cube_sz[:, None]
                treeview = None

                subcube_tmin, subcube_tmax = dda_unit(pos_t, invdirs)

                delta_t = (subcube_tmax - subcube_tmin) * cube_sz + self.step_size
                att = torch.exp(- delta_t * torch.relu(rgba[..., -1]) * delta_scale)
                weight = light_intensity[good_indices] * (1.0 - att)
                rgb = rgba[:, :-1]
                if self.data_format.format != DataFormat.RGBA:
                    # [B', 3, n_sh_coeffs]
                    rgb_sh = rgb.reshape(-1, 3, self.data_format.basis_dim)
                    rgb = torch.sigmoid(torch.sum(sh_mult * rgb_sh, dim=-1))   # [B', 3]
                else:
                    rgb = torch.sigmoid(rgb)
                rgb = weight[:, None] * rgb[:, :3]

                out_rgb[good_indices] += rgb
                light_intensity[good_indices] *= att
                t += delta_t


                mask = t < tmax
                good_indices = good_indices[mask]
                origins = origins[mask]
                dirs = dirs[mask]
                invdirs = invdirs[mask]
                t = t[mask]
                if sh_mult is not None:
                    sh_mult = sh_mult[mask]
                tmax = tmax[mask]
            out_rgb += self.background_brightness * light_intensity[:, None]
            return out_rgb
        else:
            return _VolumeRenderFunction.apply(
                self.tree.data,
                self.tree._spec(),
                _rays_spec_from_rays(rays),
                self._get_options(fast)
            )

    def render_persp(self, c2w, width=800, height=800, fx=1111.111, fy=None,
            cuda=True, fast=False):
        """
        Render a perspective image. Differentiable.

        :param c2w: torch.Tensor (3, 4) or (4, 4) camera pose matrix (c2w)
        :param width: int output image width
        :param height: int output image height
        :param fx: float output image focal length (x)
        :param fy: float output image focal length (y), if not specified uses fx
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version. *Only True supported right now*
        :param fast: if True, enables faster evaluation, potentially leading
                     to some loss of accuracy.

        :return: :code:`(height, width, rgb_dim)`
                where *rgb_dim* is :code:`tree.data_dim - 1` if
                :code:`data_format.format == DataFormat.RGBA`
                or :code:`(tree.data_dim - 1) / tree.data_format.basis_dim` else.

        """
        if fy is None:
            fy = fx

        if not cuda or _C is None or not self.tree.data.is_cuda:
            assert False  # Not supported in current version, use CUDA kernel
            origins = c2w[None, :3, 3].expand(height * width, -1)
            yy, xx = torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=c2w.device),
                torch.arange(width, dtype=torch.float32, device=c2w.device),
            )
            xx = (xx - width * 0.5) / float(fx)
            yy = (yy - height * 0.5) / float(fy)
            zz = torch.ones_like(xx)
            dirs = torch.stack((xx, -yy, -zz), dim=-1)
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)
            dirs = dirs.reshape(-1, 3)
            del xx, yy, zz
            dirs = torch.matmul(c2w[None, :3, :3], dirs[..., None])[..., 0]
            vdirs = dirs
            if self.ndc_config is not None:
                origins, dirs = convert_to_ndc(origins, dirs, self.ndc_config.focal,
                        self.ndc_config.width, self.ndc_config.height)
            rays = {
                'origins': origins,
                'dirs': dirs,
                'viewdirs': vdirs
            }
            rgb = self(rays, cuda=False, fast=fast)
            return rgb.reshape(height, width, -1)
        else:
            return _VolumeRenderImageFunction.apply(
                self.tree.data,
                self.tree._spec(),
                _make_camera_spec(c2w.to(dtype=self.tree.data.dtype),
                                  width, height, fx, fy),
                self._get_options(fast)
            )

    @property
    def data_format(self):
        return self._data_format or self.tree.data_format

    def _get_options(self, fast=False):
        """
        Make RenderOptions struct to send to C++
        """
        opts = _C.RenderOptions()
        opts.step_size = self.step_size
        opts.background_brightness = self.background_brightness

        opts.format = self.data_format.format
        opts.basis_dim = self.data_format.basis_dim
        opts.min_comp = self.min_comp
        opts.max_comp = self.max_comp

        if self.max_comp < 0:
            opts.max_comp += opts.basis_dim

        opts.density_softplus = self.density_softplus
        opts.rgb_padding = self.rgb_padding

        if self.ndc_config is not None:
            opts.ndc_width = self.ndc_config.width
            opts.ndc_height = self.ndc_config.height
            opts.ndc_focal = self.ndc_config.focal
        else:
            opts.ndc_width = -1

        if fast:
            opts.sigma_thresh = 1e-2
            opts.stop_thresh = 1e-2
        else:
            opts.sigma_thresh = 0.0
            opts.stop_thresh = 0.0
        # Override
        if hasattr(self, "sigma_thresh"):
            opts.sigma_thresh = self.sigma_thresh
        if hasattr(self, "stop_thresh"):
            opts.stop_thresh = self.stop_thresh
        return opts

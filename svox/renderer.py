#  [BSD 2-CLAUSE LICENSE]
#
#  Copyright SVOX Authors 2021
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

from svox.helpers import _get_c_extension, LocalIndex

class DataFormat:
    RGBA = 0
    SH = 1
    SG = 2
    ASG = 3
    def __init__(self, txt):
        nonalph_idx = [c.isalpha() for c in txt]
        if False in nonalph_idx:
            nonalph_idx = nonalph_idx.index(False)
            self.basis_dim = int(txt[nonalph_idx:])
            format_type = txt[:nonalph_idx]
            if format_type == "SH":
                self.format = DataFormat.SH
            elif format_type == "SG":
                self.format = DataFormat.SG
            elif format_type == "ASG":
                self.format = DataFormat.ASG
            else:
                self.format = DataFormat.RGBA
        else:
            self.format = DataFormat.RGBA
            self.basis_dim = -1

    def __repr__(self):
        if self.format == DataFormat.SH:
            r = "SH"
        elif self.format == DataFormat.SG:
            r = "SG"
        elif self.format == DataFormat.ASG:
            r = "ASG"
        else:
            r = "RGBA"
        if self.basis_dim >= 0:
            r += str(self.basis_dim)
        return r

_C = _get_c_extension()

class _VolumeRenderFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, child, extra_data,
            origins, dirs, viewdirs, offset, invradius, weight_accum, opts):
        if weight_accum is None:
            weight_accum = torch.empty(0, device=data.device)
        if extra_data is None:
            extra_data = torch.empty((0, 0), device=data.device)
        out = _C.volume_render(
            data,
            child,
            extra_data,
            origins,
            dirs,
            viewdirs,
            offset,
            invradius,
            opts["step_size"],
            opts["background_brightness"],
            opts["format"],
            opts["basis_dim"],
            opts["fast"],
            weight_accum,
        )
        ctx.save_for_backward(data, child, extra_data, origins, dirs,
                viewdirs, offset, invradius)
        ctx.opts = opts
        return out

    @staticmethod
    def backward(ctx, grad_out):
        data, child, extra_data, origins, dirs, viewdirs, offset, invradius = \
                ctx.saved_tensors
        opts = ctx.opts

        grad_out = grad_out.contiguous()
        if ctx.needs_input_grad[0]:
            grad_data = _C.volume_render_backward(
                data,
                child,
                extra_data,
                grad_out,
                origins,
                dirs,
                viewdirs,
                offset,
                invradius,
                opts["step_size"],
                opts["background_brightness"],
                opts["format"],
                opts["basis_dim"],
            )
        else:
            grad_data = None

        return grad_data, None, None, None, None, None, None, None, None, None

class _VolumeRenderImageFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, child, extra_data,
                offset, invradius, c2w, weight_accum, opts):
        if weight_accum is None:
            weight_accum = torch.empty(0, device=data.device)
        if extra_data is None:
            extra_data = torch.empty((0, 0), device=data.device)
        out = _C.volume_render_image(
            data,
            child,
            extra_data,
            offset,
            invradius,
            c2w,
            opts["fx"],
            opts["fy"],
            opts["width"],
            opts["height"],
            opts["step_size"],
            opts["background_brightness"],
            opts["format"],
            opts["basis_dim"],
            opts.get("ndc_width", 0),
            opts.get("ndc_height", 0),
            opts.get("ndc_focal", 0.0),
            opts["fast"],
            weight_accum,
        )
        ctx.save_for_backward(data, child, extra_data, offset, invradius, c2w)
        ctx.opts = opts
        return out

    @staticmethod
    def backward(ctx, grad_out):
        data, child, extra_data, offset, invradius, c2w = ctx.saved_tensors
        opts = ctx.opts

        grad_out = grad_out.contiguous()
        if ctx.needs_input_grad[0]:
            grad_data = _C.volume_render_image_backward(
                data,
                child,
                extra_data,
                grad_out,
                offset,
                invradius,
                c2w,
                opts["fx"],
                opts["fy"],
                opts["width"],
                opts["height"],
                opts["step_size"],
                opts["background_brightness"],
                opts["format"],
                opts["basis_dim"],
                opts.get("ndc_width", 0),
                opts.get("ndc_height", 0),
                opts.get("ndc_focal", 0.0),
            )
        else:
            grad_data = None

        return grad_data, None, None, None, None, None, None, None


def convert_to_ndc(origins, directions, focal, w, h, near=1.0):
    """Convert a set of rays to NDC coordinates."""
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

NDCConfig = namedtuple('NDCConfig', ["width", "height", "focal"])
Rays = namedtuple('Rays', ["origins", "dirs", "viewdirs"])

class VolumeRenderer(nn.Module):
    """
    Volume renderer
    """
    def __init__(self, tree,
            step_size : float=1e-3,
            background_brightness : float=1.0,
            ndc : NDCConfig=None,
        ):
        """
        Construct volume renderer associated with given N^3 tree.

        :param tree: N3Tree instance for rendering
        :param step_size: float step size eps, added to each DDA step
        :param background_brightness: float background brightness, 1.0 = white
        :param ndc: NDCConfig, NDC coordinate configuration,
                    namedtuple(width, height, focal).
                    None = no NDC, use usual coordinates

        """
        super().__init__()
        self.tree = tree
        self.step_size = step_size
        self.background_brightness = background_brightness
        self.ndc_config = ndc
        if tree.data_format is not None:
            self.data_format = DataFormat(tree.data_format)
        else:
            warn("Legacy N3Tree (pre 0.2.18) without data_format, auto-infering SH order")
            # Auto SH order
            ddim = tree.data_dim
            if ddim == 4:
                self.data_format = DataFormat("")
            else:
                self.data_format = DataFormat(f"SH{(ddim - 1) // 3}")
        self.tree._weight_accum = None

    def forward(self, rays : Rays, cuda=True, fast=False):
        """
        Render a batch of rays. Differentiable.

        :param rays: namedtuple Rays of origins (B, 3), dirs (B, 3), viewdirs (B, 3)
        :param rgba: (B, rgb_dim + 1)
                where *rgb_dim* is :code:`tree.data_dim - 1` if
                :code:`data_format.format == RGBA`
                or :code:`data_format.basis_dim` else
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version.
        :param fast: if True, enables faster evaluation, potentially leading
                     to some loss of accuracy.

        :return: (B, 3)

        """
        if not cuda or _C is None or not self.tree.data.is_cuda:
            assert False  # Not supported in current version, use CUDA kernel
            warn("Using slow volume rendering")
            def dda_unit(cen, invdir):
                """
                DDA ray tracing step

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
            opts = {
                'step_size': self.step_size,
                'background_brightness':self.background_brightness,
                'format': self.data_format.format,
                'basis_dim': self.data_format.basis_dim,
                'fast': fast
            }
            return _VolumeRenderFunction.apply(
                self.tree.data,
                self.tree.child,
                self.tree.extra_data,
                rays.origins,
                rays.dirs,
                rays.viewdirs,
                self.tree.offset,
                self.tree.invradius,
                self.tree._weight_accum,
                opts
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
                     uses only PyTorch version.
        :param fast: if True, enables faster evaluation, potentially leading
                     to some loss of accuracy.

        :return: (height, width, rgb_dim + 1)
                where *rgb_dim* is :code:`tree.data_dim - 1` if
                :code:`data_format.format == DataFormat.RGBA`
                or :code:`data_format.basis_dim` else

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
            opts = {
                'fx': fx,
                'fy': fy,
                'width': width,
                'height': height,
                'step_size': self.step_size,
                'background_brightness':self.background_brightness,
                'format': self.data_format.format,
                'basis_dim': self.data_format.basis_dim,
                'fast': fast
            }
            if self.ndc_config is not None:
                opts.update({
                    'ndc_width': self.ndc_config.width,
                    'ndc_height': self.ndc_config.height,
                    'ndc_focal': self.ndc_config.focal,
                })
            return _VolumeRenderImageFunction.apply(
                self.tree.data,
                self.tree.child,
                self.tree.extra_data,
                self.tree.offset,
                self.tree.invradius,
                c2w,
                self.tree._weight_accum,
                opts
            )

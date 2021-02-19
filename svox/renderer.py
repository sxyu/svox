#  [BSD 2-CLAUSE LICENSE]
#
#  Copyright Alex Yu 2021
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
from dataclasses import dataclass
from warnings import warn

from svox.helpers import _get_c_extension, LocalIndex
from svox import sh

_C = _get_c_extension()

class _VolumeRenderFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, child,
            origins, dirs, viewdirs, offset, invradius, opts):
        out = _C.volume_render(
            data,
            child,
            origins,
            dirs,
            viewdirs,
            offset,
            invradius,
            opts["step_size"],
            opts["background_brightness"],
            opts["sh_order"],
            opts["fast"]
        )
        ctx.save_for_backward(data, child, origins, dirs,
                viewdirs, offset, invradius)
        ctx.opts = opts
        return out

    @staticmethod
    def backward(ctx, grad_out):
        data, child, origins, dirs, viewdirs, offset, invradius = \
                ctx.saved_tensors
        opts = ctx.opts

        grad_out = grad_out.contiguous()
        if ctx.needs_input_grad[0]:
            grad_data = _C.volume_render_backward(
                data,
                child,
                grad_out,
                origins,
                dirs,
                viewdirs,
                offset,
                invradius,
                opts["step_size"],
                opts["background_brightness"],
                opts["sh_order"],
            )
        else:
            grad_data = None

        return grad_data, None, None, None, None, None, None, None

class _VolumeRenderImageFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, child, offset, invradius, c2w, opts):
        out = _C.volume_render_image(
            data,
            child,
            offset,
            invradius,
            c2w,
            opts["fx"],
            opts["fy"],
            opts["width"],
            opts["height"],
            opts["step_size"],
            opts["background_brightness"],
            opts["sh_order"],
            opts["fast"]
        )
        ctx.save_for_backward(data, child, offset, invradius, c2w)
        ctx.opts = opts
        return out

    @staticmethod
    def backward(ctx, grad_out):
        data, child, offset, invradius, c2w = ctx.saved_tensors
        opts = ctx.opts

        grad_out = grad_out.contiguous()
        if ctx.needs_input_grad[0]:
            grad_data = _C.volume_render_image_backward(
                data,
                child,
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
                opts["sh_order"]
            )
        else:
            grad_data = None

        return grad_data, None, None, None, None, None


class VolumeRenderer(nn.Module):
    """
    Volume renderer
    """
    def __init__(self, tree, step_size=1e-3,
            background_brightness=1.0,
            sh_order=None):
        """
        Construct volume renderer associated with given N^3 tree.

        :param tree: N3Tree instance for rendering
        :param step_size: float step size eps, added to each DDA step
        :param background_brightness: float background brightness, 1.0 = white
        :param sh_order: SH order, -1 = disable, None = auto determine

        """
        super().__init__()
        self.tree = tree
        self.step_size = step_size
        self.background_brightness = background_brightness
        if sh_order is None:
            # Auto SH order
            ddim = tree.data_dim
            if ddim == 4 * 3 + 1:
                self.sh_order = 1
            elif ddim == 9 * 3 + 1:
                self.sh_order = 2
            elif ddim == 16 * 3 + 1:
                self.sh_order = 3
            elif ddim == 25 * 3 + 1:
                self.sh_order = 4
            else:
                self.sh_order = -1
        else:
            self.sh_order = sh_order

    def forward(self, rays, cuda=True, fast=False):
        """
        Render a batch of rays. Differentiable (experimental).

        :param rays: dict[string, torch.Tensor] of origins (B, 3), dirs (B, 3), viewdirs (B, 3)
        :param rgba: (B, rgb_dim + 1)
                where *rgb_dim* is :code:`tree.data_dim - 1` if
                :code:`sh_order == -1`
                or :code:`(tree.data_dim - 1) / (sh_order + 1)^2` else
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version.
        :param fast: if True, enables faster evaluation, potentially leading
                     to some loss of accuracy.

        :return: (B, 3)

        """
        if not cuda or _C is None or not self.tree.data.is_cuda:
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

            origins, dirs, viewdirs = rays["origins"], rays["dirs"], rays["viewdirs"]
            origins = self.tree.world2tree(origins)
            B = dirs.size(0)
            assert viewdirs.size(0) == B and origins.size(0) == B
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)

            sh_mult = None
            if self.sh_order >= 0:
                sh_mult = sh.eval_sh_bases(self.sh_order, viewdirs)[:, None]

            invdirs = 1.0 / (dirs + 1e-9)
            t, tmax = dda_unit(origins, invdirs)
            light_intensity = torch.ones(B, device=origins.device)
            out_rgb = torch.zeros((B, 3), device=origins.device)

            good_indices = torch.arange(B, device=origins.device)
            delta_scale = 1.0 / self.tree.invradius
            while good_indices.numel() > 0:
                pos = origins + t[:, None] * dirs
                treeview = self.tree[LocalIndex(pos)]
                rgba = treeview.values_diff
                cube_sz = treeview.lengths_local
                pos_t = (pos - treeview.corners_local) / cube_sz[:, None]
                treeview = None

                subcube_tmin, subcube_tmax = dda_unit(pos_t, invdirs)

                delta_t = (subcube_tmax - subcube_tmin) * cube_sz + self.step_size
                att = torch.exp(- delta_t * torch.relu(rgba[..., -1]) * delta_scale)
                weight = light_intensity[good_indices] * (1.0 - att)
                rgb = rgba[:, :-1]
                if self.sh_order >= 0:
                    rgb_sh = rgb.reshape(-1, 3, (self.sh_order + 1) ** 2)  # [B', 3, n_sh_coeffs]
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
                'sh_order': self.sh_order,
                'fast': fast
            }
            return _VolumeRenderFunction.apply(
                self.tree.data,
                self.tree.child,
                rays["origins"],
                rays["dirs"],
                rays["viewdirs"],
                self.tree.offset,
                self.tree.invradius,
                opts
            )

    def render_persp(self, c2w, width=800, height=800, fx=1111.111, fy=None,
            cuda=True, fast=False):
        """
        Render a perspective image. Differentiable (experimental).

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
                :code:`sh_order == -1`
                or :code:`(tree.data_dim - 1) / (sh_order + 1)^2` else

        """
        if fy is None:
            fy = fx

        if not cuda or _C is None or not self.tree.data.is_cuda:
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
            rays = {
                'origins': origins,
                'dirs': dirs,
                'viewdirs': dirs
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
                'sh_order': self.sh_order,
                'fast': fast
            }
            return _VolumeRenderImageFunction.apply(
                self.tree.data,
                self.tree.child,
                self.tree.offset,
                self.tree.invradius,
                c2w,
                opts
            )

import torch
import numpy as np
from torch import nn, autograd
from dataclasses import dataclass

from svox.helpers import get_c_extension

_C = get_c_extension()

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

        return grad_data, *((None,) * 7)

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

        return grad_data, *((None,) * 5)


class VolumeRenderer(nn.Module):
    """
    Volume renderer
    """
    def __init__(self, tree, step_size=1e-3,
            background_brightness=1.0,
            sh_order=None):
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

    def forward(self, rays, fast=False):
        """
        Render a batch of rays
        Args:
            rays: dict[string, torch.Tensor] of origins [B, 3], dirs [B, 3], viewdirs [B, 3]
        Returns:
            rgba: [B, rgb_dim + 1] where rgb_dim is tree.data_dim - 1 if sh_order == -1
                                   or (tree.data_dim - 1) // (sh_order + 1) ** 2 else
        """
        assert _C is not None  # Pure PyTorch version not implemented
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

    def render_persp(self, c2w, width=800, height=800, fx=1111.111, fy=None, fast=False):
        """
        Render a perspective image
        Args:
            c2w: torch.Tensor [3, 4] or [4, 4] camera pose matrix (c2w)
            width: output image width
            height: output image height
            fx: output image focal length (x)
            fy: output image focal length (y), if not specified uses fx
        Returns:
            rgba: [height, width, rgb_dim + 1]
                                   where rgb_dim is tree.data_dim - 1 if sh_order == -1
                                   or (tree.data_dim - 1) // (sh_order + 1) ** 2 else
        """
        if fy is None:
            fy = fx

        assert _C is not None  # Pure PyTorch version not implemented
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

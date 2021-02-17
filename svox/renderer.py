import torch
import numpy as np
from torch import nn
from dataclasses import dataclass

from svox.helpers import get_c_extension

_C = get_c_extension()

@dataclass
class Ray:
    origins: torch.Tensor
    dirs: torch.Tensor
    viewdirs: torch.Tensor

class VolumeRenderer(nn.Module):
    """
    Volume renderer
    """
    def __init__(self, tree, step_size=1e-3,
            stop_thresh=0.0, background_brightness=1.0,
            sh_order=None):
        super().__init__()
        self.tree = tree
        self.step_size = step_size
        self.stop_thresh = stop_thresh
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

    def forward(self, rays):
        """
        Args:
            rays: dict[string, torch.Tensor] of origins [B, 3], dirs [B, 3], viewdirs [B, 3]
        Returns:
            rgba: [B, rgb_dim + 1] where rgb_dim is tree.data_dim - 1 if sh_order == -1
                                   or (tree.data_dim - 1) // (sh_order + 1) ** 2 else
        """
        assert _C is not None  # Pure PyTorch version not implemented
        return _C.volume_render(
            self.tree.data,
            self.tree.child,
            rays["origins"],
            rays["dirs"],
            rays["viewdirs"],
            self.tree.offset,
            self.tree.invradius,
            self.step_size,
            self.stop_thresh,
            self.background_brightness,
            self.sh_order
        )

    def render_persp(self, c2w, width=800, height=800, fx=1111.111, fy=None):
        """
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
        return _C.volume_render_image(
            self.tree.data,
            self.tree.child,
            self.tree.offset,
            self.tree.invradius,
            c2w,
            fx,
            fy,
            width,
            height,
            self.step_size,
            self.stop_thresh,
            self.background_brightness,
            self.sh_order
        )

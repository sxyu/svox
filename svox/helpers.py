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
import torch
import numpy as np

class N3TreeView:
    def __init__(self, tree, key):
        self.tree = tree
        local = False
        self.single_key = False
        if isinstance(key, LocalIndex):
            key = key.val
            local = True
        if isinstance(key, tuple) and len(key) >= 3:
            # Handle tree[x, y, z[, c]]
            main_key = torch.tensor(key[:3], dtype=tree.data.dtype,
                        device=tree.data.device).reshape(1, 3)
            if len(key) > 3:
                key = (main_key, *key[3:])
            else:
                key = main_key
        leaf_key = key[0] if isinstance(key, tuple) else key
        if torch.is_tensor(leaf_key) and leaf_key.ndim == 2 and leaf_key.shape[1] == 3:
            # Handle tree[P[, c]] where P is a (B, 3) matrix of 3D points
            if leaf_key.dtype != tree.data.dtype:
                leaf_key = leaf_key.to(dtype=tree.data.dtype)
            val, target = tree.forward(leaf_key, want_node_ids=True, world=not local)
            self._packed_ids = target.clone()
            leaf_node = (*tree._unpack_index(target).T,)
        else:
            self._packed_ids = None
            if isinstance(leaf_key, int):
                leaf_key = torch.tensor([leaf_key], device=tree.data.device)
                self.single_key = True
            leaf_node = self.tree._all_leaves()
            leaf_node = leaf_node.__getitem__(leaf_key).T
        if isinstance(key, tuple):
            self.key = (*leaf_node, *key[1:])
        else:
            self.key = (*leaf_node,)
        self._value = None
        self._tree_ver = tree._ver

    def __repr__(self):
        self._check_ver()
        return "N3TreeView(" + repr(self.values) + ")"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self._check_ver()
        if kwargs is None:
            kwargs = {}
        new_args = []
        for arg in args:
            if isinstance(arg, N3TreeView):
                new_args.append(arg.values)
            else:
                new_args.append(arg)
        return func(*new_args, **kwargs)

    def set(self, value):
        self._check_ver()
        if isinstance(value, N3TreeView):
            value = value.values_nograd
        if self._require_subindex():
            tmp = self.tree.data.data[self.key[:-1]]
            tmp[..., self.key[-1]] = value
            self.tree.data.data[self.key[:-1]] = tmp
        else:
            self.tree.data.data[self.key] = value

    def refine(self, repeats=1):
        """
        Refine selected leaves using tree.refine
        """
        self._check_ver()
        ret = self.tree.refine(repeats, sel=self._unique_node_key())
        return ret

    @property
    def values(self):
        """
        Values of the selected leaves (autograd enabled)

        :return: (n_leaves, data_dim) float32 note this is 2D even if key is int
        """
        self._check_ver()
        if self._require_subindex():
            ret = self.tree.data[self.key[:-1]][..., self.key[-1]]
        else:
            ret = self.tree.data[self.key]
        return ret[0] if self.single_key else ret

    @property
    def values_nograd(self):
        """
        Values of the selected leaves (no autograd)

        :return: (n_leaves, data_dim) float32 note this is 2D even if key is int
        """
        self._check_ver()
        if self._require_subindex():
            ret = self.tree.data.data[self.key[:-1]][..., self.key[-1]]
        else:
            ret = self.tree.data[self.key]
        return ret[0] if self.single_key else ret

    @property
    def shape(self):
        self._check_ver()
        return self.values_nograd.shape

    @property
    def ndim(self):
        return 2

    @property
    def depths(self):
        """
        Get a list of selected leaf depths in tree,
        in same order as values, corners.
        Root is at depth -1. Any children of
        root will have depth 0, etc.

        :return: (n_leaves) int32
        """
        self._check_ver()
        return self.tree.parent_depth[self.key[0], 1]

    @property
    def lengths(self):
        """
        Get a list of selected leaf side lengths in tree (world dimensions),
        in same order as values, corners, depths

        :return: (n_leaves, 3) float
        """
        self._check_ver()
        return (self.tree.N ** (-self.depths.float() - 1.0))[:, None] / self.tree.invradius

    @property
    def lengths_local(self):
        """
        Get a list of selected leaf side lengths in tree
        (local index :math:`[0, 1]^3`),
        in same order as values, corners, depths

        :return: (n_leaves) float
        """
        self._check_ver()
        return self.tree.N ** (-self.depths.float() - 1.0)

    @property
    def corners(self):
        """
        Get a list of selected leaf lower corners in tree
        (world coordinates),
        in same order as values, lengths, depths

        :return: (n_leaves, 3) float
        """
        self._check_ver()
        return (self.tree._calc_corners(self._indexer())
                - self.tree.offset) / self.tree.invradius

    @property
    def corners_local(self):
        """
        Get a list of selected leaf lower corners in tree
        (local index :math:`[0, 1]^3`),
        in same order as values, lengths, depths

        :return: (n_leaves, 3) float
        """
        self._check_ver()
        return self.tree._calc_corners(self._indexer())

    def sample(self, n_samples, device=None):
        """
        Sample n_samples uniform points in each selected leaf (world coordinates)

        :param n_samples: samples for each leaf
        :param device: device to output random samples in

        :return: (n_leaves, n_samples, 3) float
        """
        self._check_ver()
        if device is None:
            device = self.tree.data.device
        corn = self.corners.to(device=device)
        length = self.lengths.to(device=device)
        if length.ndim == 1:
            length = length[:, None]
        u = torch.rand((corn.shape[0], n_samples, 3),
                device=device,
                dtype=length.dtype) * length[:, None]
        return corn[:, None] + u

    def sample_local(self, n_samples):
        """
        Sample n_samples uniform points in each selected leaf
        (local index :math:`[0, 1]^3`)

        :return: (n_leaves, n_samples, 3) float
        """
        self._check_ver()
        corn = self.corners_local
        length = self.lengths_local
        u = torch.rand((corn.shape[0], n_samples, 3),
                device=length.device,
                dtype=length.dtype) * length[:, None, None]
        return corn[:, None] + u

    def aux(self, arr):
        """
        Index an auxiliary tree data array of size (capacity, N, N, N, Any)
        using this view
        """
        if self._require_subindex():
            return arr[self.key[:-1]][..., self.key[-1]]
        else:
            return arr[self.key]

    # In-place modification helpers
    def normal_(self, mean=0.0, std=1.0):
        """
        Set all values to random normal
        FIXME: inefficient

        :param mean: normal mean
        :param std: normal std

        """
        self._check_ver()
        self.set(torch.randn_like(self.values_nograd) * std + mean)

    def uniform_(self, min=0.0, max=1.0):
        """
        Set all values to random uniform
        FIXME: inefficient

        :param min: interval min
        :param max: interval max

        """
        self._check_ver()
        self.set(torch.rand_like(self.values_nograd) * (max - min) + min)

    def clamp_(self, min=None, max=None):
        """
        Clamp.
        FIXME: inefficient

        :param min: clamp min value, None=disable
        :param max: clamp max value, None=disable

        """
        self._check_ver()
        self.set(torch.clamp(self.values_nograd, min, max))

    def relu_(self):
        """
        Apply relu to all elements.
        FIXME: inefficient
        """
        self._check_ver()
        self.set(torch.relu(self.values_nograd))

    def sigmoid_(self):
        """
        Apply sigmoid to all elements.
        FIXME: inefficient
        """
        self._check_ver()
        self.set(torch.sigmoid(self.values_nograd))

    def nan_to_num_(self, inf_val=2e4):
        """
        Convert nans to 0.0 and infs to inf_val
        FIXME: inefficient
        """
        data = self.tree.data.data[self.key]
        data[torch.isnan(data)] = 0.0
        inf_mask = torch.isinf(data)
        data[inf_mask & (data > 0)] = inf_val
        data[inf_mask & (data < 0)] = -inf_val
        self.tree.data.data[self.key] = data

    def __setitem__(self, key, value):
        """
        FIXME: inefficient
        """
        val = self.values_nograd
        val.__setitem__(key, value)
        self.set(val)

    def _indexer(self):
        return torch.stack(self.key[:4], dim=-1)

    def _require_subindex(self):
        return isinstance(self.key, tuple) and len(self.key) == 5 and \
            not isinstance(self.key[-1], slice) and not isinstance(self.key[-1], int)

    def _unique_node_key(self):
        if self._packed_ids is None:
            return self.key[:4]

        uniq_ids = torch.unique(self._packed_ids)
        return (*self.tree._unpack_index(uniq_ids).T,)

    def _check_ver(self):
        if self.tree._ver > self._tree_ver:
            self.key = self._packed_ids = None
            raise RuntimeError("N3TreeView has been invalidated because tree " +
                    "data layout has changed")

# Redirect functions to Tensor
def _redirect_funcs():
    redir_funcs = ['__floordiv__', '__mod__', '__div__',
                   '__eq__', '__ne__', '__ge__', '__gt__', '__le__',
                   '__lt__', '__floor__', '__ceil__', '__round__', '__len__',
                   'item', 'size', 'dim', 'numel']
    redir_grad_funcs = ['__add__', '__mul__', '__sub__',
                   '__mod__', '__div__', '__truediv__',
                   '__radd__', '__rsub__', '__rmul__',
                   '__rdiv__', '__abs__', '__pos__', '__neg__',
                   '__len__', 'clamp', 'clamp_max', 'clamp_min', 'relu', 'sigmoid',
                   'max', 'min', 'mean', 'sum', '__getitem__']
    def redirect_func(redir_func, grad=False):
        def redir_impl(self, *args, **kwargs):
            return getattr(self.values if grad else self.values_nograd, redir_func)(
                    *args, **kwargs)
        setattr(N3TreeView, redir_func, redir_impl)
    for redir_func in redir_funcs:
        redirect_func(redir_func)
    for redir_func in redir_grad_funcs:
        redirect_func(redir_func, grad=True)
_redirect_funcs()


def _get_c_extension():
    from warnings import warn
    try:
        import svox.csrc as _C
        if not hasattr(_C, "query_vertical"):
            _C = None
    except:
        _C = None

    if _C is None:
        warn("CUDA extension svox.csrc could not be loaded! " +
             "Operations will be slow.\n" +
             "Please do not import svox in the SVOX source directory.")
    return _C

class LocalIndex:
    """
    To query N3Tree using 'local' index :math:`[0,1]^3`,
    tree[LocalIndex(points)] where points (N, 3)
    """
    def __init__(self, val):
        self.val = val

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
            assert self.basis_dim > 0, "data_format basis_dim must be positive"
            self.data_dim = 3 * self.basis_dim + 1
            if format_type == "SH":
                self.format = DataFormat.SH
                assert int(self.basis_dim ** 0.5) ** 2 == self.basis_dim, \
                       "SH basis dim must be square number"
                assert self.basis_dim <= 25, "SH only supported up to basis_dim 25"
            elif format_type == "SG":
                self.format = DataFormat.SG
            elif format_type == "ASG":
                self.format = DataFormat.ASG
            else:
                self.format = DataFormat.RGBA
        else:
            self.format = DataFormat.RGBA
            self.basis_dim = -1
            self.data_dim = None

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

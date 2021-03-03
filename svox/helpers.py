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
import torch
import numpy as np

class N3TreeView:
    def __init__(self, tree, key):
        self.tree = tree
        self._points = None
        local = False
        if isinstance(key, LocalIndex):
            key = key.val
            local = True
        if isinstance(key, tuple) and len(key) >= 3:
            # Handle tree[x, y, z[, c]]
            main_key = torch.tensor(key[:3], dtype=torch.float32,
                        device=tree.data.device).reshape(1, 3)
            key = (main_key, *key[3:]) if len(key) > 3 else main_key

        has_data_dim_index = isinstance(key, tuple)
        leaf_key = key[0] if has_data_dim_index else key
        if torch.is_tensor(leaf_key) and leaf_key.ndim == 2 and leaf_key.shape[1] == 3:
            # Handle tree[P[, c]] where P is a (B, 3) matrix of 3D points
            if leaf_key.dtype != torch.float32:
                leaf_key = leaf_key.float()
            self._points = leaf_key
            val, leaf_ids = tree.forward(leaf_key, want_data_ids=True, world=not local)
            leaf_key = leaf_ids.long()
        else:
            # Advance any indices by 1 to exclude 'null' element
            if isinstance(leaf_key, int):
                if leaf_key <= -tree.data.size(0):
                    raise IndexError()
                leaf_key = torch.tensor(leaf_key + 1 if leaf_key >= 0 else leaf_key,
                                        device=tree.data.device)
            elif isinstance(leaf_key, torch.Tensor):
                if leaf_key.dtype == torch.long:
                    leaf_key += leaf_key >= 0
                elif leaf_key.dtype == torch.bool:
                    leaf_key = torch.cat(torch.zeros(1,
                        dtype=torch.bool, device=tree.data.device), leaf_key)
            elif isinstance(leaf_key, slice):
                start = leaf_key.start + 1 if leaf_key.start is not None and \
                                              leaf_key.start >= 0 else 1
                stop = leaf_key.stop + 1 if leaf_key.stop is not None and \
                                            leaf_key.stop >= 0 else None
                leaf_key = slice(start, stop, leaf_key.step)
        self.key = (leaf_key, *key[1:]) if has_data_dim_index else leaf_key
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
        if isinstance(value, N3TreeView):
            value = value.values_nograd
        self._maybe_lazy_alloc()
        self._check_ver()
        self.tree.data.data[self.key] = value

    def refine(self, repeats=1, lazy=False):
        """
        Refine selected leaves using tree.refine
        """
        self._check_ver()
        ret = self.tree.refine(repeats, sel=self._unique_node_key(), lazy=lazy)
        return ret

    def alloc(self):
        """
        Force allocation, if applicable (only if tree.use_lazy and
        the key consists of points)
        """
        self._maybe_lazy_alloc()

    @property
    def values(self):
        """
        Values of the selected leaves (autograd enabled)

        :return: (n_leaves, data_dim) float32 note this is 2D even if key is int
        """
        self._check_ver()
        return self.tree.data[self.key]

    @property
    def values_nograd(self):
        """
        Values of the selected leaves (no autograd)

        :return: (n_leaves, data_dim) float32 note this is 2D even if key is int
        """
        self._check_ver()
        return self.tree.data.data[self.key]

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
        return self.tree.idepths[self._indexer()[:, 0]]

    @property
    def lengths(self):
        """
        Get a list of selected leaf side lengths in tree (world dimensions),
        in same order as values, corners, depths

        :return: (n_leaves, 3) float
        """
        self._check_ver()
        return (2.0 ** (-self.depths.float() - 1.0))[:, None] / self.tree.scaling

    @property
    def lengths_local(self):
        """
        Get a list of selected leaf side lengths in tree
        (local index :math:`[0, 1]^3`),
        in same order as values, corners, depths

        :return: (n_leaves) float
        """
        self._check_ver()
        return 2.0 ** (-self.depths.float() - 1.0)

    @property
    def corners(self):
        """
        Get a list of selected leaf lower corners in tree
        (world coordinates),
        in same order as values, lengths, depths

        :return: (n_leaves, 3) float
        """
        return (self.tree._calc_corners(self._indexer())
                - self.tree.offset) / self.tree.scaling

    @property
    def corners_local(self):
        """
        Get a list of selected leaf lower corners in tree
        (local index :math:`[0, 1]^3`),
        in same order as values, lengths, depths

        :return: (n_leaves, 3) float
        """
        return self.tree._calc_corners(self._indexer())

    def sample(self, n_samples):
        """
        Sample n_samples uniform points in each selected leaf (world coordinates)

        :return: (n_leaves, n_samples, 3) float
        """
        corn = self.corners
        length = self.lengths
        if length.ndim == 1:
            length = length[:, None]
        u = torch.rand((corn.shape[0], n_samples, 3),
                device=length.device,
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
        Index an auxiliary tree data array of size (capacity, N, N, N, *)
        using this view
        """
        return arr[self.key]

    # In-place modification helpers
    def normal_(self, mean=0.0, std=1.0):
        """
        Set all values to random normal

        :param mean: normal mean
        :param std: normal std

        """
        self._check_ver()
        self.tree.data.data[self.key] = torch.randn_like(
                self.tree.data.data[self.key]) * std + mean

    def uniform_(self, min=0.0, max=1.0):
        """
        Set all values to random uniform

        :param min: interval min
        :param max: interval max

        """
        self._check_ver()
        self.tree.data.data[self.key] = torch.rand_like(
                self.tree.data.data[self.key]) * (max - min) + min

    def __setitem__(self, key, value):
        """
        Warning: inefficient impl
        """
        self._maybe_lazy_alloc()
        self._check_ver()
        self.values_nograd.__setitem__(key, value)

    def _indexer(self):
        return self.tree._unpack_index(self._indexer_packed())

    def _indexer_packed(self):
        self._maybe_lazy_alloc()
        self._check_ver()
        leaf_key = self.key[0] if isinstance(self.key, tuple) else self.key
        if isinstance(leaf_key, torch.Tensor) and leaf_key.ndim == 0:
            leaf_key = leaf_key[None]
        return self.tree.back_link[leaf_key].long()

    def _unique_node_key(self):
        uniq_ids = torch.unique(self._indexer_packed())
        return self.tree._unpack_index(uniq_ids).unbind(-1)

    def _check_ver(self):
        if self.tree._ver > self._tree_ver:
            self.key = self._packed_ids = None
            raise RuntimeError("N3TreeView has been invalidated because tree " +
                    "data layout has changed")

    def _maybe_lazy_alloc(self):
        self._check_ver()
        if self._points is not None:
            leaf_key = self.tree._maybe_lazy_alloc(self._points)
            self.key = (leaf_key, self.key[1:]) if isinstance(self.key, tuple) else leaf_key
            self._tree_ver = self.tree._ver  # Auto-synced indices
            self._points = None

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
    redir_inplace_funcs = ['relu_', 'sigmoid_', 'clamp_',
                           'clamp_min_', 'clamp_max_', 'sqrt_']
    def redirect_func(redir_func, grad=False, inplace=False):
        def redir_impl(self, *args, **kwargs):
            if inplace:
                self._maybe_lazy_alloc()
            self._check_ver()
            return getattr(self.values if grad else self.values_nograd, redir_func)(
                    *args, **kwargs)
        setattr(N3TreeView, redir_func, redir_impl)
    for redir_func in redir_funcs:
        redirect_func(redir_func)
    for redir_func in redir_grad_funcs:
        redirect_func(redir_func, grad=True)
    for redir_func in redir_inplace_funcs:
        redirect_func(redir_func, inplace=True)
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

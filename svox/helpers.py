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
import torch
import numpy as np

class N3TreeView:
    def __init__(self, tree, key):
        self.tree = tree
        local = False
        if isinstance(key, LocalIndex):
            key = key.val
            local = True
        if isinstance(key, tuple) and len(key) >= 3:
            main_key = torch.tensor(key[:3], dtype=torch.float32,
                        device=tree.data.device).reshape(1, 3)
            if len(key) > 3:
                key = (main_key, *key[3:])
            else:
                key = main_key
        leaf_key = key[0] if isinstance(key, tuple) else key
        if torch.is_tensor(leaf_key) and leaf_key.ndim == 2 and leaf_key.shape[1] == 3:
            if leaf_key.dtype != torch.float32:
                leaf_key = leaf_key.float()
            val, target = tree.forward(leaf_key, want_node_ids=True, world=not local)
            leaf_node = (*tree._unpack_index(target).T,)
        else:
            if isinstance(leaf_key, int):
                leaf_key = torch.tensor([leaf_key], device=tree.data.device)
            leaf_node = self.tree._all_leaves()
            leaf_node = leaf_node.__getitem__(leaf_key).T
        if isinstance(key, tuple):
            self.key = (*leaf_node, *key[1:])
        else:
            self.key = (*leaf_node,)
        self._value = None;

    def __repr__(self):
        return "N3TreeView(" + repr(self.values) + ")"

    def __torch_function__(self, func, types, args=(), kwargs=None):
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
        self.tree.data.data[self.key] = value

    def refine(self):
        """
        Refine selected leaves using tree.refine
        """
        return self.tree.refine(sel=self.key[:4])

    @property
    def values(self):
        """
        Values of the selected leaves (autograd enabled)

        :return: (n_leaves, data_dim) float32 note this is 2D even if key is int 
        """
        return self.tree.data[self.key]

    @property
    def values_nograd(self):
        """
        Values of the selected leaves (no autograd)

        :return: (n_leaves, data_dim) float32 note this is 2D even if key is int 
        """
        return self.tree.data.data[self.key]

    @property
    def shape(self):
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
        return self.tree.parent_depth[self.key[0], 1]
    
    @property
    def lengths(self):
        """
        Get a list of selected leaf side lengths in tree (world dimensions),
        in same order as values, corners, depths

        :return: (n_leaves) float
        """
        return 2.0 ** (-self.depths.float() - 1.0) / self.tree.invradius

    @property
    def lengths_local(self):
        """
        Get a list of selected leaf side lengths in tree
        (local index :math:`[0, 1]^3`),
        in same order as values, corners, depths

        :return: (n_leaves) float
        """
        return 2.0 ** (-self.depths.float() - 1.0)

    @property
    def corners(self):
        """
        Get a list of selected leaf lower corners in tree
        (world coordinates),
        in same order as values, lengths, depths

        :return: (n_leaves, 3) float
        """
        return self.tree._calc_corners(self._indexer())

    @property
    def corners_local(self):
        """
        Get a list of selected leaf lower corners in tree
        (local index :math:`[0, 1]^3`),
        in same order as values, lengths, depths

        :return: (n_leaves, 3) float
        """
        return (self.tree._calc_corners(self._indexer())
                - self.tree.offset) / self.tree.invradius

    def sample(self, n_samples):
        """
        Sample n_samples uniform points in each selected leaf (world coordinates)

        :return: (n_leaves, n_samples, 3) float
        """
        corn = self.corners
        length = self.lengths
        u = torch.rand((corn.shape[0], n_samples, 3),
                device=length.device,
                dtype=length.dtype) * length[:, None, None]
        return corn[:, None] + u

    def sample_local(self, n_samples):
        """
        Sample n_samples uniform points in each selected leaf
        (local index :math:`[0, 1]^3`)

        :return: (n_leaves, n_samples, 3) float
        """
        corn = self.corners_local
        length = self.lengths_local
        u = torch.rand((corn.shape[0], n_samples, 3),
                device=length.device,
                dtype=length.dtype) * length[:, None, None]
        return corn[:, None] + u

    # In-place modification helpers
    def normal_(self, mean=0.0, std=1.0):
        """
        Set all values to random normal

        :param mean: normal mean
        :param std: normal std

        """
        self.tree.data.data[self.key] = torch.randn_like(
                self.tree.data.data[self.key]) * std + mean

    def uniform_(self, min=0.0, max=1.0):
        """
        Set all values to random uniform

        :param min: interval min
        :param max: interval max

        """
        self.tree.data.data[self.key] = torch.rand_like(
                self.tree.data.data[self.key]) * (max - min) + min

    def clamp_(self, min=None, max=None):
        """
        Clamp.

        :param min: clamp min value, None=disable
        :param max: clamp max value, None=disable

        """
        self.tree.data.data[self.key] = self.tree.data.data[self.key].clamp(min, max)

    def relu_(self):
        """
        Relu.
        """
        self.tree.data.data[self.key] = torch.relu(self.tree.data.data[self.key])

    def sigmoid_(self):
        """
        Sigmoid.
        """
        self.tree.data.data[self.key] = torch.sigmoid(self.tree.data.data[self.key])

    def _indexer(self):
        return torch.stack(self.key[:4], dim=-1)

# Redirect functions to Tensor
def _redirect_funcs():
    redir_funcs = ['__floordiv__', '__mod__', '__div__',
                   '__eq__', '__ne__', '__ge__', '__gt__', '__le__',
                   '__lt__', '__floor__', '__ceil__', '__round__', '__len__',
                   'item', 'size', 'dim', 'numel']
    redir_grad_funcs = ['__add__', '__mul__', '__sub__'
                   '__mod__', '__div__', '__radd__', '__rsub__', '__rmul__',
                   '__rdiv__', '__abs__', '__pos__', '__neg__',
                   '__len__', 'clamp', 'clamp_max', 'clamp_min', 'relu', 'sigmoid']
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
             "Operations will be slow " +
             "Please do not import svox in the SVOX source directory.")
    return _C

class LocalIndex:
    def __init__(self, val):
        self.val = val

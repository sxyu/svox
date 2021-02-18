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

class _N3TreeView:
    def __init__(self, tree, key):
        self.tree = tree
        local = False
        if isinstance(key, LocalIndex):
            key = key.val
            local = True
        if torch.is_tensor(key) and key.ndim == 2 and key.shape[1] == 3:
            val, target = tree.forward(key, want_node_ids=True, world=not local)
            key = tree._reverse_leaf_search(target)
        self.key = key
        self._value = None;

    def __repr__(self):
        return "N3TreeView(" + repr(self.values) + ")"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        new_args = []
        for arg in args:
            if isinstance(arg, _N3TreeView):
                new_args.append(arg.values)
            else:
                new_args.append(arg)
        return func(*new_args, **kwargs)

    def set(self, value):
        leaf_node = self.tree._all_leaves()
        leaf_node = leaf_node.__getitem__(self._leaf_key())
        if isinstance(self.key, tuple):
            leaf_node_sel = (*leaf_node.T, *self.key[1:])
        else:
            leaf_node_sel = (*leaf_node.T,)
        self.tree.data.data[leaf_node_sel] = value

    def refine(self):
        return self.tree.refine(sel=self._leaf_key())

    @property
    def values(self):
        if self._value is None:
            self._value = self.tree.values(sel=self.key)
        return self._value

    @property
    def values_diff(self):
        return self.tree.values(sel=self.key, diff=True)

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return 2

    @property
    def mask(self):
        """
        Get leaf mask (1D)
        """
        n_leaves = self.tree.n_leaves
        result = np.zeros((n_leaves,), dtype=np.bool)
        result.__setitem__(self._leaf_key(), True)
        return result

    @property
    def depths(self):
        return self.tree.depths(self._leaf_key())
    
    @property
    def lengths(self):
        return self.tree.lengths(self._leaf_key())

    @property
    def lengths_local(self):
        return self.tree.lengths(self._leaf_key(), world=False)

    @property
    def corners(self):
        return self.tree.corners(self._leaf_key())

    @property
    def corners_local(self):
        return self.tree.corners(self._leaf_key(), world=False)

    def sample(self, n_samples):
        return self.tree.sample(n_samples, self._leaf_key())

    def _leaf_key(self):
        return self.key[0] if isinstance(self.key, tuple) else self.key

# Redirect functions to Tensor
def _redirect_funcs():
    redir_funcs = ['__add__', '__mul__', '__sub__', '__floordiv__',
                   '__mod__', '__div__', '__radd__', '__rsub__', '__rmul__',
                   '__rdiv__', '__eq__', '__ne__', '__ge__', '__gt__', '__le__',
                   '__lt__', '__abs__', '__floor__', '__ceil__', '__pos__', '__neg__',
                   '__round__',
                   'item', 'size', 'dim', 'numel', 'detach', 'cpu', 'cuda', 'to',
                   'float', 'double', 'int', 'long', 'clamp_', 'relu_', 'sigmoid_',
                   'clamp_max_', 'clamp_min_']
    def redirect_func(redir_func):
        def redir_impl(self, *args, **kwargs):
            return getattr(self.values, redir_func)(*args, **kwargs)
        setattr(_N3TreeView, redir_func, redir_impl)
    for redir_func in redir_funcs:
        redirect_func(redir_func)
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

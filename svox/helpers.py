import torch
import numpy as np

class _N3TreeView:
    def __init__(self, tree, key):
        self.tree = tree
        if torch.is_tensor(key) and key.ndim == 2 and key.shape[1] == 3:
            val, target = tree.get(key, want_node_ids=True)
            key = tree._reverse_leaf_search(target)
        self.key = key
        self._value = None;

    def __repr__(self):
        return "N3TreeView(" + repr(self.value) + ")"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        new_args = []
        for arg in args:
            if isinstance(arg, _N3TreeView):
                new_args.append(arg.value)
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
        return self.tree.refine(mask=self.mask)

    @property
    def value(self):
        if self._value is None:
            self._value = self.tree.values().__getitem__(self.key)
        return self._value

    @property
    def shape(self):
        return self.value.shape

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
    def corners(self):
        return self.tree.corners(self._leaf_key())

    def sample(self, n_samples):
        return self.tree.sample(n_samples, self._leaf_key())

    @property
    def requires_grad(self):
        return self.value.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self.value.requires_grad = value

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
                   'float', 'double', 'int', 'long']
    def redirect_func(redir_func):
        def redir_impl(self, *args, **kwargs):
            return getattr(self.value, redir_func)(*args, **kwargs)
        setattr(_N3TreeView, redir_func, redir_impl)
    for redir_func in redir_funcs:
        redirect_func(redir_func)
_redirect_funcs()


def get_c_extension():
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

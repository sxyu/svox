"""
[BSD 2-CLAUSE LICENSE]

Copyright Alex Yu 2021

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import os.path as osp
import numpy as np
from warnings import warn

class _N3TreeView:
    def __init__(self, tree, key):
        self.tree = tree
        if isinstance(key, np.ndarray) and key.ndim == 2 and key.shape[1] == 3:
            packed_sorted = tree._pack_index(tree._all_leaves())
            val, target = tree.get(key, want_node_ids=True)
            key = np.searchsorted(packed_sorted, target, side='left')
        self.key = key
        self._value = None

    def __repr__(self):
        return "N3TreeView(" + repr(self.value()) + ")"

    def value(self):
        if self._value is None:
            self._value = self.tree.values().__getitem__(self.key)
        return self._value

    def __array_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        new_args = []
        for arg in args:
            if isinstance(arg, _N3TreeView):
                new_args.append(arg.value())
            else:
                new_args.append(arg)
        return func(*new_args, **kwargs)

    def set(self, value):
        self.tree._push_to_leaf()
        leaf_node = self.tree._all_leaves()
        key1 = self.key[0] if isinstance(self.key, tuple) else self.key
        leaf_node = leaf_node.__getitem__(key1)
        if isinstance(self.key, tuple):
            leaf_node_sel = (*leaf_node.T, *self.key[1:])
        else:
            leaf_node_sel = (*leaf_node.T,)
        self.tree.data[leaf_node_sel] = value

    def refine(self):
        return self.tree.refine(mask=self.mask)

    @property
    def mask(self):
        """
        Get leaf mask (1D)
        """
        n_leaves = self.tree.n_leaves
        result = np.zeros((n_leaves,), dtype=np.bool)
        key1 = self.key[0] if isinstance(self.key, tuple) else self.key
        result.__setitem__(key1, True)
        return result


    @property
    def shape(self):
        return self.value().shape

# Redirect functions to Tensor
def _redirect_funcs():
    redir_funcs = ['__add__', '__mul__', '__sub__', '__floordiv__',
                   '__mod__', '__div__', '__radd__', '__rsub__', '__rmul__',
                   '__rdiv__', '__eq__', '__ne__', '__ge__', '__gt__', '__le__',
                   '__lt__', '__abs__', '__floor__', '__ceil__', '__pos__', '__neg__',
                   '__round__',
                   'item', 'astype']
    def redirect_func(redir_func):
        def redir_impl(self, *args, **kwargs):
            val = self.value()
            return getattr(self.value(), redir_func)(*args, **kwargs)
        setattr(_N3TreeView, redir_func, redir_impl)
    for redir_func in redir_funcs:
        redirect_func(redir_func)
_redirect_funcs()

class N3Tree:
    """
    N^3 tree prototype implementaton
    """
    def __init__(self, N=4, data_dim=4, depth_limit=4,
            init_reserve=4, init_refine=0, geom_resize_fact=1.5, padding_mode="zeros",
            radius=0.5, center=[0.5, 0.5, 0.5], map_location="cpu"):
        """
        :param N branching factor N
        :param data_dim size of data stored at each leaf
        :param depth_limit maximum depth of tree to stop branching/refining
        :param init_reserve amount of nodes to reserve initially
        :param init_refine number of times to refine entire tree initially
        :param geom_resize_fact geometric resizing factor
        :param padding_mode padding mode for coords outside grid, zeros | border
        :param radius 1/2 side length of cube
        :param center center of space
        """
        assert N >= 2
        assert depth_limit >= 0
        self.N = N
        self.data_dim = data_dim

        if init_refine > 0:
            for i in range(1, init_refine + 1):
                init_reserve += (N ** i) ** 3

        self.data = np.zeros((init_reserve, N, N, N, data_dim), dtype=np.float32)
        self.child = np.zeros(
            (init_reserve, N, N, N), dtype=np.int32)
        self.parent_depth = np.zeros((init_reserve, 2), dtype=np.int32)

        self.n_internal = 1
        self.max_depth = 0

        self.invradius = 0.5 / radius
        center = np.array(center)
        self.offset = 0.5 * (1.0 - center / radius)

        self.depth_limit = depth_limit
        self.geom_resize_fact = geom_resize_fact
        self.padding_mode = padding_mode

        self._last_all_leaves = None
        self.refine(repeats=init_refine)


    # Main accesors
    def set(self, indices, values, cuda=True):
        """
        Set tree values,
        :param indices (Q, 3)
        :param values (Q, K)
        :param cuda ignored for legacy reasons
        Beware: If multiple indices point to same leaf node,
        only one of them will be taken
        """
        assert len(indices.shape) == 2

        indices = self._transform_coord(indices)

        if self.padding_mode == "zeros":
            outside_mask = ((indices >= 1.0) | (indices < 0.0)).any(axis=-1)
            indices = indices[~outside_mask]

        n_queries, _ = indices.shape
        indices = np.clip(indices, 0.0, 1.0 - 1e-10)
        ind = indices.copy()

        node_ids = np.zeros((n_queries,), dtype=np.long)
        accum = np.zeros((n_queries, self.data_dim), dtype=np.float32)
        remain_mask = np.ones((n_queries,), dtype=np.bool)
        while remain_mask.any():
            ind_floor = np.minimum(np.floor(ind[remain_mask] * self.N), self.N - 1)
            sel = (node_ids[remain_mask], *(ind_floor.astype(np.long).T),)

            deltas = self.child[sel]
            vals = self.data[sel]

            nonterm_partial_mask = deltas != 0
            nonterm_mask = np.zeros((n_queries,), dtype=np.bool)
            nonterm_mask[remain_mask] = nonterm_partial_mask

            accum[nonterm_mask] += vals[nonterm_partial_mask]

            node_ids[remain_mask] += deltas
            ind[remain_mask] = ind[remain_mask] * self.N - ind_floor

            term_mask = remain_mask & ~nonterm_mask
            vals[~nonterm_partial_mask] = values[term_mask] - accum[term_mask]
            self.data[sel] = vals

            remain_mask &= nonterm_mask

    def get(self, indices, want_node_ids=False, cuda=True):
        """
        Get tree values,
        :param indices (Q, 3)
        """
        return self(indices, want_node_ids, cuda=cuda)

    def __call__(self, indices, want_node_ids=False, cuda=True):
        """
        Get tree values,
        :param indices (Q, 3)
        """
        assert len(indices.shape) == 2

        indices = self._transform_coord(indices)

        if self.padding_mode == "zeros":
            outside_mask = ((indices >= 1.0) | (indices < 0.0)).any(axis=-1)
        indices = np.clip(indices, 0.0, 1.0 - 1e-10)

        n_queries, _ = indices.shape
        ind = indices.copy()
        node_ids = np.zeros((n_queries,), dtype=np.long)
        result = np.zeros((n_queries, self.data_dim), dtype=np.float32)
        remain_mask = np.ones((n_queries,), dtype=np.bool)
        if self.padding_mode == "zeros":
            remain_mask &= ~outside_mask
        if want_node_ids:
            subidx = np.zeros((n_queries, 3), dtype=np.long)

        while remain_mask.any():
            ind_floor = np.minimum(np.floor(ind[remain_mask] * self.N), self.N - 1)
            sel = (node_ids[remain_mask], *(ind_floor.astype(np.long).T),)

            deltas = self.child[sel]
            vals = self.data[sel]

            nonterm_partial_mask = deltas != 0
            nonterm_mask = np.zeros((n_queries,), dtype=np.bool)
            nonterm_mask[remain_mask] = nonterm_partial_mask

            result[remain_mask] += vals
            if want_node_ids:
                subidx[remain_mask & (~nonterm_partial_mask)] = \
                            ind_floor.astype(np.long)[~nonterm_partial_mask]

            node_ids[remain_mask] += deltas
            ind[remain_mask] = ind[remain_mask] * self.N - ind_floor
            remain_mask = remain_mask & nonterm_mask

        if self.padding_mode == "zeros":
            result[outside_mask] = 0.0

        if want_node_ids:
            txyz = np.concatenate([node_ids[:, None], subidx], axis=-1)
            return result, self._pack_index(txyz)
        return result

    # In-place modification helpers
    def randn_(self, mean=0.0, std=1.0):
        """
        Set all values to random normal
        Side effect: pushes values to leaf.
        """
        self._push_to_leaf()
        leaf_node = self._all_leaves()  # NNC, 4
        leaf_node_sel = (*leaf_node.T,)
        self.data[leaf_node_sel] = np.randn_like(
                self.data[leaf_node_sel]) * std + mean

    def clamp_(self, min, max, axis=None):
        """
        Clamp all values to random normal
        Side effect: pushes values to leaf.
        """
        self._push_to_leaf()
        leaf_node = self._all_leaves()  # NNC, 4
        if dim is None:
            leaf_node_sel = (*leaf_node.T,)
        else:
            leaf_node_sel = (*leaf_node.T, np.ones_like(leaf_node[..., 0]) * dim)
        self.data[leaf_node_sel] = self.data[leaf_node_sel].clamp(min, max)

    # Leaf refinement methods
    def refine(self, axis=-1, thresh=None, mask=None, max_refine=None, repeats=1):
        """
        Refine each leaf node, optionally filtering by value at dimension 'dim' >= 'thresh'.
        Respects depth_limit.
        Side effect: pushes values to leaf.
        :param dim dimension to check (used if thresh != None). Can be negative (like -1)
        :param thresh threshold for dimension.
        :param mask leaf mask
        :param max_refine maximum number of leaves to refine.
        'max_refine' random leaves (without replacement)
        meeting the threshold are refined in case more leaves
        satisfy it. Leaves at lower depths are sampled exponentially
        more often.
        """
        resized = False
        for _ in range(repeats):
            filled = self.n_internal
            good_mask = self.child[:filled] == 0
            if mask is not None:
                good_mask[good_mask] &= mask

            if thresh is not None:
                self._push_to_leaf()
                good_mask &= (self.data[:filled, ..., dim] >= thresh)

            good_mask &= (self.parent_depth[:filled, -1] <
                    self.depth_limit)[:, None, None, None]

            leaf_node = np.stack(good_mask.nonzero(), axis=-1)  # NNC, 4
            if leaf_node.shape[0] == 0:
                # Nothing to do
                return False

            if max_refine is not None and max_refine < leaf_node.shape[0]:
                prob = np.pow(1.0 / (self.N ** 3), self.parent_depth[leaf_node[:, 0], 1])
                prob = prob.astype(np.float64)
                prob /= prob.sum()
                choices = np.random.choice(leaf_node.shape[0], max_refine, replace=False,
                        p=prob)
                leaf_node = leaf_node[choices]

            leaf_node_sel = (*leaf_node.T,)
            num_nc = leaf_node.shape[0]
            new_filled = filled + num_nc

            cap_needed = new_filled - self.capacity
            if cap_needed > 0:
                self._resize_add_cap(cap_needed)
                resized = True

            new_idxs = np.arange(filled, filled + num_nc, dtype=self.child.dtype) # NNC

            self.child[filled:new_filled] = 0
            self.child[leaf_node_sel] = new_idxs - leaf_node[:, 0].astype(np.int32)
            self.parent_depth[filled:new_filled, 0] = self._pack_index(leaf_node)  # parent
            self.parent_depth[filled:new_filled, 1] = self.parent_depth[
                    leaf_node[:, 0], 1] + 1  # depth
            self.max_depth = max(self.parent_depth[filled:new_filled, 1].max().item(),
                    self.max_depth)

            self.n_internal += num_nc
        if repeats > 0:
            self._last_all_leaves = None
        return resized

    def _refine_at(self, intnode_idx, xyzi):
        """
        Advanced: refine specific leaf node.
        :param intnode_idx index of internal node for identifying leaf
        :param xyzi tuple of size 3 with each element in {0, ... N-1}
        in xyz orde rto identify leaf within internal node
        """
        assert min(xyzi) >= 0 and max(xyzi) < self.N
        if self.parent_depth[intnode_idx, 1] >= self.depth_limit:
            return

        xi, yi, zi = xyzi
        if self.child[intnode_idx, xi, yi, zi] != 0:
            # Already has child
            return

        resized = False
        filled = self.n_internal
        if filled >= self.capacity:
            self._resize_add_cap(1)
            resized = True

        self.child[filled] = 0
        self.child[intnode_idx, xi, yi, zi] = filled - intnode_idx
        depth = self.parent_depth[intnode_idx, 1] + 1
        self.parent_depth[filled, 0] = self._pack_index(np.array(
            [[intnode_idx, xi, yi, zi]], dtype=np.int32))[0]
        self.parent_depth[filled, 1] = depth
        self.max_depth = max(self.max_depth, depth)
        self.n_internal += 1
        self._last_all_leaves = None
        return resized

    def shrink_to_fit(self):
        """
        Shrink data & buffers to tightly needed fit tree data.
        """
        new_cap = self.n_internal
        if new_cap >= self.capacity:
            return False
        self.data = self.data[:new_cap]
        self.child.resize(new_cap, *self.child.shape[1:])
        self.parent_depth.resize(new_cap, *self.parent_depth.shape[1:])
        return True

    # Misc
    @property
    def n_leaves(self):
        """
        Get number of leaf nodes (WARNING: slow)
        """
        return self._all_leaves().shape[0]

    @property
    def n_nodes(self):
        """
        Get number of total leaf+internal nodes (WARNING: slow)
        """
        return self.n_internal + self.n_leaves

    @property
    def capacity(self):
        """
        Get capacity (n_internal is amount taken)
        """
        return self.parent_depth.shape[0]

    def values(self, depth=None):
        """
        Get a list of all leaf values in tree
        Side effect: pushes values to leaf.
        :return (n_leaves, data_dim)
        """
        self._push_to_leaf()
        leaf_node = self._all_leaves()
        leaf_node_sel = (*leaf_node.T,)
        data = self.data[leaf_node_sel]
        if depth is not None:
            depths = self.parent_depth[leaf_node[:, 0], 1]
            data = data[depths == depth]
        return data

    @property
    def depth(self):
        return self.depths()

    def depths(self):
        """
        Get a list of leaf depths in tree,
        in same order as values(), corners().
        Root is at depth 0.
        :return (n_leaves) int32
        """
        leaf_node = self._all_leaves()
        return self.parent_depth[leaf_node[:, 0], 1]

    def corners(self):
        """
        Get a list of leaf lower xyz corners in tree,
        in same order as values(), depths().
        :return (n_leaves, 3)
        """
        leaf_node = self._all_leaves()
        corners = self._calc_corners(leaf_node)
        return corners

    def savez(self, path):
        data = {
            "data_dim" : self.data_dim,
            "child" : self.child,
            "parent_depth" : self.parent_depth,
            "n_internal" : self.n_internal,
            "max_depth" : self.max_depth,
            "invradius" : self.invradius,
            "offset" : self.offset,
            "depth_limit": self.depth_limit,
            "geom_resize_fact": self.geom_resize_fact,
            "padding_mode": self.padding_mode
        }
        if self.data_dim != 3 and self.data_dim != 4:
            data["data"] = self.data
        else:
            import imageio
            data_path = osp.splitext(path)[0] + '_data.exr'
            imageio.imwrite(data_path, self.data.reshape(-1,
                self.N ** 2, self.data_dim))
        np.savez_compressed(path, **data)

    def loadz(self, path):
        z = np.load(path)
        self.data_dim = int(z["data_dim"])
        self.child = z["child"]
        self.N = self.child.shape[-1]
        self.parent_depth = z["parent_depth"]
        self.n_internal = z["n_internal"].item()
        self.max_depth = z["max_depth"].item()
        self.invradius = z["invradius"].item()
        self.offset = z["offset"]
        self.depth_limit = int(z["depth_limit"])
        self.geom_resize_fact = float(z["geom_resize_fact"])
        self.padding_mode = str(z["padding_mode"])
        if self.data_dim != 3 and self.data_dim != 4:
            self.data = z["data"]
        else:
            import imageio
            data_path = osp.splitext(path)[0] + '_data.exr'
            self.data = imageio.imread(data_path).reshape(
                            -1, self.N, self.N, self.N, self.data_dim
                        )

    # Magic
    def __repr__(self):
        return ("svox.N3Tree(N={}, data_dim={}, depth_limit={};" +
                " capacity:{}/{} max_depth:{})").format(
                    self.N, self.data_dim, self.depth_limit,
                    self.n_internal, self.capacity, self.max_depth)

    def __getitem__(self, key):
        if isinstance(key, list):
            key = np.array(key)
        if isinstance(key, tuple) and len(key) == 3:
            # Use x,y,z format
            return self.get(np.array(key, dtype=np.float32)[None])[0]
        else:
            return _N3TreeView(self, key)

    def __setitem__(self, key, val):
        if isinstance(key, list):
            key = np.array(key)
        if isinstance(key, tuple) and len(key) == 3:
            # Use x,y,z format
            key_tensor = np.array(key, dtype=np.float32)[None]
            self.set(key_tensor, self._make_val_tensor(val))
        else:
            _N3TreeView(self, key).set(val)


    def __iadd__(self, val):
        self.data[0] += self._make_val_tensor(val)[None, None]
        return self

    def __isub__(self, val):
        self.data[0] -= self._make_val_tensor(val)[None, None]
        return self

    def __imul__(self, val):
        self.data *= self._make_val_tensor(val)[None, None, None]
        return self

    def __idiv__(self, val):
        self.data /= self._make_val_tensor(val)[None, None, None]
        return self

    # Internal utils
    def _push_to_leaf(self):
        """
        Push tree values to leaf
        """
        filled = self.n_internal

        leaf_node = np.stack((self.child[:filled] == 0).nonzero(), axis=-1)  # NNC, 4
        curr = leaf_node.copy()

        while True:
            good_mask = curr[:, 0] != 0
            if not good_mask.any():
                break
            curr = curr[good_mask]
            leaf_node = leaf_node[good_mask]

            curr = self._unpack_index(self.parent_depth[curr[:, 0], 0].astype(np.long))
            self.data[(*leaf_node.T,)] += self.data[(*curr.T,)]

        with_child = np.stack(self.child[:filled].nonzero(), axis=-1)  # NNC, 4
        with_child_sel = (*with_child.T,)
        self.data[with_child_sel] = 0.0


    def _calc_corners(self, nodes):
        """
        Compute lower bbox corners for given nodes
        :nodes (Q, 4)
        :return (Q, 3)
        """
        Q, _ = nodes.shape
        filled = self.n_internal

        curr = nodes.copy()
        mask = np.ones((Q,), dtype=np.bool)
        output = np.zeros((Q, 3), dtype=np.float32)

        while True:
            output[mask] += curr[:, 1:]
            output[mask] /= self.N

            good_mask = curr[:, 0] != 0
            if not good_mask.any():
                break
            mask[mask] = good_mask

            curr = self._unpack_index(self.parent_depth[curr[good_mask, 0], 0].astype(np.long))

        return output


    def _pack_index(self, txyz):
        return txyz[:, 0] * (self.N ** 3) + txyz[:, 1] * (self.N ** 2) + \
               txyz[:, 2] * self.N + txyz[:, 3]

    def _unpack_index(self, flat):
        t = []
        for i in range(3):
            t.append(flat % self.N)
            flat //= self.N
        return np.stack((flat, t[2], t[1], t[0]), axis=-1)

    def _resize_add_cap(self, cap_needed):
        """
        Helper for increasing capacity
        """
        cap_needed = max(cap_needed, int(self.capacity * (self.geom_resize_fact - 1.0)))
        self.data = np.concatenate((self.data,
                        np.zeros((cap_needed, *self.data.shape[1:]), dtype=np.float32)), axis=0)
        self.child.resize(self.capacity + cap_needed, *self.child.shape[1:])
        self.parent_depth.resize(self.capacity + cap_needed, *self.parent_depth.shape[1:])

    def _make_val_tensor(self, val):
        val_tensor = np.array(val, dtype=np.float32)
        while len(val_tensor.shape) < 2:
            val_tensor = val_tensor[None]
        if val_tensor.shape[-1] == 1:
            val_tensor = np.repeat(val_tensor, self.data_dim, axis=-1)
        else:
            assert val_tensor.shape[-1] == self.data_dim
        return val_tensor

    def _all_leaves(self):
        """
        Get all leaves of tree
        """
        if self._last_all_leaves is None:
            self._last_all_leaves = np.stack(
                    (self.child[:self.n_internal] == 0).nonzero(), axis=-1)
        return self._last_all_leaves

    def _transform_coord(self, indices):
        return self.offset + indices * self.invradius

    # Array analogy
    @property
    def shape(self):
        return (self.n_leaves, self.data_dim)

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return self.n_leaves * self.data_dim

    def __len__(self):
        return self.n_leaves

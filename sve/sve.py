import torch
import numpy as np
from torch import nn
from warnings import warn
try:
    import sve.csrc as _C
    if not hasattr(_C, "query_vertical"):
        warn("CUDA extension sve.csrc could not be loaded! " +
             "Operations will be slow " +
             "Please do not import sve in the SVE source directory.")
        _C = None
except:
    _C = None

torch.autograd.set_detect_anomaly(True)

class _SVEQueryVerticalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, child, indices, vary_non_leaf):
        out = _C.query_vertical(data, child, indices, vary_non_leaf)
        ctx.save_for_backward(child, indices)
        ctx.vary_non_leaf = vary_non_leaf
        return out

    @staticmethod
    def backward(ctx, grad_out):
        child, indices = ctx.saved_tensors

        grad_out = grad_out.contiguous()
        if ctx.needs_input_grad[0]:
            grad_data = _C.query_vertical_backward(
                    child, indices, grad_out, ctx.vary_non_leaf)
        else:
            grad_data = None

        return grad_data, None, None, None


class N3Tree(nn.Module):
    """
    N^3 tree prototype implementaton
    """
    def __init__(self, N=4, data_dim=4, depth_limit=4,
            vary_non_leaf=True, init_reserve=4,
            init_refine=0, geom_resize_fact=1.5):
        """
        :param N branching factor N
        :param data_dim size of data stored at each leaf
        :param depth_limit maximum depth of tree to stop branching/refining
        :param vary_non_leaf use internal node values (add to leaf along path)
        :param init_reserve amount of nodes to reserve initially
        :param init_refine number of times to refine entire tree initially
        :param geom_resize_fact geometric resizing factor
        WARNING: nn.Parameters can change due to refinement, if refine returns True
        please re-make any optimizers
        """
        super().__init__()
        assert N >= 2
        assert depth_limit >= 0
        self.N = N
        self.data_dim = data_dim
        self.vary_non_leaf = vary_non_leaf

        if init_refine > 0:
            for i in range(1, init_refine + 1):
                init_reserve += (N ** i) ** 3

        self.register_parameter("data", nn.Parameter(
            torch.zeros(init_reserve, N, N, N, data_dim)))
        self.register_buffer("child", torch.zeros(
            init_reserve, N, N, N, dtype=torch.int32))
        self.register_buffer("parent_depth", torch.zeros(
            init_reserve, 2, dtype=torch.int32))

        self.register_buffer("n_internal", torch.tensor(1))
        self.register_buffer("max_depth", torch.tensor(0))
        self.depth_limit = depth_limit
        self.geom_resize_fact = geom_resize_fact

        self.refine_all(init_refine)


    # Main accesors
    def set(self, indices, values, cuda=True):
        """
        Set tree values,
        :param indices (Q, 3)
        :param values (Q, K)
        Beware: If multiple indices point to same leaf node,
        only one of them will be taken
        """
        assert len(indices.shape) == 2
        assert not indices.requires_grad  # Grad wrt indices not supported
        assert not values.requires_grad  # Grad wrt values not supported
        indices = indices.to(device=self.data.device)
        values = values.to(device=self.data.device)

        if not cuda or _C is None:
            warn("Using slow assignment")
            n_queries, _ = indices.shape
            ind = indices.clone()
            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            accum = torch.zeros((n_queries, self.data_dim), dtype=torch.float32,
                                  device=indices.device)
            remain_mask = torch.ones(n_queries, dtype=torch.bool, device=indices.device)
            while remain_mask.any():
                ind_floor = torch.floor(ind[remain_mask] * self.N)
                sel = (node_ids[remain_mask], *(ind_floor.long().T),)

                deltas = self.child[sel]
                vals = self.data.data[sel]

                nonterm_partial_mask = deltas != 0
                nonterm_mask = torch.zeros(n_queries, dtype=torch.bool, device=indices.device)
                nonterm_mask[remain_mask] = nonterm_partial_mask

                if self.vary_non_leaf:
                    accum[nonterm_mask] += vals[nonterm_partial_mask]

                node_ids[remain_mask] += deltas
                ind[remain_mask] = ind[remain_mask] * self.N - ind_floor

                term_mask = remain_mask & ~nonterm_mask
                vals[~nonterm_partial_mask] = values[term_mask] - accum[term_mask]
                self.data.data[sel] = vals

                remain_mask &= nonterm_mask
        else:
            _C.assign_vertical(self.data, self.child, indices,
                               values, self.vary_non_leaf)

    def get(self, indices, cuda=True):
        """
        Get tree values,
        :param indices (Q, 3)
        """
        indices = indices.to(device=self.data.device)
        return self.forward(indices, cuda=cuda)

    def forward(self, indices, cuda=True):
        """
        Get tree values,
        :param indices (Q, 3)
        """
        assert not indices.requires_grad  # Grad wrt indices not supported
        assert len(indices.shape) == 2

        if not cuda or _C is None:
            warn("Using slow query")
            n_queries, _ = indices.shape
            ind = indices.clone()
            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            result = torch.zeros((n_queries, self.data_dim), dtype=torch.float32,
                                  device=indices.device)
            remain_mask = torch.ones(n_queries, dtype=torch.bool, device=indices.device)
            while remain_mask.any():
                ind_floor = torch.floor(ind[remain_mask] * self.N)
                sel = (node_ids[remain_mask], *(ind_floor.long().T),)

                deltas = self.child[sel]
                vals = self.data[sel]

                nonterm_partial_mask = deltas != 0
                nonterm_mask = torch.zeros(n_queries, dtype=torch.bool, device=indices.device)
                nonterm_mask[remain_mask] = nonterm_partial_mask

                if self.vary_non_leaf:
                    result[remain_mask] += vals

                node_ids[remain_mask] += deltas
                ind[remain_mask] = ind[remain_mask] * self.N - ind_floor
                remain_mask = remain_mask & nonterm_mask

                if not self.vary_non_leaf:
                    result[remain_mask] = vals[nonterm_partial_mask]
            return result
        else:
            return _SVEQueryVerticalFunction.apply(
                    self.data, self.child, indices, self.vary_non_leaf)

    # In-place modification helpers
    def randn_(self, mean=0.0, std=1.0):
        """
        Set all values to random normal
        Side effect: pushes values to leaf.
        """
        self._push_to_leaf()
        leaf_node = self._all_leaves()  # NNC, 4
        leaf_node_sel = (*leaf_node.T,)
        self.data.data[leaf_node_sel] = torch.randn_like(self.data.data[leaf_node_sel]) * std + mean

    def clamp_(self, min, max, dim=None):
        """
        Clamp all values to random normal
        Side effect: pushes values to leaf.
        """
        self._push_to_leaf()
        leaf_node = self._all_leaves()  # NNC, 4
        if dim is None:
            leaf_node_sel = (*leaf_node.T,)
        else:
            leaf_node_sel = (*leaf_node.T, torch.ones_like(leaf_node[..., 0]) * dim)
        self.data.data[leaf_node_sel] = self.data.data[leaf_node_sel].clamp(min, max)

    # Leaf refinement methods
    def refine_thresh(self, dim, thresh, max_refine=None):
        """
        Refine each leaf node whose value at dimension 'dim' >= 'thresh'.
        Respects depth_limit.
        Side effect: pushes values to leaf.
        :param dim dimension to check. Can be negative (like -1)
        :param thresh threshold for dimension.
        :param max_refine maximum number of leaves to refine.
        'max_refine' random leaves (without replacement)
        meeting the threshold are refined in case more leaves
        satisfy it. Leaves at lower depths are sampled exponentially
        more often.
        """
        with torch.no_grad():
            filled = self.n_internal
            resized = False

            good_mask = self.child[:filled] == 0
            if thresh is not None:
                self._push_to_leaf()
                good_mask &= (self.data[:filled, ..., dim] >= thresh)

            good_mask &= (self.parent_depth[:filled, -1] < self.depth_limit)[:, None, None, None]

            leaf_node = good_mask.nonzero(as_tuple=False)  # NNC, 4
            if leaf_node.shape[0] == 0:
                # Nothing to do
                return False

            if max_refine is not None and max_refine < leaf_node.shape[0]:
                prob = torch.pow(1.0 / self.N, self.parent_depth[leaf_node[:, 0], 1])
                choices = np.random.choice(leaf_node.shape[0], max_refine, replace=False,
                        p=prob)
                choices = torch.from_numpy(choices).to(device=leaf_node.device)
                leaf_node = leaf_node[choices]

            leaf_node_sel = (*leaf_node.T,)
            num_nc = leaf_node.shape[0]
            new_filled = filled + num_nc

            cap_needed = new_filled - self.capacity
            if cap_needed > 0:
                self._resize_add_cap(cap_needed)
                resized = True

            new_idxs = torch.arange(filled, filled + num_nc,
                    device=self.child.device, dtype=self.child.dtype) # NNC

            if not self.vary_non_leaf:
                self.data[filled:new_filled] = self.data[leaf_node_sel][:, None, None, None]

            self.child[filled:new_filled] = 0
            self.child[leaf_node_sel] = new_idxs - leaf_node[:, 0].to(torch.int32)
            self.parent_depth[filled:new_filled, 0] = self._pack_index(leaf_node)  # parent
            self.parent_depth[filled:new_filled, 1] = self.parent_depth[
                    leaf_node[:, 0], 1] + 1  # depth
            self.max_depth.fill_(max(self.parent_depth[filled:new_filled, 1].max().item(),
                    self.max_depth.item()))

            self.n_internal += num_nc
            return resized

    def refine_all(self, repeats=1):
        """
        Refine all leaves. Respects depth_limit
        :param repeats number of times to repeat procedure
        """
        resized = False
        for _ in range(repeats):
            resized = self.refine_thresh(0, None) or resized
        return resized

    def refine_at(self, intnode_idx, xyzi):
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
        filled = self.n_internal.item()
        if filled >= self.capacity:
            self._resize_add_cap(1)
            resized = True

        if not self.vary_non_leaf:
            self.data[filled] = self.data[intnode_idx, xi, yi, zi][None, None, None]

        self.child[filled] = 0
        self.child[intnode_idx, xi, yi, zi] = filled - intnode_idx
        depth = self.parent_depth[intnode_idx, 1] + 1
        self.parent_depth[filled, 0] = self._pack_index(torch.tensor(
            [[intnode_idx, xi, yi, zi]], dtype=torch.int32))[0]
        self.parent_depth[filled, 1] = depth
        self.max_depth = max(self.max_depth, depth)
        self.n_internal += 1
        return resized

    def shrink_to_fit(self):
        """
        Shrink data & buffers to tightly needed fit tree data.
        Will change the nn.Parameter size (data), breaking optimizer!
        """
        new_cap = self.n_internal.item()
        if new_cap >= self.capacity:
            return False
        self.data = nn.Parameter(self.data.data[:new_cap])
        self.child.resize_(new_cap, *self.child.shape[1:])
        self.parent_depth.resize_(new_cap, *self.parent_depth.shape[1:])
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
        return self.n_internal.item() + self.n_leaves

    @property
    def capacity(self):
        """
        Get capacity (n_internal is amount taken)
        """
        return self.parent_depth.shape[0]

    def values(self):
        """
        Get a list of all values in tree
        Side effect: pushes values to leaf.
        :return (n_leaves, data_dim)
        """
        self._push_to_leaf()
        leaf_node = self._all_leaves()
        leaf_node_sel = (*leaf_node.T,)
        return self.data[leaf_node_sel]

    # Magic
    def __repr__(self):
        return ("sve.N3Tree(N={}, data_dim={}, depth_limit={};" +
                " capacity:{}/{} max_depth:{})").format(
                    self.N, self.data_dim, self.depth_limit,
                    self.n_internal.item(), self.capacity, self.max_depth.item())

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start is None and key.stop is None:
            # Everything
            return self
        elif isinstance(key, tuple) and len(key) == 3:
            # Use x,y,z format
            return self.get(torch.tensor(key, dtype=torch.float32,
                device=self.data.device)[None])[0]
        else:
            raise NotImplementedError("Unsupported getitem magic")

    def __setitem__(self, key, val):
        if isinstance(key, slice) and key.start is None and key.stop is None:
            # Everything
            self.data.data.zero_()
            self.data.data[0] = self._make_val_tensor(val)
        elif isinstance(key, int):
            # By channel
            self.data.data[..., key].zero_()
            self.data.data[0, ..., key] = val
        elif isinstance(key, tuple) and len(key) == 3:
            # Use x,y,z format
            key_tensor = torch.tensor(key, dtype=torch.float32,
                device=self.data.device)[None]
            self.set(key_tensor, self._make_val_tensor(val))
        else:
            raise NotImplementedError("Unsupported setitem magic")


    def __iadd__(self, val):
        self.data.data[0] += self._make_val_tensor(val)[None, None]
        return self

    def __isub__(self, val):
        self.data.data[0] -= self._make_val_tensor(val)[None, None]
        return self

    def __imul__(self, val):
        self.data.data *= self._make_val_tensor(val)[None, None, None]
        return self

    def __idiv__(self, val):
        self.data.data /= self._make_val_tensor(val)[None, None, None]
        return self

    # Internal utils
    def _push_to_leaf(self):
        """
        Push tree values to leaf
        """
        if not self.vary_non_leaf:
            return
        filled = self.n_internal.item()

        leaf_node = (self.child[:filled] == 0).nonzero(as_tuple=False)  # NNC, 4
        curr = leaf_node.clone()

        while True:
            good_mask = curr[:, 0] != 0
            if not good_mask.any():
                break
            curr = curr[good_mask]
            leaf_node = leaf_node[good_mask]

            curr = self._unpack_index(self.parent_depth[curr[:, 0], 0].long())
            self.data.data[(*leaf_node.T,)] += self.data[(*curr.T,)]

        with_child = self.child[:filled].nonzero(as_tuple=False)  # NNC, 4
        with_child_sel = (*with_child.T,)
        self.data.data[with_child_sel] = 0.0

    def _pack_index(self, txyz):
        return txyz[:, 0] * (self.N ** 3) + txyz[:, 1] * (self.N ** 2) + \
               txyz[:, 2] * self.N + txyz[:, 3]

    def _unpack_index(self, flat):
        t = []
        for i in range(3):
            t.append(flat % self.N)
            flat //= self.N
        return torch.stack((flat, t[2], t[1], t[0]), dim=-1)

    def _resize_add_cap(self, cap_needed):
        """
        Helper for increasing capacity
        """
        cap_needed = max(cap_needed, int(self.capacity * (self.geom_resize_fact - 1.0)))
        self.data = nn.Parameter(torch.cat((self.data.data,
                        torch.zeros((cap_needed, *self.data.data.shape[1:]),
                                device=self.data.device)), dim=0))
        self.child.resize_(self.capacity + cap_needed, *self.child.shape[1:])
        self.parent_depth.resize_(self.capacity + cap_needed, *self.parent_depth.shape[1:])

    def _make_val_tensor(self, val):
        val_tensor = torch.tensor(val, dtype=torch.float32,
            device=self.data.device)
        while len(val_tensor.shape) < 2:
            val_tensor = val_tensor[None]
        if val_tensor.shape[-1] == 1:
            val_tensor = val_tensor.expand(-1, self.data_dim).contiguous()
        else:
            assert val_tensor.shape[-1] == self.data_dim
        return val_tensor

    def _all_leaves(self):
        """
        Get all leaves of tree
        """
        return (self.child[:self.n_internal.item()] == 0).nonzero(as_tuple=False)

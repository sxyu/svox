"""
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
import torch
import numpy as np
from torch import nn
from warnings import warn
try:
    import svox.csrc as _C
    if not hasattr(_C, "query_vertical"):
        warn("CUDA extension svox.csrc could not be loaded! " +
             "Operations will be slow " +
             "Please do not import svox in the SVE source directory.")
        _C = None
except:
    _C = None


class _SVEQueryVerticalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, child, indices, offset, invradius, padding_mode_c):
        out = _C.query_vertical(data, child, indices, offset, invradius, padding_mode_c)
        ctx.save_for_backward(child, indices, offset, invradius)
        ctx.padding_mode_c = padding_mode_c
        return out

    @staticmethod
    def backward(ctx, grad_out):
        child, indices, offset, invradius = ctx.saved_tensors

        grad_out = grad_out.contiguous()
        if ctx.needs_input_grad[0]:
            grad_data = _C.query_vertical_backward(
                    child, indices, grad_out,
                    offset, invradius,
                    ctx.padding_mode_c)
        else:
            grad_data = None

        return grad_data, None, None, None, None, None


class N3Tree(nn.Module):
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
        WARNING: nn.Parameters can change due to refinement, if refine returns True
        please re-make any optimizers
        """
        super().__init__()
        assert N >= 2
        assert depth_limit >= 0
        self.N = N
        self.data_dim = data_dim

        if init_refine > 0:
            for i in range(1, init_refine + 1):
                init_reserve += (N ** i) ** 3

        self.register_parameter("data", nn.Parameter(
            torch.zeros(init_reserve, N, N, N, data_dim, device=map_location)))
        self.register_buffer("child", torch.zeros(
            init_reserve, N, N, N, dtype=torch.int32, device=map_location))
        self.register_buffer("parent_depth", torch.zeros(
            init_reserve, 2, dtype=torch.int32, device=map_location))

        self.register_buffer("_n_internal", torch.tensor(1, device=map_location))
        self.register_buffer("max_depth", torch.tensor(0, device=map_location))

        radius = torch.tensor(radius, device=map_location)
        center = torch.tensor(center, device=map_location)
        self.register_buffer("invradius", 0.5 / radius, persistent=False)
        self.register_buffer("offset", 0.5 * (1.0 - center / radius), persistent=False)

        self.depth_limit = depth_limit
        self.geom_resize_fact = geom_resize_fact
        self.padding_mode = padding_mode

        if _C is not None:
            self.padding_mode_c = _C.parse_padding_mode(padding_mode)

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

        if not cuda or _C is None or not self.data.is_cuda:
            warn("Using slow assignment")
            indices = self._transform_coord(indices)

            if self.padding_mode == "zeros":
                outside_mask = ((indices >= 1.0) | (indices < 0.0)).any(dim=-1)
                indices = indices[~outside_mask]

            n_queries, _ = indices.shape
            indices.clamp_(0.0, 1.0 - 1e-10)
            ind = indices.clone()

            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            accum = torch.zeros((n_queries, self.data_dim), dtype=torch.float32,
                                  device=indices.device)
            remain_mask = torch.ones(n_queries, dtype=torch.bool, device=indices.device)
            while remain_mask.any():
                ind_floor = torch.floor(ind[remain_mask] * self.N)
                ind_floor.clamp_max_(self.N - 1)
                sel = (node_ids[remain_mask], *(ind_floor.long().T),)

                deltas = self.child[sel]
                vals = self.data.data[sel]

                nonterm_partial_mask = deltas != 0
                nonterm_mask = torch.zeros(n_queries, dtype=torch.bool, device=indices.device)
                nonterm_mask[remain_mask] = nonterm_partial_mask

                accum[nonterm_mask] += vals[nonterm_partial_mask]

                node_ids[remain_mask] += deltas
                ind[remain_mask] = ind[remain_mask] * self.N - ind_floor

                term_mask = remain_mask & ~nonterm_mask
                vals[~nonterm_partial_mask] = values[term_mask] - accum[term_mask]
                self.data.data[sel] = vals

                remain_mask &= nonterm_mask
        else:
            _C.assign_vertical(self.data, self.child, indices,
                               values,
                               self.offset,
                               self.invradius,
                               self.padding_mode_c)

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

        if not cuda or _C is None or not self.data.is_cuda:
            warn("Using slow query")
            indices = self._transform_coord(indices)

            if self.padding_mode == "zeros":
                outside_mask = ((indices >= 1.0) | (indices < 0.0)).any(dim=-1)
            indices.clamp_(0.0, 1.0 - 1e-10)

            n_queries, _ = indices.shape
            ind = indices.clone()
            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            result = torch.zeros((n_queries, self.data_dim), dtype=torch.float32,
                                  device=indices.device)
            remain_mask = torch.ones(n_queries, dtype=torch.bool, device=indices.device)
            if self.padding_mode == "zeros":
                remain_mask &= ~outside_mask

            while remain_mask.any():
                ind_floor = torch.floor(ind[remain_mask] * self.N)
                ind_floor.clamp_max_(self.N - 1)
                sel = (node_ids[remain_mask], *(ind_floor.long().T),)

                deltas = self.child[sel]
                vals = self.data[sel]

                nonterm_partial_mask = deltas != 0
                nonterm_mask = torch.zeros(n_queries, dtype=torch.bool, device=indices.device)
                nonterm_mask[remain_mask] = nonterm_partial_mask

                result[remain_mask] += vals

                node_ids[remain_mask] += deltas
                ind[remain_mask] = ind[remain_mask] * self.N - ind_floor
                remain_mask = remain_mask & nonterm_mask

            if self.padding_mode == "zeros":
                result[outside_mask] = 0.0

            return result
        else:
            return _SVEQueryVerticalFunction.apply(
                    self.data, self.child, indices,
                    self.offset, self.invradius,
                    self.padding_mode_c)

    # In-place modification helpers
    def randn_(self, mean=0.0, std=1.0):
        """
        Set all values to random normal
        Side effect: pushes values to leaf.
        """
        self._push_to_leaf()
        leaf_node = self._all_leaves()  # NNC, 4
        leaf_node_sel = (*leaf_node.T,)
        self.data.data[leaf_node_sel] = torch.randn_like(
                self.data.data[leaf_node_sel]) * std + mean

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
                prob = torch.pow(1.0 / (self.N ** 3), self.parent_depth[leaf_node[:, 0], 1])
                prob = prob.cpu().numpy().astype(np.float64)
                prob /= prob.sum()
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

            self.child[filled:new_filled] = 0
            self.child[leaf_node_sel] = new_idxs - leaf_node[:, 0].to(torch.int32)
            self.parent_depth[filled:new_filled, 0] = self._pack_index(leaf_node)  # parent
            self.parent_depth[filled:new_filled, 1] = self.parent_depth[
                    leaf_node[:, 0], 1] + 1  # depth
            self.max_depth.fill_(max(self.parent_depth[filled:new_filled, 1].max().item(),
                    self.max_depth.item()))

            self._n_internal += num_nc
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
        filled = self.n_internal
        if filled >= self.capacity:
            self._resize_add_cap(1)
            resized = True

        self.child[filled] = 0
        self.child[intnode_idx, xi, yi, zi] = filled - intnode_idx
        depth = self.parent_depth[intnode_idx, 1] + 1
        self.parent_depth[filled, 0] = self._pack_index(torch.tensor(
            [[intnode_idx, xi, yi, zi]], dtype=torch.int32))[0]
        self.parent_depth[filled, 1] = depth
        self.max_depth = max(self.max_depth, depth)
        self._n_internal += 1
        return resized

    def shrink_to_fit(self):
        """
        Shrink data & buffers to tightly needed fit tree data.
        Will change the nn.Parameter size (data), breaking optimizer!
        """
        new_cap = self.n_internal
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
        return self.n_internal + self.n_leaves

    @property
    def n_internal(self):
        """
        Get number of total internal nodes
        """
        return self._n_internal.item()

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

    def depths(self):
        """
        Get a list of leaf depths in tree,
        in same order as values(), corners().
        Root is at depth 0.
        :return (n_leaves) int32
        """
        leaf_node = self._all_leaves()
        return self.parent_depth[leaf_node[:, 0], 1]

    def corners(self, depth=None):
        """
        Get a list of leaf lower xyz corners in tree,
        in same order as values(), depths().
        :return (n_leaves, 3)
        """
        leaf_node = self._all_leaves()
        corners = self._calc_corners(leaf_node)
        if depth is not None:
            depths = self.parent_depth[leaf_node[:, 0], 1]
            corners = corners[depths == depth]
        return corners

    def savez(self, path):
        data = {
            "data_dim" : self.data_dim,
            "child" : self.child.cpu(),
            "parent_depth" : self.parent_depth.cpu(),
            "n_internal" : self._n_internal.cpu().item(),
            "max_depth" : self.max_depth.cpu().item(),
            "invradius" : self.invradius.cpu().item(),
            "offset" : self.offset.cpu(),
            "depth_limit": self.depth_limit,
            "geom_resize_fact": self.geom_resize_fact,
            "padding_mode": self.padding_mode
        }
        if self.data_dim != 3 and self.data_dim != 4:
            data["data"] = self.data.data.cpu()
        else:
            import imageio
            data_path = osp.splitext(path)[0] + '_data.exr'
            imageio.imwrite(data_path, self.data.data.cpu().reshape(-1,
                self.N ** 2, self.data_dim))
        np.savez_compressed(path, **data)

    def loadz(self, path):
        z = np.load(path)
        device = self.data.data.device
        self.data_dim = int(z["data_dim"])
        self.child = torch.from_numpy(z["child"]).to(device)
        self.N = self.child.shape[-1]
        self.parent_depth = torch.from_numpy(z["parent_depth"]).to(device)
        self._n_internal.fill_(z["n_internal"].item())
        self.max_depth.fill_(z["max_depth"].item())
        self.invradius.fill_(z["invradius"].item())
        self.offset = torch.from_numpy(z["offset"]).to(device)
        self.depth_limit = int(z["depth_limit"])
        self.geom_resize_fact = float(z["geom_resize_fact"])
        self.padding_mode = str(z["padding_mode"])
        if _C is not None:
            self.padding_mode_c = _C.parse_padding_mode(self.padding_mode)
        if self.data_dim != 3 and self.data_dim != 4:
            self.data.data = torch.from_numpy(z["data"]).to(device)
        else:
            import imageio
            data_path = osp.splitext(path)[0] + '_data.exr'
            self.data.data = torch.from_numpy(
                        imageio.imread(data_path).reshape(
                            -1, self.N, self.N, self.N, self.data_dim
                        )
                    ).to(device)

    # Magic
    def __repr__(self):
        return ("svox.N3Tree(N={}, data_dim={}, depth_limit={};" +
                " capacity:{}/{} max_depth:{})").format(
                    self.N, self.data_dim, self.depth_limit,
                    self.n_internal, self.capacity, self.max_depth.item())

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start is None and key.stop is None:
            # Everything
            return self
        elif isinstance(key, int):
            # By channel
            return self.values()[..., key]
        elif isinstance(key, tuple) and len(key) == 3:
            # Use x,y,z format
            return self.get(torch.tensor(key, dtype=torch.float32,
                device=self.data.device)[None])[0]
        elif isinstance(key, torch.Tensor):
            assert key.dim() == 1
            return self.values()[key]
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
        elif isinstance(key, torch.Tensor):
            assert key.dim() == 1
            self._push_to_leaf()
            leaf_node = self._all_leaves()[key]
            leaf_node_sel = (*leaf_node.T,)
            self.data.data[leaf_node_sel] = val
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
        filled = self.n_internal

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


    def _calc_corners(self, nodes):
        """
        Compute lower bbox corners for given nodes
        :nodes (Q, 4)
        :return (Q, 3)
        """
        Q, _ = nodes.shape
        filled = self.n_internal

        curr = nodes.clone()
        mask = torch.ones(Q, device=curr.device, dtype=torch.bool)
        output = torch.zeros(Q, 3)

        while True:
            output[mask] += curr[:, 1:]
            output[mask] /= self.N

            good_mask = curr[:, 0] != 0
            if not good_mask.any():
                break
            mask[mask] = good_mask 

            curr = self._unpack_index(self.parent_depth[curr[good_mask, 0], 0].long())

        return output


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
        return (self.child[:self.n_internal] == 0).nonzero(as_tuple=False)

    def _transform_coord(self, indices):
        return torch.addcmul(self.offset, indices, self.invradius)

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
"""
Sparse voxel N^3 tree
"""

import os.path as osp
import torch
import numpy as np
from torch import nn, autograd
from svox.helpers import _N3TreeView, _get_c_extension
from warnings import warn

_C = _get_c_extension()

class _QueryVerticalFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, child, indices, offset, invradius):
        out = _C.query_vertical(data, child, indices, offset, invradius)
        ctx.save_for_backward(child, indices, offset, invradius)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        child, indices, offset, invradius = ctx.saved_tensors

        grad_out = grad_out.contiguous()
        if ctx.needs_input_grad[0]:
            grad_data = _C.query_vertical_backward(
                    child, indices, grad_out,
                    offset, invradius)
        else:
            grad_data = None

        return grad_data, *((None,) * 4)


class N3Tree(nn.Module):
    """
    PyTorch :math:`N^3`-tree library with CUDA acceleration.
    By :math:`N^3`-tree we mean a 3D tree with branching factor N at each interior node,
    where :math:`N=2` is the familiar octree.

.. warning::
    `nn.Parameters` can change due to refinement. If any refine() call returns True,
    please re-make any optimizers
    """
    def __init__(self, N=2, data_dim=4, depth_limit=10,
            init_reserve=1, init_refine=0, geom_resize_fact=1.5,
            radius=0.5, center=[0.5, 0.5, 0.5], map_location="cpu"):
        """
        Construct N^3 Tree

        :param N: int branching factor N
        :param data_dim: int size of data stored at each leaf
        :param depth_limit: int maximum depth of tree to stop branching/refining
        :param init_reserve: int amount of nodes to reserve initially
        :param init_refine: int number of times to refine entire tree initially
        :param geom_resize_fact: float geometric resizing factor
        :param radius: float 1/2 side length of cube
        :param center: list center of space
        :param map_location: str device to put data

        """
        super().__init__()
        assert N >= 2
        assert depth_limit >= 0
        self.N : int = N
        self.data_dim : int = data_dim

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
        self._last_all_leaves = None

        self.refine(repeats=init_refine)


    # Main accesors
    def set(self, indices, values, cuda=True):
        """
        Set tree values,

        :param indices: torch.Tensor (Q, 3)
        :param values: torch.Tensor (Q, K)
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version.

.. warning::
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
            indices = self.world2tree(indices)

            n_queries, _ = indices.shape
            indices.clamp_(0.0, 1.0 - 1e-10)
            ind = indices.clone()

            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
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

                node_ids[remain_mask] += deltas
                ind[remain_mask] = ind[remain_mask] * self.N - ind_floor

                term_mask = remain_mask & ~nonterm_mask
                vals[~nonterm_partial_mask] = values[term_mask]
                self.data.data[sel] = vals

                remain_mask &= nonterm_mask
        else:
            _C.assign_vertical(self.data, self.child, indices,
                               values,
                               self.offset,
                               self.invradius)

    def forward(self, indices, cuda=True, want_node_ids=False, world=True):
        """
        Get tree values. Differentiable.

        :param indices: :math:`(Q, 3)` the points
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version.
        :param want_node_ids: if true, returns node ID for each query.
        :param world: use world space instead of :math:`[0,1]^3`, default True

        :return: (Q, data_dim), [(Q)]

        """
        assert not indices.requires_grad  # Grad wrt indices not supported
        assert len(indices.shape) == 2

        if not cuda or _C is None or not self.data.is_cuda or want_node_ids:
            if not want_node_ids:
                warn("Using slow query")
            if world:
                indices = self.world2tree(indices)

            indices.clamp_(0.0, 1.0 - 1e-10)

            n_queries, _ = indices.shape
            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            result = torch.empty((n_queries, self.data_dim), dtype=torch.float32,
                                  device=indices.device)
            #  remain_mask = torch.ones(n_queries, dtype=torch.bool, device=indices.device)
            remain_indices = torch.arange(n_queries, dtype=torch.long, device=indices.device)
            ind = indices.clone()

            if want_node_ids:
                subidx = torch.zeros((n_queries, 3), dtype=torch.long, device=indices.device)

            while remain_indices.numel():
                ind *= self.N
                ind_floor = torch.floor(ind)
                ind_floor.clamp_max_(self.N - 1)
                ind -= ind_floor

                sel = (node_ids[remain_indices], *(ind_floor.long().T),)

                deltas = self.child[sel]

                term_mask = deltas == 0
                term_indices = remain_indices[term_mask]

                vals = self.data[sel]
                result[term_indices] = vals[term_mask]
                if want_node_ids:
                    subidx[term_indices] = ind_floor.to(torch.long)[term_mask]

                node_ids[remain_indices] += deltas
                remain_indices = remain_indices[~term_mask]
                ind = ind[~term_mask]

            if want_node_ids:
                txyz = torch.cat([node_ids[:, None], subidx], axis=-1)
                return result, self._pack_index(txyz)

            return result
        else:
            return _QueryVerticalFunction.apply(
                    self.data, self.child, indices,
                    self.offset if world else torch.tensor(
                        [0.0, 0.0, 0.0], device=indices.device),
                    self.invradius if world else 1.0)

    # Special leaf data accessors (Implemented in Python only)
    # Integrated with _N3TreeView
    def values(self, sel=None, diff=False):
        """
        Get a list of all leaf values in tree

        :param sel: leaf selector to choose voxels to sample.
                    If none (default), samples all leaves
        :param diff: if True, the result will be differentiable
                     wrt tree data parameter

        :return: (n_leaves, data_dim)

        """
        leaf_node = self._all_leaves()
        if sel is not None:
            leaf_node = leaf_node[sel]
        leaf_node_sel = (*leaf_node.T,)
        if diff:
            return self.data[leaf_node_sel]
        else:
            return self.data.data[leaf_node_sel]

    def depths(self, sel=None):
        """
        Get a list of leaf depths in tree,
        in same order as values(), corners().
        Root is at depth -1. Any children of
        root will have depth 0, etc.

        :param sel: leaf selector to choose voxels to sample.
                    If none (default), samples all leaves

        :return: (n_leaves) int32

        """
        leaf_node = self._all_leaves()
        if sel is not None:
            leaf_node = leaf_node[sel]
            if leaf_node.ndim == 1:
                leaf_node = leaf_node[None]
        return self.parent_depth[leaf_node[:, 0], 1]

    def lengths(self, sel=None, world=True):
        """
        Get a list of leaf side lenghts in tree,
        in same order as values(), corners(), depths()
        Root is at depth -1. Any children of
        root will have depth 0, etc.

        :param sel: leaf selector to choose voxels to sample.
                    If none (default), samples all leaves
        :param world: whether to return in world coords or :math:`[0,1]^3`

        :return: (n_leaves) float

        """
        lens = 2.0 ** (-self.depths(sel).float() - 1.0)
        if world:
            lens /= self.invradius
        return lens

    def corners(self, sel=None, world=True):
        """
        Get a list of leaf lower xyz corners in tree,
        in same order as values(), depths().

        :param sel: leaf selector to choose voxels to sample.
                    If none (default), samples all leaves
        :param world: whether to return in world coords or :math:`[0,1]^3`

        :return: (n_leaves, 3)
        """
        leaf_node = self._all_leaves()
        if sel is not None:
            leaf_node = leaf_node[sel]
            if leaf_node.ndim == 1:
                leaf_node = leaf_node[None]
        corners = self._calc_corners(leaf_node)
        if world:
            corners = self.tree2world(corners)
        return corners

    def sample(self, n_samples, sel=None, world=True):
        """
        Sample n_samples points in each leaf voxel

        :param n_samples: int samples per voxel
        :param sel: leaf selector to choose voxels to sample.
                    If none (default), samples all leaves
        :param world: whether to return in world coords or :math:`[0,1]^3`

        :return: (n_leaves, n_samples, 3)

        """
        corn = self.corners(sel=sel, world=False)[:, None]
        sz = self.lengths(self=sel, world=False)
        u = torch.rand((corn.shape[0], n_samples, 3)) * sz[:, None, None]
        corn += u
        if world:
            corn = self.tree2world(corn)
        return corn

    def snap(self, indices, world=True):
        """
        Snap indices to lowest corner of corresponding leaf voxel

        :param indices: (B, 3) indices to snap

        :return: (B, 3)

        """
        _, node_ids = self.forward(indices, want_node_ids=True)
        leaf_ids = self._reverse_leaf_search(node_ids)
        return self.corners(sel=leaf_ids, world=world)

    def partial(self, data_sel=None):
        """
        Get partial tree with some of the data dimensions (channels)
        E.g. tree.partial(-1) to get tree with data_dim 1 of last channel only

        :param data_sel: data channel selector, default is all channels

        :return: partial N3Tree (copy)

        """
        sel_indices = torch.arange(self.data_dim)[data_sel]
        if sel_indices.ndim == 0:
            sel_indices = sel_indices.unsqueeze(0)
        t2 = N3Tree(N=self.N, data_dim=sel_indices.numel(),
                depth_limit=self.depth_limit,
                geom_resize_fact=self.geom_resize_fact,
                map_location=self.data.data.device)
        t2.invradius = self.invradius.clone()
        t2.offset = self.offset.clone()
        t2.child = self.child.clone()
        t2.parent_depth = self.parent_depth.clone()
        t2._n_internal = self._n_internal.clone()
        t2.max_depth = self.max_depth.clone()
        t2.data.data = self.data.data[..., sel_indices].contiguous()
        return t2



    # In-place modification helpers
    def randn_(self, mean=0.0, std=1.0):
        """
        Set all values to random normal

        :param mean: normal mean
        :param std: normal std

        """
        leaf_node = self._all_leaves()  # NNC, 4
        leaf_node_sel = (*leaf_node.T,)
        self.data.data[leaf_node_sel] = torch.randn_like(
                self.data.data[leaf_node_sel]) * std + mean

    def clamp_(self, min, max, dim=None):
        leaf_node = self._all_leaves()  # NNC, 4
        if dim is None:
            leaf_node_sel = (*leaf_node.T,)
        else:
            leaf_node_sel = (*leaf_node.T, torch.ones_like(leaf_node[..., 0]) * dim)
        self.data.data[leaf_node_sel] = self.data.data[leaf_node_sel].clamp(min, max)

    # Leaf refinement & memory management methods
    def refine(self, repeats=1, dim=-1, thresh=None, sel=None, max_refine=None):
        """
        Refine each leaf node, optionally filtering by value at dimension 'dim' >= 'thresh'.
        Respects depth_limit.

        :param repeats: int number of times to repeat
        :param dim: dimension to check (used if thresh != None). Can be negative (like -1)
        :param thresh: threshold for dimension.
        :param sel: leaf selector
        :param max_refine: maximum number of leaves to refine.
                        'max_refine' random leaves (without replacement)
                        meeting the threshold are refined in case more leaves
                        satisfy it. Leaves at lower depths are sampled exponentially
                        more often.

        :return: True iff N3Tree.data parameter was resized, requiring
                 optimizer reinitialization if you're using an optimizer

.. warning::
    `nn.Parameters` can change due to refinement. If any refine() call returns True, please re-make any optimizers
        """
        with torch.no_grad():
            resized = False
            for _ in range(repeats):
                filled = self.n_internal
                good_mask = self.child[:filled] == 0
                if thresh is not None:
                    good_mask &= (self.data.data[:filled, ..., dim] >= thresh)

                good_mask &= (self.parent_depth[:filled, -1] < self.depth_limit)[:, None, None, None]

                leaf_node = good_mask.nonzero(as_tuple=False)  # NNC, 4
                if sel is not None:
                    leaf_node = leaf_node[sel]
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
                self.data.data[filled:new_filled] = self.data.data[
                        leaf_node_sel][:, None, None, None]
                self.parent_depth[filled:new_filled, 0] = self._pack_index(leaf_node)  # parent
                self.parent_depth[filled:new_filled, 1] = self.parent_depth[
                        leaf_node[:, 0], 1] + 1  # depth
                self.max_depth.fill_(max(self.parent_depth[filled:new_filled, 1].max().item(),
                        self.max_depth.item()))

                self._n_internal += num_nc
        if repeats > 0:
            self._last_all_leaves = None
        return resized

    def _refine_at(self, intnode_idx, xyzi):
        """
        Advanced: refine specific leaf node. Mostly for testing purposes.

        :param intnode_idx: index of internal node for identifying leaf
        :param xyzi: tuple of size 3 with each element in {0, ... N-1}
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
        self.data.data[filled, :, :, :] = self.data.data[intnode_idx, xi, yi, zi]
        self.data.data[intnode_idx, xi, yi, zi] = 0
        self.max_depth = max(self.max_depth, depth)
        self._n_internal += 1
        self._last_all_leaves = None
        return resized

    def shrink_to_fit(self):
        """
        Shrink data & buffers to tightly needed fit tree data.

.. warning::
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
        return self._all_leaves().shape[0]

    @property
    def n_internal(self):
        return self._n_internal.item()

    @property
    def capacity(self):
        return self.parent_depth.shape[0]

    # Persistence
    def save(self, path):
        """
        Save to from npz file

        :param path: npz path

        """
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
            "data": self.data.data.cpu().numpy().astype(np.float16)
        }
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path, map_location='cpu'):
        """
        Load from npz file

        :param path: npz path
        :param map_location: device to put data

        """
        tree = cls(map_location=map_location)
        z = np.load(path)
        tree.data_dim = int(z["data_dim"])
        tree.child = torch.from_numpy(z["child"]).to(map_location)
        tree.N = tree.child.shape[-1]
        tree.parent_depth = torch.from_numpy(z["parent_depth"]).to(map_location)
        tree._n_internal.fill_(z["n_internal"].item())
        tree.max_depth.fill_(z["max_depth"].item())
        tree.invradius.fill_(z["invradius"].item())
        tree.offset = torch.from_numpy(z["offset"].astype(np.float32)).to(map_location)
        tree.depth_limit = int(z["depth_limit"])
        tree.geom_resize_fact = float(z["geom_resize_fact"])
        tree.data.data = torch.from_numpy(z["data"].astype(np.float32)).to(map_location)
        return tree

    # Magic
    def __repr__(self):
        return ("svox.N3Tree(N={}, data_dim={}, depth_limit={};" +
                " capacity:{}/{} max_depth:{})").format(
                    self.N, self.data_dim, self.depth_limit,
                    self.n_internal, self.capacity, self.max_depth.item())

    def __getitem__(self, key):
        """
        Indexing
        """
        if isinstance(key, tuple) and len(key) == 3:
            # Use x,y,z format
            return self.forward(torch.tensor(key, dtype=torch.float32,
                device=self.data.device)[None])[0]
        else:
            return _N3TreeView(self, key)

    def __setitem__(self, key, val):
        if isinstance(key, tuple) and len(key) == 3:
            # Use x,y,z format
            key_tensor = torch.tensor(key, dtype=torch.float32,
                device=self.data.device)[None]
            self.set(key_tensor, self._make_val_tensor(val))
        else:
            _N3TreeView(self, key).set(val)


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
    def _calc_corners(self, nodes):
        Q, _ = nodes.shape
        filled = self.n_internal

        curr = nodes.clone()
        mask = torch.ones(Q, device=curr.device, dtype=torch.bool)
        output = torch.zeros(Q, 3, device=curr.device, dtype=torch.float32)

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
        if self._last_all_leaves is None:
            self._last_all_leaves = (self.child[
                :self.n_internal] == 0).nonzero(as_tuple=False)
        return self._last_all_leaves

    def _reverse_leaf_search(self, target):
        packed_sorted = self._pack_index(self._all_leaves())
        return torch.searchsorted(packed_sorted, target)

    def world2tree(self, indices):
        """
        Scale world points to tree (:math:`[0,1]^3`)
        """
        return torch.addcmul(self.offset, indices, self.invradius)

    def tree2world(self, indices):
        """
        Scale tree points (:math:`[0,1]^3`) to world accoording to center/radius
        """
        return (indices  - self.offset) / self.invradius

    # Tensor analogy
    @property
    def shape(self):
        return torch.Size((self.n_leaves, self.data_dim))

    @property
    def ndim(self):
        return 2

    def dim(self):
        return 2

    def numel(self):
        return self.n_leaves * self.data_dim

    def __len__(self):
        return self.n_leaves

    def size(self, *args):
        if len(args) == 1:
            return self.shape[args[0]]
        return self.shape
